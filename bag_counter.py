"""
Bag Counter - counts bags carried to a truck from CCTV video.

How it works
  1. Every person is detected and tracked (YOLO + ByteTrack). Their positions
     are saved to tracks.csv. This is the slow part and wants a GPU.
  2. A "gate" line is drawn across the path to the truck (over the plank and
     the ground beside it). Each time a person's feet cross the gate TOWARDS
     the truck, the picture is checked for a sack on their head: the shape
     above the head is measured, and a sack is much wider than a head.
       - wide shape, came up the plank        -> BAG
       - borderline shape, or came by ground  -> UNCLEAR (counted, flagged)
       - narrow shape                          -> no bag (not counted, logged)
     People walking back from the truck are ignored.

Usage
  1) Describe the camera view once (zone = where the truck's tailgate is,
     gate = line across the path to it):
       python bag_counter.py setup --video V.mp4 --zone cam.json \\
              --points "500,110 585,118 585,152 503,145" --gate "570,115 570,185"
     Without --points you click the zone corners in a window (needs a screen).
     Check cam_preview.jpg.

  2) Count:
       python bag_counter.py run --video V.mp4 --zone cam.json
     Output folder <video name>_results/: summary.txt, events.csv, snapshots/,
     no_bag.csv + no_bag/ (crossings rejected as "no bag"), tracks.csv.

  3) Changed the gate or thresholds? Re-count in minutes without the GPU:
       python bag_counter.py recount --video V.mp4 --zone cam.json
"""
import argparse, collections, csv, json, os, re, sys, time
from datetime import datetime, timedelta

import cv2
import numpy as np

DEFAULTS = {"plank_part": 0.62,      # first part of the gate (from its 1st point) is the plank, the rest is ground
            "bag_width_sure": 30,    # px (at the frame_size in the file): shape above the head this wide = bag
            "bag_width_min": 22}     # narrower than this = no bag


# ----------------------------------------------------------------- helpers
def open_video(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}\nIf it's H.265 from the NVR, convert first:\n"
                 f'  ffmpeg -i "{video}" -c:v libx264 -crf 23 -preset veryfast converted.mp4')
    return cap


def frame_at(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000 - 5)
    ok, frame = cap.read()
    return frame if ok else None


def clock_from_filename(path):
    """Hikvision-style names like Main_Godown_2026-11-07_11-50-00.mp4 -> datetime."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", os.path.basename(path))
    if m:
        return datetime.strptime(m.group(1) + " " + m.group(2), "%Y-%m-%d %H-%M-%S")
    return None


def start_clock(a):
    if a.start_clock:
        return datetime.strptime(a.start_clock, "%Y-%m-%d %H:%M:%S")
    return clock_from_filename(a.video) if a.file_clock else None


def fmt_td(sec):
    return str(timedelta(seconds=int(sec)))


def draw_view(img, cfg, color=(0, 200, 255)):
    """Zone outline (truck tailgate) and the gate line."""
    cv2.polylines(img, [cfg["zone"].reshape(-1, 1, 2)], True, color, 1)
    if cfg.get("gate") is not None:
        g1, g2 = cfg["gate"].astype(int)
        cv2.line(img, tuple(map(int, g1)), tuple(map(int, g2)), (0, 255, 255), 2)


def load_cfg(path, W, H):
    """Read the camera file and scale it to the video's resolution."""
    raw = json.load(open(path))
    sx, sy = W / raw["frame_size"][0], H / raw["frame_size"][1]
    cfg = dict(DEFAULTS, **{k: raw[k] for k in DEFAULTS if k in raw})
    cfg["scale"] = sx
    cfg["zone"] = (np.array(raw["zone"], float) * [sx, sy]).astype(np.int32)
    cfg["gate"] = np.array(raw["gate"], float) * [sx, sy] if raw.get("gate") else None
    return cfg


# ------------------------------------------------------------------- setup
def cmd_setup(a):
    cap = open_video(a.video)
    frame = frame_at(cap, a.at)
    cap.release()
    if frame is None:
        sys.exit("Could not read a frame at that time.")
    h, w = frame.shape[:2]
    if a.points:
        pts = [tuple(map(int, p.split(","))) for p in a.points.split()]
    else:
        pts = []

        def on_click(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

        cv2.namedWindow("Mark loading zone", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mark loading zone", on_click)
        while True:
            disp = frame.copy()
            for p in pts:
                cv2.circle(disp, p, 4, (0, 0, 255), -1)
            if len(pts) >= 2:
                cv2.polylines(disp, [np.array(pts)], len(pts) > 2, (0, 200, 255), 2)
            cv2.putText(disp, "Click corners. ENTER=save  R=redo  ESC=cancel", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Mark loading zone", disp)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 10) and len(pts) >= 3:
                break
            if k in (ord("r"), ord("R")):
                pts.clear()
            if k == 27:
                sys.exit("Cancelled.")
        cv2.destroyAllWindows()

    raw = {"zone": [list(p) for p in pts], "frame_size": [w, h]}
    if a.gate:
        raw["gate"] = [list(map(int, p.split(","))) for p in a.gate.split()]
        if len(raw["gate"]) != 2:
            sys.exit('--gate needs exactly two points, e.g. --gate "570,115 570,185"')
    raw.update(DEFAULTS)
    if a.plank_part is not None:
        raw["plank_part"] = a.plank_part
    raw["note"] = ("zone = truck tailgate. gate = line across the path to the truck; a person whose feet cross it "
                   "towards the zone is checked for a bag. plank_part = share of the gate (from its first point) "
                   "that is the plank; the rest is the ground beside it.")
    with open(a.zone, "w") as f:
        json.dump(raw, f, indent=2)
    preview = frame.copy()
    draw_view(preview, load_cfg(a.zone, w, h))
    prev_path = os.path.splitext(a.zone)[0] + "_preview.jpg"
    cv2.imwrite(prev_path, preview)
    print(f"Saved camera file -> {a.zone}\nPreview          -> {prev_path}")
    if not a.gate:
        print('No gate yet: run setup again with --gate "x,y x,y" (a line across the path to the truck).')


# ------------------------------------------------------------ tracking pass
def track_video(a, cap, cfg, out, fps, total, W, H):
    """Detect and track people; write tracks.csv (one row per person per analysed frame)."""
    from ultralytics import YOLO
    import torch

    duration = total / fps                # approximate; only used for the progress line
    period = 1.0 / a.sample_fps           # seconds of video between analysed frames
    end_t = a.end or float("inf")
    if a.start > 0:                       # seek a little early; the loop skips to the exact time
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, a.start - 5) * 1000)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(a.model)
    print(f"Video {W}x{H} @ {fps:.1f}fps, {total} frames ({fmt_td(total / fps)}). "
          f"Analysing ~{a.sample_fps:g} frames per second of video on {'GPU' if device == 0 else 'CPU'}.")

    writer = None
    if a.save_video:
        writer = cv2.VideoWriter(os.path.join(out, "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), min(a.sample_fps, fps), (W, H))
    tracks_path = os.path.join(out, "tracks.csv")
    tracks_f = open(tracks_path, "w", newline="")
    tracks_w = csv.writer(tracks_f)
    tracks_w.writerow(["t", "track_id", "x1", "y1", "x2", "y2", "conf"])

    t0 = time.time()
    fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    next_t, last_t, n_done = a.start, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Real timestamp of this frame. NVR files have a variable frame rate, so
        # frame number / fps drifts away from the true video time.
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and fidx > 0:           # backend gave no timestamp: fall back to frame number
            t = fidx / fps
        tsec = last_t = max(t, last_t)    # never go backwards
        fidx += 1
        if tsec >= end_t:
            break
        if tsec < next_t:                 # not yet time for the next analysed frame
            continue
        next_t += period * (int((tsec - next_t) / period) + 1)
        n_done += 1

        r = model.track(frame, persist=True, classes=[0], conf=a.conf, imgsz=a.imgsz,
                        tracker="bytetrack.yaml", device=device, verbose=False)[0]
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            for (x1, y1, x2, y2), tid, cf in zip(boxes, ids, r.boxes.conf.cpu().tolist()):
                tracks_w.writerow([f"{tsec:.3f}", tid, f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", f"{cf:.2f}"])
            if writer is not None:
                for (x1, y1, x2, y2), tid in zip(boxes.astype(int), ids):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, str(tid), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if writer is not None:
            draw_view(frame, cfg)
            writer.write(frame)

        if n_done % 200 == 0 and total:
            done = min(1.0, (tsec - a.start) / max(1e-6, min(end_t, duration) - a.start))
            el = time.time() - t0
            eta = el / done - el if done > 0 else 0
            print(f"  {done * 100:5.1f}%  video {fmt_td(tsec)}  ETA {fmt_td(eta)}", flush=True)

    tracks_f.close()
    if writer is not None:
        writer.release()
    print(f"Tracking done in {fmt_td(time.time() - t0)} ({'GPU' if device == 0 else 'CPU'}).")
    return tracks_path


# ------------------------------------------------------------ counting pass
def find_crossings(tracks_path, cfg, min_gap):
    """Every time a tracked person's feet cross the gate towards the truck."""
    g1, g2 = cfg["gate"]
    gv = g2 - g1
    side = lambda p: gv[0] * (p[1] - g1[1]) - gv[1] * (p[0] - g1[0])
    truck = 1.0 if side(cfg["zone"].mean(axis=0)) >= 0 else -1.0

    tracks = collections.defaultdict(list)
    for r in csv.DictReader(open(tracks_path)):
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        tracks[int(r["track_id"])].append((float(r["t"]), (x1, y1, x2, y2)))

    found = []
    for tid, pts in tracks.items():
        last = -1e9
        for (ta, ba), (tb, bb) in zip(pts, pts[1:]):
            if tb - ta > 1.2:                         # the track was lost in between
                continue
            p = np.array([(ba[0] + ba[2]) / 2, ba[3]])   # foot point = bottom centre of the box
            q = np.array([(bb[0] + bb[2]) / 2, bb[3]])
            sp, sq = side(p) * truck, side(q) * truck
            if not (sp < 0 <= sq):                    # not a move onto the truck side
                continue
            hit = p + (q - p) * (sp / (sp - sq))
            u = float(np.dot(hit - g1, gv) / np.dot(gv, gv))   # where along the gate, 0..1
            if 0 <= u <= 1 and tb - last >= min_gap:
                last = tb
                found.append({"t": tb, "track_id": tid, "box": bb,
                              "path": "plank" if u <= cfg["plank_part"] else "ground"})
    return sorted(found, key=lambda e: e["t"])


def bag_width(cap, t, box, scale):
    """Width (px) of the widest thing above the person's head that is not background.

    Background = median of frames a few seconds before and after, when the spot is empty.
    A sack on the head is about three times as wide as a bare head or a hat.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    xa, xb = int(max(0, cx - 36 * scale)), int(cx + 36 * scale)
    ya, yb = int(max(0, y1 - 22 * scale)), int(y1 + 10 * scale)
    frame = frame_at(cap, t)
    if frame is None or yb <= ya or xb <= xa:
        return None, frame
    around = [f[ya:yb, xa:xb] for f in (frame_at(cap, t + dt) for dt in (-9, -6, -3, 3, 6)) if f is not None]
    if len(around) < 3:
        return None, frame
    bg = np.median(np.stack(around), axis=0)
    diff = np.abs(frame[ya:yb, xa:xb].astype(float) - bg).max(axis=2)
    mask = cv2.morphologyEx((diff > 28).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return int(mask.sum(axis=1).max()) / scale, frame


def count_from_tracks(a, cap, cfg, out, tracks_path):
    if cfg["gate"] is None:
        sys.exit('The camera file has no gate. Add one: python bag_counter.py setup ... --gate "x,y x,y"')
    for d in ("snapshots", "no_bag"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
        for f in os.listdir(os.path.join(out, d)):       # clear pictures from an earlier count
            if f.endswith(".jpg"):
                os.remove(os.path.join(out, d, f))
    clock0 = start_clock(a)
    events, rejected = [], []
    t0 = time.time()
    for c in find_crossings(tracks_path, cfg, a.min_gap):
        width, frame = bag_width(cap, c["t"], c["box"], cfg["scale"])
        if frame is None:
            continue
        if width is not None and width < cfg["bag_width_min"]:
            status = "no bag"
        elif width is None or width < cfg["bag_width_sure"] or c["path"] == "ground":
            status = "unclear"
        else:
            status = "bag"
        t = c["t"]
        clock = (clock0 + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if clock0 else ""
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        img = frame.copy()
        draw_view(img, cfg)
        row = {"video_time": fmt_td(t), "clock_time": clock, "path": c["path"],
               "bag_width": "" if width is None else round(width), "track_id": c["track_id"]}
        stamp = fmt_td(t).replace(":", "-")
        if status == "no bag":
            row["snapshot"] = f"nobag_{stamp}_{c['track_id']}.jpg"
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
            cv2.putText(img, f"NO BAG  {clock or fmt_td(t)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.imwrite(os.path.join(out, "no_bag", row["snapshot"]), img)
            rejected.append(row)
        else:
            n = len(events) + 1
            colour = (0, 0, 255) if status == "bag" else (0, 140, 255)
            row = dict(bag_no=n, status=status, **row)
            row["snapshot"] = f"bag_{n:04d}_{stamp}{'_UNCLEAR' if status == 'unclear' else ''}.jpg"
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
            label = f"BAG #{n}" if status == "bag" else f"UNCLEAR #{n}"
            cv2.putText(img, f"{label}  {clock or fmt_td(t)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
            cv2.imwrite(os.path.join(out, "snapshots", row["snapshot"]), img)
            events.append(row)

    with open(os.path.join(out, "events.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["bag_no", "video_time", "clock_time", "status", "path",
                                           "bag_width", "track_id", "snapshot"])
        wr.writeheader()
        wr.writerows(events)
    with open(os.path.join(out, "no_bag.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["video_time", "clock_time", "path", "bag_width", "track_id", "snapshot"])
        wr.writeheader()
        wr.writerows(rejected)

    unclear = [e for e in events if e["status"] == "unclear"]
    summary = (f"Video: {os.path.basename(a.video)}\n"
               f"TOTAL BAGS COUNTED: {len(events)}\n"
               f"  sure: {len(events) - len(unclear)}\n"
               f"  unclear (counted, please check the snapshot): {len(unclear)}\n")
    for e in unclear:
        why = "came by the ground, not the plank" if e["path"] == "ground" else "bag shape borderline"
        summary += f"    #{e['bag_no']} at {e['video_time']}  {why}  -> snapshots/{e['snapshot']}\n"
    summary += (f"People who crossed towards the truck without a bag (not counted): {len(rejected)}"
                f"  -> no_bag.csv, no_bag/\n")
    open(os.path.join(out, "summary.txt"), "w").write(summary)
    print("\n" + summary + f"Counting pass took {fmt_td(time.time() - t0)}.\nResults -> {out}")


# ----------------------------------------------------------------- commands
def video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return fps, total, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def cmd_run(a):
    out = a.out or os.path.splitext(a.video)[0] + "_results"
    os.makedirs(out, exist_ok=True)
    cap = open_video(a.video)
    fps, total, W, H = video_info(cap)
    cfg = load_cfg(a.zone, W, H)
    tracks_path = track_video(a, cap, cfg, out, fps, total, W, H)
    count_from_tracks(a, cap, cfg, out, tracks_path)
    cap.release()


def cmd_recount(a):
    out = a.out or os.path.splitext(a.video)[0] + "_results"
    tracks_path = a.tracks or os.path.join(out, "tracks.csv")
    if not os.path.exists(tracks_path):
        sys.exit(f"No track log at {tracks_path}. Run 'bag_counter.py run' first (it writes tracks.csv).")
    os.makedirs(out, exist_ok=True)
    cap = open_video(a.video)
    fps, total, W, H = video_info(cap)
    count_from_tracks(a, cap, load_cfg(a.zone, W, H), out, tracks_path)
    cap.release()


def add_common(p):
    p.add_argument("--video", required=True)
    p.add_argument("--zone", required=True, help="camera file made by setup")
    p.add_argument("--out", help="output folder")
    p.add_argument("--min-gap", type=float, default=8,
                   help="seconds before the same tracked person can be counted again")
    p.add_argument("--start-clock", help='real time of the video start "YYYY-MM-DD HH:MM:SS"')
    p.add_argument("--file-clock", action="store_true",
                   help="take the start time from the file name (NVR file names are not always exact)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Count bags carried to a truck from CCTV video")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="describe the camera view (zone and gate)")
    s.add_argument("--video", required=True)
    s.add_argument("--zone", default="zone.json", help="where to save the camera file")
    s.add_argument("--at", type=float, default=5, help="take the frame at this second")
    s.add_argument("--points", help='zone corners without clicking: "x,y x,y x,y ..."')
    s.add_argument("--gate", help='line across the path to the truck: "x,y x,y" (plank end first)')
    s.add_argument("--plank-part", type=float, help="share of the gate, from its first point, that is the plank")

    r = sub.add_parser("run", help="track people, then count bags")
    add_common(r)
    r.add_argument("--model", default="yolo11s.pt", help="yolo11n.pt = faster, yolo11m.pt = more accurate")
    r.add_argument("--imgsz", type=int, default=1280, help="detection size; 1280 helps with small far-away people")
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--sample-fps", type=float, default=6, help="frames analysed per second of video")
    r.add_argument("--start", type=float, default=0, help="start at this second")
    r.add_argument("--end", type=float, default=0, help="stop at this second (0 = end)")
    r.add_argument("--save-video", action="store_true", help="also write annotated.mp4")

    c = sub.add_parser("recount", help="count again from an existing tracks.csv (no GPU needed)")
    add_common(c)
    c.add_argument("--tracks", help="track log (default: <out>/tracks.csv)")

    a = p.parse_args()
    {"setup": cmd_setup, "run": cmd_run, "recount": cmd_recount}[a.cmd](a)
