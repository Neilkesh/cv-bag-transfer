# Model and settings

Nothing is trained in this project. The counter uses a stock person detector
plus a small per-camera settings file.

| Piece | What it is |
|---|---|
| `yolo11s.pt` | Ultralytics YOLO11-small, the general COCO model (only the `person` class is used). Downloaded automatically on first run; not stored in this repository. Tested with `ultralytics` 8.4.215. |
| `../main_godown_zone.json` | Example camera file: tailgate zone, gate line, plank share, and the two bag-shape thresholds. |
| `../bag_counter.py` | The tracking pass and the counting rule. |

Run settings used for the reported result: `--imgsz 1280 --conf 0.25 --sample-fps 6 --min-gap 8`,
tracker ByteTrack (Ultralytics default).

To pin the detector, keep a copy of `yolo11s.pt` next to `bag_counter.py`, or
pass `--model path/to/yolo11s.pt`.
