#!/bin/bash
# Startup script for a Compute Engine GPU VM (Deep Learning VM image, e.g. family
# pytorch-*-ubuntu-2204-nvidia-*). One-shot run: fetch the notebook from a bucket,
# execute it on the GPU, upload the executed notebook and this log, then power off.
#
# Before use: set BUCKET_DIR below, upload bag_counter_gpu.ipynb to $BUCKET_DIR/code/,
# and set BUCKET / RESULTS_BUCKET inside the notebook. Pass this file as the VM's
# startup-script metadata. The VM's service account needs read/write on the bucket.
export HOME=/root
LOG=/var/log/bag-counter.log
BUCKET_DIR="gs://YOUR_BUCKET/bag-counter"      # <-- change this
STAMP=$(date -u +%Y-%m-%d_%H%M)
exec > >(tee -a "$LOG") 2>&1
echo "[bag] start $(date -u)"

main() {
  set -e
  for i in $(seq 90); do nvidia-smi >/dev/null 2>&1 && break; echo "[bag] waiting for GPU driver ($i)"; sleep 10; done
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "[bag] tools: $(command -v gsutil gcloud zip python3 python 2>&1 | tr '\n' ' ')"; ls /opt
  mkdir -p /opt/bag && cd /opt/bag
  gsutil -q cp "$BUCKET_DIR/code/bag_counter_gpu.ipynb" .

  PYBASE=""
  for c in /opt/conda/bin/python3 /opt/conda/bin/python /usr/local/bin/python3 /usr/bin/python3; do
    if [ -x "$c" ] && "$c" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then PYBASE=$c; break; fi
  done
  echo "[bag] python with CUDA torch: ${PYBASE:-none found, will install torch}"
  apt-get -o DPkg::Lock::Timeout=300 update -q
  apt-get -o DPkg::Lock::Timeout=300 install -y -q zip python3-venv libgl1 libglib2.0-0
  if [ -n "$PYBASE" ]; then
    "$PYBASE" -m venv --system-site-packages venv || { apt-get -o DPkg::Lock::Timeout=300 install -y -q python3-venv; "$PYBASE" -m venv --system-site-packages venv; }
  else
    python3 -m venv venv || { apt-get -o DPkg::Lock::Timeout=300 install -y -q python3-venv; python3 -m venv venv; }
    venv/bin/pip install -q torch torchvision
  fi
  export PATH=/opt/bag/venv/bin:$PATH
  pip install -q ultralytics lap nbconvert ipykernel
  python -c "import torch, cv2, ultralytics; print('[bag] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cv2', cv2.__version__, 'ultralytics', ultralytics.__version__)"

  ( while sleep 60; do echo "[bag] hb $(date -u +%T) snapshots=$(ls /opt/bag/results/*/snapshots 2>/dev/null | wc -l) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
  HB=$!
  echo "[bag] executing notebook $(date -u +%T)"
  set +e
  pip list 2>/dev/null | grep -iE "^(nbconvert|nbclient|nbformat|jupyter|ipykernel)" | tr '\n' ' '; echo
  cat > run_nb.py <<'PYEOF'
import sys, nbformat
from nbclient import NotebookClient
nb = nbformat.read("bag_counter_gpu.ipynb", as_version=4)
rc = 0
try:
    NotebookClient(nb, timeout=None, kernel_name="python3").execute()
except Exception as e:
    rc = 1
    print("[bag] notebook error:", repr(e)[:3000])
finally:
    nbformat.write(nb, "executed_bag_counter_gpu.ipynb")
sys.exit(rc)
PYEOF
  python run_nb.py
  NB=$?
  kill $HB
  set -e
  echo "[bag] notebook finished rc=$NB $(date -u +%T)"
  gsutil -q cp executed_bag_counter_gpu.ipynb "$BUCKET_DIR/results/gpu-run-$STAMP/" || true
  cat results/totals.csv
  [ "$NB" = 0 ] && grep -q "mp4" results/totals.csv
}

( main ); RC=$?
echo "[bag] DONE rc=$RC $(date -u)"
gsutil -q cp "$LOG" "$BUCKET_DIR/results/gpu-run-$STAMP/run-log.txt"
# On failure stay up for 20 minutes so a fixed script can be retried with a reset
# (stopping would give up the GPU, which may not be available again).
[ "$RC" = 0 ] || { echo "[bag] failed; powering off in 20 minutes unless reset"; sleep 1200; }
poweroff
