#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  run_record_teach.sh [--task "pick up the mug"] [extra args]

Default: replay-record workflow -> LeRobot dated run dir.

Before running:
  Terminal 1:  bash ARX_CAN/arx_can/arx_can1.sh
  Terminal 2:  ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  Terminal 3:  bash scripts/run_collect_session.sh --task "your task"

Flow (default --format lerobot):
  0/Space  start -> drag arm (trajectory only, no camera save)
  s/Enter  stop -> auto replay -> record D405 + joints -> save LeRobot
  d        delete last saved LeRobot episode
  q        quit

Output:
  data/arx_x5_real/runs/YYYY-MM-DD_HHMMSS/  (+ runs/latest symlink)

Legacy manual HDF5:
  bash scripts/run_record_teach.sh --format hdf5 --output-dir data/teach_demos
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
X5_WS="${REPO_ROOT}/ROS2/X5_ws"
PROJECT="${REPO_ROOT}/lewm_x5_grasp"

source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED="${PIP_TRUSTED:-pypi.tuna.tsinghua.edu.cn}"

python3 -c "import cv2; import numpy; assert int(cv2.__version__.split('.')[0]) >= 4" 2>/dev/null || {
  echo "[record] fixing cv2/numpy (install opencv-python-headless>=4.10 for NumPy 2)..."
  python3 -m pip install --user \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    --timeout 120 \
    "opencv-python-headless>=4.10"
}

python3 -c "import pyarrow, pandas" 2>/dev/null || {
  echo "[record] installing pyarrow pandas for LeRobot save..."
  python3 -m pip install --user \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    pyarrow pandas imageio imageio-ffmpeg
}

exec python3 "${PROJECT}/scripts/record_teach.py" \
  --format lerobot \
  --hz 20 \
  --camera-width 480 \
  --camera-height 480 \
  --camera-fps 20 \
  --no-camera-display \
  "$@"
