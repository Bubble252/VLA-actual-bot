#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  run_teleop_record.sh [extra args for teleop_record.py]

U-Arm master -> ARX X5 integrated session: teleop, record, replay menu.

Before running:
  Terminal 1 (sudo):  bash ARX_CAN/arx_can/arx_can1.sh
  Terminal 2:         ros2 launch arx_x5_controller open_single_arm.launch.py
  Terminal 3:         this script

Session keys:
  0/Space  start recording
  s/Enter  save -> shows path -> r/l/n/q menu
  x/Esc    abort recording (discard, ready for Space again)
  r        HDF5 joint replay
  p        EEF replay (mode=4 damped IK)
  v        EEF→joint replay (diff IK→mode=5)
  l        LeRobot   n  new   q  quit

Quick-record (skip replay menu; one key starts next episode):
  bash scripts/run_teleop_record.sh --quick-record
  After save: n / 0 / Space / Enter -> next episode; m -> full menu; q -> quit

Optional: specify an existing HDF5 for comparison replay:
  bash scripts/run_teleop_record.sh --replay-file data/teleop_demos/episode_0.hdf5
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
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export LANG="${LANG:-C.UTF-8}"
export LC_ALL="${LC_ALL:-C.UTF-8}"
export OPENCV_VIDEOIO_PRIORITY_LIST="${OPENCV_VIDEOIO_PRIORITY_LIST:-GTK,FFMPEG}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Kill stale teleop session that may hold RealSense USB open.
if pgrep -f "teleop_record.py" >/dev/null 2>&1; then
  echo "[run_teleop_record] stopping stale teleop_record.py …"
  pkill -f "teleop_record.py" || true
  sleep 2
fi

echo "============================================================"
echo "Data paths (details printed by teleop_record.py):"
echo "  HDF5 + cameras : lewm_x5_grasp/data/teleop_demos/episode_<时间戳>.hdf5"
echo "  LeRobot dataset: lewm_x5_grasp/data/arx_x5_real/runs/<dated>/"
echo "  Convert cmd    : bash scripts/run_convert_to_lerobot.sh data/teleop_demos --fps 30"
echo "============================================================"
echo "Recording keys (focus this terminal):"
echo "  0/Space   start recording"
echo "  s/Enter   stop and save (NOT auto-save)"
echo "  x/Esc     abort current take (discard, ready again)"
echo "  q         quit"
echo "  tip: --quick-record  for one-key next episode after save"
echo "============================================================"

exec python3 "${PROJECT}/scripts/teleop_record.py" \
  --config "${PROJECT}/config/teleop_uarm.example.json" \
  "$@"
