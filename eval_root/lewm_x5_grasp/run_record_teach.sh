#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  run_record_teach.sh [output_dir] [extra args for record_teach.py]

Single-arm kinesthetic teaching recorder for ARX X5 + D405.

Before running:
  Terminal 1 (sudo):  bash ARX_CAN/arx_can/arx_can1.sh
  Terminal 2:         ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  Terminal 3:         this script

Default output_dir: ./data/teach_demos

Controls in the recorder terminal:
  0 / Space  start episode
  s / Enter  stop and save
  d          delete last episode
  q          quit
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
X5_WS="${REPO_ROOT}/ROS2/X5_ws"
PROJECT="${REPO_ROOT}/lewm_x5_grasp"

OUTPUT_DIR="${1:-${PROJECT}/data/teach_demos}"
shift || true

source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

exec python3 "${PROJECT}/scripts/record_teach.py" \
  --output-dir "${OUTPUT_DIR}" \
  --task "grab the bottle" \
  --hz 20 \
  --format hdf5 \
  "$@"
