#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
X5_WS="${REPO_ROOT}/ROS2/X5_ws"
PROJECT="${REPO_ROOT}/lewm_x5_grasp"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [dataset_dir] [--episode N] [extra args]

Replay a saved LeRobot episode on the real arm (verify recorded data).

Prerequisites (same as collect):
  Terminal 1: bash ARX_CAN/arx_can/arx_can1.sh
  Terminal 2: ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  Terminal 3: bash scripts/run_replay_lerobot_dataset.sh

Default dataset: data/arx_x5_real/runs/latest

Examples:
  bash scripts/run_replay_lerobot_dataset.sh
  bash scripts/run_replay_lerobot_dataset.sh data/arx_x5_real/runs/2026-07-04_173936 --episode 0
EOF
  exit 0
fi

DATASET="${PROJECT}/data/arx_x5_real/runs/latest"
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  DATASET="${1}"
  shift
fi

source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

exec python3 "${PROJECT}/scripts/replay_lerobot_dataset.py" \
  --dataset "${DATASET}" \
  "$@"
