#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_batch_eef_replay.sh [input_dir] [batch_eef_replay.py options]

Batch EEF (mode=4) replay for all episode_*.hdf5 (default: observations/eef + smooth, no bake).

Prerequisites (same as teleop):
  Terminal 1: bash ARX_CAN/arx_can/arx_can1.sh
  Terminal 2: ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  Terminal 3: this script

Examples:
  # Batch EEF replay (default, no bake)
  bash scripts/run_batch_eef_replay.sh data/teleop_demos

  # Optional: bake eef_fk first (~1 min per episode), then replay with eef_source=eef_fk
  bash scripts/run_batch_eef_replay.sh data/teleop_demos --phase both --force-rebake

  # Bake only (skips files that already have eef_fk)
  bash scripts/run_batch_eef_replay.sh data/teleop_demos --phase bake

  # Pause for Enter between episodes
  bash scripts/run_batch_eef_replay.sh data/teleop_demos --step

  # List episodes without moving arm
  bash scripts/run_batch_eef_replay.sh data/teleop_demos --dry-run
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

INPUT_DIR="${PROJECT}/data/teleop_demos"
EXTRA_ARGS=()
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  INPUT_DIR="${1}"
  shift
fi
EXTRA_ARGS+=("$@")

exec python3 "${PROJECT}/scripts/batch_eef_replay.py" \
  "${INPUT_DIR}" \
  --config "${PROJECT}/config/teleop_uarm.example.json" \
  "${EXTRA_ARGS[@]}"
