#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  /home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_wrist_click.sh [max_steps]

Runs the wrist-mounted D405 click-to-grasp loop with ROS2 and X5 workspace
environment variables loaded automatically.

Default max_steps: 600
EOF
  exit 0
fi

source /opt/ros/humble/setup.bash

X5_WS=/home/casia/project/ARX_X5_clean/ROS2/X5_ws
PROJECT=/home/casia/project/ARX_X5_clean/lewm_x5_grasp

source "${X5_WS}/install/setup.bash"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

exec /usr/bin/python3 "${PROJECT}/scripts/run_grasp.py" \
  --config "${PROJECT}/config/grasp_task.wrist_click.json" \
  --max-steps "${1:-600}"
