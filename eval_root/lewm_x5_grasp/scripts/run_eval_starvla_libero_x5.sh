#!/usr/bin/env bash
set -eo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  run_eval_starvla_libero_x5.sh [extra args for eval_starvla_libero_x5.py]

Laptop-side client: D405 + ROS2 /arm_cmd, talks to StarVLA policy server.

Prerequisites:
  1. X5 CAN + ros2 launch arx_x5_controller open_single_arm.launch.py
  2. Policy server running on GPU machine (see run_policy_server_on_server.sh)
  3. SSH tunnel on laptop if needed:
       ssh -N -L 5694:127.0.0.1:5694 user@SERVER

Environment overrides:
  STARVLA_ROOT   path to starVLA repo (for ModelClient import)
  POLICY_HOST    default 127.0.0.1
  POLICY_PORT    default 5694
  ARX_X5_ROOT    path to ARX_X5_clean

Safe first run:
  bash run_eval_starvla_libero_x5.sh --dry-run --max-steps 20
EOF
  exit 0
fi

ARX_X5_ROOT="${ARX_X5_ROOT:-$HOME/project/ARX_X5_clean}"
STARVLA_ROOT="${STARVLA_ROOT:-$HOME/project/starVLA}"
X5_WS="${ARX_X5_ROOT}/ROS2/X5_ws"
PROJECT="${ARX_X5_ROOT}/lewm_x5_grasp"

source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"

export STARVLA_ROOT
export PYTHONPATH="${STARVLA_ROOT}:${PROJECT}/src:${PYTHONPATH:-}"

exec python3 "${PROJECT}/scripts/eval_starvla_libero_x5.py" \
  --host "${POLICY_HOST:-127.0.0.1}" \
  --port "${POLICY_PORT:-5694}" \
  --unnorm-key franka \
  "$@"
