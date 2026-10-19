#!/usr/bin/env bash
# Launch X5 Play Viz (MuJoCo).
# Usage:
#   bash run_viz.sh --demo
#   bash run_viz.sh --bus
#   bash run_viz.sh --demo --headless --steps 60
set -eo pipefail
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${TOOL_DIR}${PYTHONPATH:+:$PYTHONPATH}"
# Wayland / headless GL fallbacks — uncomment if viewer fails to open:
# export MUJOCO_GL="${MUJOCO_GL:-glfw}"
exec python3 -m x5_play_viz "$@"
