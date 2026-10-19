#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
X5_WS="${REPO_ROOT}/ROS2/X5_ws"
PROJECT="${REPO_ROOT}/lewm_x5_grasp"

DEFAULT_HDF5="/home/casia/project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/data/pour_foam_from_cup/episode_2026-07-14_200904.hdf5"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  run_replay_hdf5_eef_via_joint.sh [hdf5_path] [extra args]

Default mode: eef_via_joint (EEF → differential IK → mode=5).

Examples:
  bash scripts/run_replay_hdf5_eef_via_joint.sh
  bash scripts/run_replay_hdf5_eef_via_joint.sh /path/to/episode.hdf5
  bash scripts/run_replay_hdf5_eef_via_joint.sh --dry-run --max-steps 80
  bash scripts/run_replay_hdf5_eef_via_joint.sh --mode joint
  bash scripts/run_replay_hdf5_eef_via_joint.sh --mode eef   # old mode=4

Before real replay: CAN + ros2 launch open_single_arm
EOF
  exit 0
fi

HDF5_ARGS=()
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  HDF5_ARGS=(--hdf5 "$1")
  shift
else
  HDF5_ARGS=(--hdf5 "${DEFAULT_HDF5}")
fi

# episode_replay imports rclpy at module load; always source ROS.
source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"

export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

cd "${PROJECT}"
exec python3 scripts/replay_hdf5_eef_via_joint.py "${HDF5_ARGS[@]}" "$@"
