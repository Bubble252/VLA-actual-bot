#!/usr/bin/env bash
# Pull teach-collect updates from GPU server to this laptop.
set -eo pipefail

SERVER="${SERVER:-zhangjh@172.18.127.46}"
REMOTE_ROOT="${REMOTE_ROOT:-/9950backfile/zhangjh/wh/ARX_X5_clean}"
DEST="${DEST:-$HOME/project/ARX_X5_clean}"

echo "[sync] ${SERVER}:${REMOTE_ROOT} -> ${DEST}"

rsync -av \
  "${SERVER}:${REMOTE_ROOT}/lewm_x5_grasp/README_TEACH_COLLECT.md" \
  "${SERVER}:${REMOTE_ROOT}/lewm_x5_grasp/requirements.txt" \
  "${SERVER}:${REMOTE_ROOT}/lewm_x5_grasp/scripts/record_teach.py" \
  "${SERVER}:${REMOTE_ROOT}/lewm_x5_grasp/scripts/run_record_teach.sh" \
  "${DEST}/lewm_x5_grasp/"

rsync -av \
  "${SERVER}:${REMOTE_ROOT}/lewm_x5_grasp/src/lewm_x5_grasp/" \
  "${DEST}/lewm_x5_grasp/src/lewm_x5_grasp/"

rsync -av \
  "${SERVER}:${REMOTE_ROOT}/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/config/single_arm_teach.yaml" \
  "${SERVER}:${REMOTE_ROOT}/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/launch/x5_v1/open_single_arm_teach.launch.py" \
  "${DEST}/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/"

echo "[sync] rebuilding arx_x5_controller..."
source /opt/ros/humble/setup.bash
cd "${DEST}/ROS2/X5_ws"
colcon build --packages-select arx_x5_controller
echo "[sync] done. Run: pip install -r ${DEST}/lewm_x5_grasp/requirements.txt (if deps changed)"
