#!/usr/bin/env bash
# 采集会话：全程只用 teach launch，record_teach.py 自动切换手拖/回放。
set -eo pipefail

REPO="${REPO:-$HOME/project/ARX_X5_clean}"
X5_WS="${REPO}/ROS2/X5_ws"
PROJECT="${REPO}/lewm_x5_grasp"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
用法: $(basename "$0") [record_teach.py 的额外参数]

全程只需一个控制 launch（示教模式），不用在 normal/teach 之间手动切换。

三个终端 / tmux pane:
  1) CAN:     cd ${REPO}/ARX_CAN/arx_can && bash arx_can1.sh
  2) 控制:    source /opt/ros/humble/setup.bash
              cd ${X5_WS} && source install/setup.bash
              ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  3) 采集:    bash ${PROJECT}/scripts/run_collect_session.sh

模式切换（自动，在 pane 3 由 Python 完成）:
  手拖示教   -> 只记轨迹，不存相机
  按 s 保存  -> 自动回放 (mode=5) + 双相机录制 -> 保存 LeRobot
  默认相机: cam_wrist (D405) + cam_high (D435i 桌面)
  仅腕部:    加参数 --no-desktop-camera
  播完自动回到 mode=3 手拖

LeRobot 输出目录（每次会话新建）:
  data/arx_x5_real/runs/YYYY-MM-DD_HHMMSS/
  data/arx_x5_real/runs/latest  -> 最近一次

首次使用或改了 X5Controller.cpp 后请:
  cd ${X5_WS} && colcon build --packages-select arx_x5_controller
EOF
  exit 0
fi

source /opt/ros/humble/setup.bash
source "${X5_WS}/install/setup.bash"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"

exec bash "${PROJECT}/scripts/run_record_teach.sh" "$@"
