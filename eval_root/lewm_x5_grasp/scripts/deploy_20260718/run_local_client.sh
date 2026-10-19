#!/usr/bin/env bash
# 本机（实验室笔记本）按 MODEL 起 eval client。
# Usage:
#   MODEL=3p bash run_local_client.sh --dry-run --max-steps 20 --task "pick and place the can"
#   MODEL=1p bash run_local_client.sh --dry-run --desktop-camera --task "stack the cans"
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARX_X5_ROOT="${ARX_X5_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
# Play-viz package (MuJoCo UI bus) — prefer sibling get-data repo, else local tool/
_PLAY_VIZ_TOOL_CANDIDATES=(
  "${PLAY_VIZ_TOOL_DIR:-}"
  "${HOME}/project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/scripts/tool"
  "${ARX_X5_ROOT}/lewm_x5_grasp/scripts/tool"
)
for _d in "${_PLAY_VIZ_TOOL_CANDIDATES[@]}"; do
  if [[ -n "${_d}" && -d "${_d}/x5_play_viz" ]]; then
    export PYTHONPATH="${_d}${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
done
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/models.env"

export ARX_X5_ROOT
export STARVLA_ROOT="${STARVLA_ROOT:-$HOME/project/starVLA}"
# 公网端口常被安全组拦截；默认走本机隧道 127.0.0.1
# 先: ssh -p 27219 -N -f -L ${POLICY_PORT}:127.0.0.1:${POLICY_PORT} root@115.190.90.101
export POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
export POLICY_PORT UNNORM_KEY ACTION_SPACE CONTROL_HZ FRAME_ALIGN

EXTRA=(
  --unnorm-key "${UNNORM_KEY}"
  --action-space "${ACTION_SPACE}"
  --control-hz "${CONTROL_HZ}"
  --frame-align "${FRAME_ALIGN:-none}"
)
if [[ "${ACTION_SPACE}" == "eef" || "${ACTION_SPACE}" == "eef_via_joint" ]]; then
  EXTRA+=(--pos-scale "${POS_SCALE}" --rot-scale "${ROT_SCALE}")
fi

# Joint / eef_via_joint 默认带桌面相机（与训练 [cam_high,cam_wrist] 对齐）
# 可 export FORCE_NO_DESKTOP=1 关闭
USER_ARGS=("$@")
has_desktop_flag=0
for a in "${USER_ARGS[@]}"; do
  if [[ "$a" == "--desktop-camera" || "$a" == "--no-desktop-camera" ]]; then
    has_desktop_flag=1
  fi
done
if [[ "${FORCE_NO_DESKTOP:-0}" != "1" && "${has_desktop_flag}" -eq 0 ]]; then
  EXTRA+=(--desktop-camera)
fi

echo "[local-client] ws://${POLICY_HOST}:${POLICY_PORT} MODEL=${MODEL} space=${ACTION_SPACE} hz=${CONTROL_HZ} frame_align=${FRAME_ALIGN:-none}"
exec bash "${ARX_X5_ROOT}/lewm_x5_grasp/scripts/run_eval_starvla_libero_x5.sh" \
  "${EXTRA[@]}" \
  "${USER_ARGS[@]}"
