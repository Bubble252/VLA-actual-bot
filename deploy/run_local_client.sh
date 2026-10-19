#!/usr/bin/env bash
# 本机 client：按 arxx5_ljn_pick_stage2_90k 配方固定参数。
#
# 与 0813 run_local_client.sh 的关键差异：
#   --state-history-len 2        (0813: 16)
#   --no-normalize-state          (0813: --normalize-state；本次 state 不归一化)
#   --state-gripper-mode openpi   (0813: sign [+g,-g]；本次 [pad=0, gripper=closedness])
#   --gripper-mode continuous_openpi (0813: binary 阈值化)
#   --gripper-qpos-open/closed    (continuous 模式线性映射；默认 arx_x5_ljn)
#   不传 --state-stats-json       (state 不归一化，无需 stats)
#   不传 --gripper-open/close      (continuous 模式不用二值)
#
# Usage:
#   bash run_local_client.sh --dry-run --max-steps 20 --task "pick and place the can"
#   CONTROL_HZ=10 bash run_local_client.sh --max-steps 300 --task "pick and place the can" \
#     --rtc-mode track_b
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

ARX_X5_ROOT="${ARX_X5_ROOT:-${EVAL_ROOT}}"
STARVLA_ROOT="${STARVLA_ROOT}"
export ARX_X5_ROOT STARVLA_ROOT
export POLICY_HOST POLICY_PORT

# 本配方专用 params（frame_align=none, scale=1.0, remap=off）；勿改全局默认 YAML
export X5_PLAY_PARAMS="${X5_PLAY_PARAMS:-${SCRIPT_DIR}/x5_play_params.yaml}"

# RTC client 包：GETDATA_ROOT/RTC/rtc_arx
if [[ -d "${GETDATA_ROOT}/RTC/rtc_arx" ]]; then
  export RTC_ROOT="${RTC_ROOT:-${GETDATA_ROOT}/RTC}"
  export PYTHONPATH="${RTC_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Play-viz 旁路（若传 --play-viz）
_PLAY_VIZ_TOOL="${GETDATA_ROOT}/lewm_x5_grasp/scripts/tool"
if [[ -d "${_PLAY_VIZ_TOOL}/x5_play_viz" ]]; then
  export PYTHONPATH="${_PLAY_VIZ_TOOL}${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ ! -d "${ARX_X5_ROOT}/lewm_x5_grasp" ]]; then
  echo "ERROR: EVAL_ROOT/ARX_X5_ROOT 无效: ${ARX_X5_ROOT}" >&2
  exit 1
fi

EXTRA=(
  --unnorm-key "${UNNORM_KEY}"
  --action-space "${ACTION_SPACE}"
  --control-hz "${CONTROL_HZ}"
  --frame-align "${FRAME_ALIGN}"
  --pos-scale "${POS_SCALE}"
  --rot-scale "${ROT_SCALE}"
  --send-state
  --send-eef-state
  --eef-state-8d
  --eef-state-source "${EEF_STATE_SOURCE:-eef_link_fk}"
  # ⚠️ 与 0813 不同：T=2（不是 16），state 不归一化，state gripper 用 openpi closedness
  --state-history-len "${STATE_HISTORY_LEN:-2}"
  --no-normalize-state
  --state-gripper-mode openpi
  # ⚠️ 与 0813 不同：连续夹爪（不是 binary 阈值化）
  --gripper-mode "${GRIPPER_MODE:-continuous_openpi}"
  --gripper-qpos-open "${GRIPPER_QPOS_OPEN:-4.491874016821384}"
  --gripper-qpos-closed "${GRIPPER_QPOS_CLOSED:--0.06599569314857945}"
  # ⚠️ 与 0813 不同：模型 rot_delta 是 rotvec 语义（训练 action=state rotvec 差分），
  #    必须转换后再进 IK（元素级欧拉加在 yaw≈90° 会把 roll 执行成 pitch → 姿态漂移）
  --rot-delta-semantics "${ROT_DELTA_SEMANTICS:-rotvec}"
  # IK Jacobian 用 eef_link tip（训练对齐点），不用 0813 的 link6 法兰
  --ik-fk "${IK_FK:-eef_link}"
)

# 可选：录制轨迹+视频（RECORD_DIR 非空则开启）
#   RECORD_DIR=./rollouts bash run_local_client.sh ...   # 存到 ./rollouts/<时间戳>/
if [[ -n "${RECORD_DIR:-}" ]]; then
  EXTRA+=(--record-dir "${RECORD_DIR}")
  [[ -n "${RECORD_FPS:-}" ]] && EXTRA+=(--record-fps "${RECORD_FPS}")
  echo "[local-client] RECORD → ${RECORD_DIR}/<timestamp>/ (cam avi + traj.jsonl)"
fi

# 默认带桌面相机（与训练 cam_high + cam_wrist 对齐）
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

echo "[local-client] ws://${POLICY_HOST}:${POLICY_PORT} unnorm=${UNNORM_KEY} space=${ACTION_SPACE} hz=${CONTROL_HZ}"
echo "[local-client] frame_align=${FRAME_ALIGN} pos_scale=${POS_SCALE} rot_scale=${ROT_SCALE}"
echo "[local-client] eef-state-8d + state-history-len=${STATE_HISTORY_LEN:-2} + no-normalize + state-gripper=openpi"
echo "[local-client] gripper-mode=${GRIPPER_MODE:-continuous_openpi} qpos_open=${GRIPPER_QPOS_OPEN:-4.492} qpos_closed=${GRIPPER_QPOS_CLOSED:--0.066}"
echo "[local-client] X5_PLAY_PARAMS=${X5_PLAY_PARAMS}"

exec bash "${ARX_X5_ROOT}/lewm_x5_grasp/scripts/run_eval_starvla_libero_x5.sh" \
  "${EXTRA[@]}" \
  "${USER_ARGS[@]}"
