#!/usr/bin/env bash
# arxx5_ljn_pick_stage2_90k 部署常量（本机 client + 远程 101 policy）
# 本文件位于迁移仓 deploy/；本机路径默认相对仓库根目录，换机不必改路径结构。
# shellcheck disable=SC2034

_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SERVER_USER="${SERVER_USER:-root}"
export SERVER_HOST="${SERVER_HOST:-115.190.90.101}"
export SERVER_PORT="${SERVER_PORT:-27219}"

export REMOTE_STARVLA_ROOT="${REMOTE_STARVLA_ROOT:-/root/code/Starvla}"
export REMOTE_PYTHON="${REMOTE_PYTHON:-/root/starvla_cu124/bin/python}"
export GPU_IDS="${GPU_IDS:-1}"

# ckpt 仍在 101（本仓不含权重）
export RUN_DIR="${RUN_DIR:-/vepfs-mlp2/c20250405/400040/transfer/arx_x5_ljn_pick_stage2_90000_transfer/arx_x5_ljn_pick_stage2_90000_transfer}"
export CKPT_STEP="${CKPT_STEP:-90000}"
export CKPT_PATH="${CKPT_PATH:-${RUN_DIR}/checkpoints/steps_${CKPT_STEP}/pytorch_model.pt}"

export UNNORM_KEY="${UNNORM_KEY:-franka}"
export POLICY_PORT="${POLICY_PORT:-5697}"
export POLICY_HOST="${POLICY_HOST:-127.0.0.1}"

export ACTION_SPACE="${ACTION_SPACE:-eef_via_joint}"
export CONTROL_HZ="${CONTROL_HZ:-20}"
export FRAME_ALIGN="${FRAME_ALIGN:-none}"
export POS_SCALE="${POS_SCALE:-1.0}"
export ROT_SCALE="${ROT_SCALE:-1.0}"
export EEF_STATE_SOURCE="${EEF_STATE_SOURCE:-eef_link_fk}"

export VLM_REAL="${VLM_REAL:-/vepfs-mlp2/c20250405/400040/transfer/starvla/starVLA/spikebrain/SpikeBrain-2.0-VL}"
export VLM_CONFIG_PATH="${VLM_CONFIG_PATH:-/mnt/jfzn/liujn/code/Starvla/Starvla/spikebrain/SpikeBrain-2.0-VL}"

# ---- 本机路径：默认指向本迁移仓 ----
export GETDATA_ROOT="${GETDATA_ROOT:-${_BUNDLE_ROOT}/getdata_root}"
export EVAL_ROOT="${EVAL_ROOT:-${_BUNDLE_ROOT}/eval_root}"
export STARVLA_ROOT="${STARVLA_ROOT:-${_BUNDLE_ROOT}/starVLA}"
export LOCAL_STATS_JSON="${LOCAL_STATS_JSON:-${_BUNDLE_ROOT}/.local/dataset_statistics.json}"

export ACTION_HORIZON="${ACTION_HORIZON:-20}"
export STATE_HISTORY_LEN="${STATE_HISTORY_LEN:-2}"
export GRIPPER_MODE="${GRIPPER_MODE:-continuous_openpi}"

# 旋转语义 / IK 参考点（ljn 必开；勿改成 euler/link6）
export ROT_DELTA_SEMANTICS="${ROT_DELTA_SEMANTICS:-rotvec}"
export IK_FK="${IK_FK:-eef_link}"

echo "[env] BUNDLE=${_BUNDLE_ROOT}"
echo "[env] CKPT_PATH=${CKPT_PATH}"
echo "[env] port=${POLICY_PORT} unnorm=${UNNORM_KEY} space=${ACTION_SPACE} hz=${CONTROL_HZ}"
echo "[env] EVAL_ROOT=${EVAL_ROOT}"
echo "[env] rot_delta_semantics=${ROT_DELTA_SEMANTICS} ik_fk=${IK_FK}"
