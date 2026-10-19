#!/usr/bin/env bash
# 在 101 上启动 arx_x5_ljn_pick_stage2_90k Policy Server。
# 与 0813 的差异：端口 5697 / unnorm_key=franka / gpu=1 / 必须 STARVLA_DEPLOY=1（绕过 video metadata）
# Usage:
#   bash start_remote_policy.sh
#   CKPT_STEP=60000 FORCE_RESTART=1 bash start_remote_policy.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

# ⚠️ 部署必须设 STARVLA_DEPLOY=1：让 ArxX5RealLjnDataConfig.transform() 走干净分支（无 video aug），
# 否则 PolicyNormProcessor 构造报 "Video key primary_image not found"。
export STARVLA_DEPLOY=1

FORCE_RESTART="${FORCE_RESTART:-0}"
SSH=(ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=20 "${SERVER_USER}@${SERVER_HOST}")

log() { echo "[start-remote] $*"; }

# 确保 SpikeBrain 路径 + HF remote-code 缓存可用（复用本目录 ensure_vlm_path_on_101.sh）
bash "${SCRIPT_DIR}/ensure_vlm_path_on_101.sh"

# 校验 ckpt + 旁路 config/stats（parents[1] = checkpoints/，已软链到 metadata/）
"${SSH[@]}" "set -e
CKPT='${CKPT_PATH}'
if [[ ! -f \"\$CKPT\" ]]; then
  echo \"ERROR: missing \$CKPT\" >&2
  ls -la '${RUN_DIR}/checkpoints/' >&2 || true
  exit 1
fi
RUN=\$(dirname \$(dirname \"\$CKPT\"))
test -f \"\$RUN/config.yaml\"      # checkpoints/config.yaml -> ../metadata/config.yaml
test -f \"\$RUN/dataset_statistics.json\"
echo OK ckpt=\$CKPT run=\$RUN
"

stop_remote_policy() {
  log "停止 port ${POLICY_PORT} 上旧 Policy Server..."
  "${SSH[@]}" "set -e
PORT='${POLICY_PORT}'
STARVLA_ROOT='${REMOTE_STARVLA_ROOT}'
PIDFILE=/tmp/starvla_policy_\${PORT}.pid
LOG=\${STARVLA_ROOT}/logs/policy_server_\${PORT}.log
if [[ -f \"\${PIDFILE}\" ]]; then kill \"\$(cat \"\${PIDFILE}\")\" 2>/dev/null || true; fi
pkill -f \"server_policy.py.*--port \${PORT}\" 2>/dev/null || true
pkill -f \"server_policy.py.*--port=\${PORT}\" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then fuser -k \"\${PORT}/tcp\" 2>/dev/null || true; fi
for i in \$(seq 1 30); do
  if ! ss -tln 2>/dev/null | grep -q \":\${PORT} \"; then
    if [[ -f \"\${LOG}\" ]]; then
      mv -f \"\${LOG}\" \"\${LOG}.prev.\$(date +%Y%m%d_%H%M%S)\" 2>/dev/null || rm -f \"\${LOG}\"
    fi
    rm -f \"\${PIDFILE}\"
    echo \"port \${PORT} free\"
    exit 0
  fi
  sleep 1
done
echo \"ERROR: port \${PORT} still busy\" >&2
exit 1
"
}

need_start=1
if [[ "${FORCE_RESTART}" == "1" || "${FORCE_RESTART}" == "true" || "${FORCE_RESTART}" == "yes" ]]; then
  if "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
    stop_remote_policy
  else
    log "port ${POLICY_PORT} 空闲，无需停止"
  fi
elif "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
  log "port ${POLICY_PORT} 已在监听，跳过启动（换权重请 FORCE_RESTART=1）"
  need_start=0
fi

if [[ "${need_start}" == "1" ]]; then
  log "启动 Policy Server step=${CKPT_STEP} port=${POLICY_PORT} gpu=${GPU_IDS} unnorm=${UNNORM_KEY}"
  log "CKPT=${CKPT_PATH}"
  log "STARVLA_DEPLOY=1 (deploy transform, no video aug)"
  "${SSH[@]}" "set -e
STARVLA_ROOT='${REMOTE_STARVLA_ROOT}'
CKPT_PATH='${CKPT_PATH}'
UNNORM_KEY='${UNNORM_KEY}'
BASE_PORT='${POLICY_PORT}'
GPU_IDS='${GPU_IDS}'
star_vla_python='${REMOTE_PYTHON}'
export STARVLA_DEPLOY=1
export PYTHONPATH=\"\${STARVLA_ROOT}:\${PYTHONPATH:-}\"
mkdir -p \"\${STARVLA_ROOT}/logs\"
cd \"\${STARVLA_ROOT}\"
: > \"logs/policy_server_\${BASE_PORT}.log\"
nohup env STARVLA_DEPLOY=1 CUDA_VISIBLE_DEVICES=\${GPU_IDS} \${star_vla_python} deployment/model_server/server_policy.py \
  --ckpt_path \"\${CKPT_PATH}\" \
  --port \"\${BASE_PORT}\" \
  --unnorm_key \"\${UNNORM_KEY}\" \
  --use_bf16 \
  --idle_timeout -1 \
  > \"logs/policy_server_\${BASE_PORT}.log\" 2>&1 < /dev/null &
echo \$! > /tmp/starvla_policy_\${BASE_PORT}.pid
echo started_pid=\$(cat /tmp/starvla_policy_\${BASE_PORT}.pid)
"
fi

log "等待 server running (port ${POLICY_PORT})..."
for ((i = 1; i <= 180; i++)); do
  if "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} ' && grep -q 'server running' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null"; then
    log "就绪: ws://127.0.0.1:${POLICY_PORT} (via tunnel)  step=${CKPT_STEP} unnorm=${UNNORM_KEY}"
    "${SSH[@]}" "grep -E 'ckpt_path|server running|unnorm|Loading from|PolicyServerWrapper ready|robot_type' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log | tail -16" || true
    exit 0
  fi
  if (( i % 10 == 0 )); then
    if "${SSH[@]}" "grep -q 'Traceback' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null && ! ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
      log "ERROR: server 进程已退出，日志含 Traceback："
      "${SSH[@]}" "tail -40 ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log" || true
      exit 1
    fi
  fi
  if (( i % 15 == 0 )); then
    log "仍在加载... (${i}/180)"
    "${SSH[@]}" "tail -5 ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null" || true
  fi
  sleep 2
done

log "ERROR: 超时。查看: ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} tail -80 ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log"
exit 1
