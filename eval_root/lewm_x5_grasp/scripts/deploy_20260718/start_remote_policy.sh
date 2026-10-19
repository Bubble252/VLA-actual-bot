#!/usr/bin/env bash
# 在 115.190.90.101 上启动指定模型的 Policy Server。
# Usage:
#   MODEL=3p CKPT_STEP=10000 bash start_remote_policy.sh
#   MODEL=1p CKPT_STEP=mid   bash start_remote_policy.sh
#   MODEL=2p                 bash start_remote_policy.sh
#   FORCE_RESTART=1 MODEL=3p CKPT_STEP=5000 bash start_remote_policy.sh
# 换权重推荐直接用: switch_remote_policy.sh（内部 FORCE_RESTART=1）
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/models.env"

SERVER_USER="${SERVER_USER:-root}"
SERVER_HOST="${SERVER_HOST:-115.190.90.101}"
SERVER_PORT="${SERVER_PORT:-27219}"
REMOTE_STARVLA_ROOT="${REMOTE_STARVLA_ROOT:-/root/code/Starvla}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/starvla_cu124/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
FORCE_RESTART="${FORCE_RESTART:-0}"

SSH=(ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=20 "${SERVER_USER}@${SERVER_HOST}")

log() { echo "[start-remote] $*"; }

# Resolve RUN_DIR prefix (e.g. .../20260721-1616 → unique match) and CKPT_STEP=latest on remote.
resolve_remote_ckpt() {
  local resolved
  resolved="$("${SSH[@]}" \
    "CKPT_ROOT='${CKPT_ROOT}' RUN_DIR='${RUN_DIR}' CKPT_STEP='${CKPT_STEP}' CKPT_PATH='${CKPT_PATH}' bash -lc '
      set -e
      resolve_run_dir() {
        local d=\"\$1\"
        if [[ -d \"\$d\" ]]; then echo \"\$d\"; return 0; fi
        # prefix / glob match under Checkpoints
        local matches
        matches=( \$d* )
        if [[ \${#matches[@]} -eq 1 && -d \"\${matches[0]}\" ]]; then
          echo \"\${matches[0]}\"
          return 0
        fi
        if [[ \${#matches[@]} -gt 1 ]]; then
          echo \"ERROR: RUN_DIR prefix matches multiple dirs:\" >&2
          printf \"  %s\\n\" \"\${matches[@]}\" >&2
          return 1
        fi
        echo \"ERROR: RUN_DIR not found: \$d\" >&2
        ls -1d \"\${CKPT_ROOT}\"/20260721* 2>/dev/null | head -20 >&2 || true
        return 1
      }
      RD=\$(resolve_run_dir \"\$RUN_DIR\")
      CKPT_DIR=\"\${RD}/checkpoints\"
      if [[ ! -d \"\$CKPT_DIR\" ]]; then
        echo \"ERROR: no checkpoints/ under \$RD\" >&2
        ls -la \"\$RD\" >&2 || true
        exit 1
      fi
      STEP=\"\$CKPT_STEP\"
      if [[ \"\$STEP\" == \"latest\" || \"\$STEP\" == \"last\" || \"\$STEP\" == \"LATEST\" ]]; then
        STEP=\$(ls -1 \"\$CKPT_DIR\"/steps_*_pytorch_model.pt 2>/dev/null \
          | sed -n \"s|.*/steps_\\([0-9]*\\)_pytorch_model.pt|\\1|p\" \
          | sort -n | tail -1)
        if [[ -z \"\$STEP\" ]]; then
          echo \"ERROR: no steps_*_pytorch_model.pt under \$CKPT_DIR\" >&2
          ls -la \"\$CKPT_DIR\" >&2 || true
          exit 1
        fi
      fi
      PT=\"\${CKPT_DIR}/steps_\${STEP}_pytorch_model.pt\"
      if [[ ! -f \"\$PT\" ]]; then
        echo \"ERROR: missing \$PT\" >&2
        ls -1 \"\$CKPT_DIR\"/steps_*_pytorch_model.pt 2>/dev/null | tail -20 >&2 || true
        exit 1
      fi
      printf \"%s\\n%s\\n%s\\n\" \"\$RD\" \"\$STEP\" \"\$PT\"
    '")" || return 1
  local rd step pt
  rd="$(printf '%s\n' "$resolved" | sed -n '1p')"
  step="$(printf '%s\n' "$resolved" | sed -n '2p')"
  pt="$(printf '%s\n' "$resolved" | sed -n '3p')"
  if [[ -z "$rd" || -z "$step" || -z "$pt" ]]; then
    log "ERROR: failed to resolve remote checkpoint"
    return 1
  fi
  export RUN_DIR="$rd"
  export CKPT_STEP="$step"
  export CKPT_PATH="$pt"
  log "resolved RUN_DIR=${RUN_DIR}"
  log "resolved CKPT_STEP=${CKPT_STEP}"
  log "resolved CKPT_PATH=${CKPT_PATH}"
}

resolve_remote_ckpt || exit 1

stop_remote_policy() {
  log "停止 port ${POLICY_PORT} 上旧 Policy Server..."
  "${SSH[@]}" \
    "PORT='${POLICY_PORT}' STARVLA_ROOT='${REMOTE_STARVLA_ROOT}' bash -lc '
      set -e
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
      ss -tlnp 2>/dev/null | grep \":\${PORT} \" || true
      exit 1
    '"
}

need_start=1
if [[ "${FORCE_RESTART}" == "1" || "${FORCE_RESTART}" == "true" || "${FORCE_RESTART}" == "yes" ]]; then
  if "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
    stop_remote_policy
  else
    log "port ${POLICY_PORT} 空闲，无需停止"
  fi
elif "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
  log "port ${POLICY_PORT} 已在监听，跳过启动（换权重请用: MODEL=${MODEL} CKPT_STEP=... bash switch_remote_policy.sh）"
  need_start=0
fi

if [[ "${need_start}" == "1" ]]; then
  log "启动 Policy Server MODEL=${MODEL} step=${CKPT_STEP} port=${POLICY_PORT} gpu=${GPU_IDS}"
  log "CKPT=${CKPT_PATH}"
  "${SSH[@]}" \
    "STARVLA_ROOT='${REMOTE_STARVLA_ROOT}' CKPT_PATH='${CKPT_PATH}' UNNORM_KEY='${UNNORM_KEY}' BASE_PORT='${POLICY_PORT}' GPU_IDS='${GPU_IDS}' star_vla_python='${REMOTE_PYTHON}' bash -lc '
      set -e
      export PYTHONPATH=\"\${STARVLA_ROOT}:\${PYTHONPATH:-}\"
      mkdir -p \"\${STARVLA_ROOT}/logs\"
      cd \"\${STARVLA_ROOT}\"
      : > \"logs/policy_server_\${BASE_PORT}.log\"
      nohup env CUDA_VISIBLE_DEVICES=\${GPU_IDS} \${star_vla_python} deployment/model_server/server_policy.py \
        --ckpt_path \"\${CKPT_PATH}\" \
        --port \"\${BASE_PORT}\" \
        --unnorm_key \"\${UNNORM_KEY}\" \
        --use_bf16 \
        --idle_timeout -1 \
        > \"logs/policy_server_\${BASE_PORT}.log\" 2>&1 < /dev/null &
      echo \$! > /tmp/starvla_policy_\${BASE_PORT}.pid
      echo started_pid=\$(cat /tmp/starvla_policy_\${BASE_PORT}.pid)
    '"
fi

log "等待 server running (port ${POLICY_PORT})..."
ready_mark="steps_${CKPT_STEP}"
for ((i = 1; i <= 180; i++)); do
  if "${SSH[@]}" "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} ' && grep -q 'server running' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null"; then
    log "就绪: ws://${SERVER_HOST}:${POLICY_PORT}  MODEL=${MODEL} step=${CKPT_STEP} unnorm=${UNNORM_KEY}"
    "${SSH[@]}" "grep -E 'ckpt_path|server running|unnorm' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log | tail -8" || true
    if ! "${SSH[@]}" "grep -q '${ready_mark}' ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null"; then
      log "WARN: 日志里未看到 ${ready_mark}，请人工确认 ckpt_path"
    fi
    exit 0
  fi
  if (( i % 15 == 0 )); then
    log "仍在加载... (${i}/180)"
    "${SSH[@]}" "tail -3 ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log 2>/dev/null" || true
  fi
  sleep 2
done

log "ERROR: 超时。查看: ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} tail -80 ${REMOTE_STARVLA_ROOT}/logs/policy_server_${POLICY_PORT}.log"
exit 1
