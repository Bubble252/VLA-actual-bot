#!/usr/bin/env bash
# One-shot automation: remote Policy Server + SSH tunnel + local eval client.
set -eo pipefail

SERVER_USER="${SERVER_USER:-zhangjh}"
SERVER_HOST="${SERVER_HOST:-172.18.127.46}"
ARX_X5_ROOT="${ARX_X5_ROOT:-$HOME/project/ARX_X5_clean}"
STARVLA_ROOT="${STARVLA_ROOT:-$HOME/project/starVLA}"
REMOTE_STARVLA_ROOT="${REMOTE_STARVLA_ROOT:-/9950backfile/zhangjh/wh/starvla_wh/starVLA}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/9950backfile/zhangjh/.conda/envs/starvla/bin/python}"
CKPT="${CKPT:-4in1}"
GPU_IDS="${GPU_IDS:-0}"
POLICY_PORT="${POLICY_PORT:-5694}"
UNNORM_KEY="${UNNORM_KEY:-franka}"

TUNNEL_PID=""
STOP_SERVER_ON_EXIT="${STOP_SERVER_ON_EXIT:-0}"
MODE="full"
EVAL_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra eval args]

全自动：在 GPU 服务器起 Policy Server → SSH 隧道 → 本机跑 eval client。

Options:
  --server-only       只起 server + 隧道 + 连通性测试（不连机械臂/D405）
  --dry-run           传给 eval：只推理，不发 /arm_cmd（默认在 --full 时启用）
  --real              真机控制（关闭 dry-run）
  --max-steps N       eval 最大步数（默认 20）
  --task TEXT         语言指令（默认 "pick up the mug"）
  --ckpt NAME         服务器 checkpoint：4in1 | ljn82500（默认 4in1）
  --stop-server       脚本退出时停止远程 policy server
  --skip-server       假定 server 已在远程运行
  --skip-tunnel       不建隧道，直连 POLICY_HOST（默认仍用 127.0.0.1）
  -h, --help

环境变量：SERVER_USER SERVER_HOST ARX_X5_ROOT STARVLA_ROOT CKPT GPU_IDS POLICY_PORT

示例：
  bash $(basename "$0") --server-only
  bash $(basename "$0") --dry-run --max-steps 20
  bash $(basename "$0") --real --max-steps 50 --task "pick up the mug"
EOF
}

log() { echo "[starvla-auto] $*"; }
die() { log "ERROR: $*"; exit 1; }

remote_ckpt_path() {
  case "${CKPT}" in
    4in1) echo "${REMOTE_STARVLA_ROOT}/playground/Pretrained_models/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt" ;;
    ljn82500) echo "${REMOTE_STARVLA_ROOT}/results/ljn/v1/steps_82500/steps_82500_pytorch_model.pt" ;;
    *) echo "${CKPT}" ;;
  esac
}

remote_port_listening() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${SERVER_USER}@${SERVER_HOST}" \
    "ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '" 2>/dev/null
}

local_port_listening() {
  ss -tln 2>/dev/null | grep -q ":${POLICY_PORT} "
}

wait_remote_server() {
  local ckpt_path log_file
  ckpt_path="$(remote_ckpt_path)"
  log_file="logs/policy_server_${POLICY_PORT}.log"
  log "等待远程 Policy Server 就绪（模型加载可能需数分钟）..."
  for ((i = 1; i <= 300; i++)); do
    if ssh -o BatchMode=yes -o ConnectTimeout=8 "${SERVER_USER}@${SERVER_HOST}" \
      "grep -q 'server running' ${REMOTE_STARVLA_ROOT}/${log_file} 2>/dev/null || ss -tln 2>/dev/null | grep -q ':${POLICY_PORT} '"; then
      log "远程 server 已就绪 (port ${POLICY_PORT})"
      return 0
    fi
    if (( i % 15 == 0 )); then
      log "  仍在加载... (${i}/300, ~$((i * 2))s)"
      ssh -o BatchMode=yes "${SERVER_USER}@${SERVER_HOST}" \
        "tail -3 ${REMOTE_STARVLA_ROOT}/${log_file} 2>/dev/null" || true
    fi
    sleep 2
  done
  die "远程 server 超时未就绪，请检查: ssh ${SERVER_USER}@${SERVER_HOST} tail -50 ${REMOTE_STARVLA_ROOT}/${log_file}"
}

start_remote_server() {
  local ckpt_path log_file pid_file remote_script
  ckpt_path="$(remote_ckpt_path)"
  log_file="logs/policy_server_${POLICY_PORT}.log"
  pid_file="/tmp/starvla_policy_${POLICY_PORT}.pid"
  remote_script="${REMOTE_ARX_ROOT:-/9950backfile/zhangjh/wh/ARX_X5_clean}/lewm_x5_grasp/scripts/run_policy_server_on_server.sh"

  if remote_port_listening; then
    log "远程 port ${POLICY_PORT} 已在监听，跳过启动"
    return 0
  fi

  log "在 ${SERVER_USER}@${SERVER_HOST} 启动 Policy Server (CKPT=${CKPT})..."
  ssh -o BatchMode=yes "${SERVER_USER}@${SERVER_HOST}" bash -s <<REMOTE
set -e
CKPT_PATH="${ckpt_path}"
STARVLA_ROOT="${REMOTE_STARVLA_ROOT}"
LOG="\${STARVLA_ROOT}/${log_file}"
PIDF="${pid_file}"
PORT=${POLICY_PORT}
REMOTE_SCRIPT="${remote_script}"

if [[ ! -f "\${CKPT_PATH}" ]]; then
  echo "checkpoint missing: \${CKPT_PATH}" >&2
  exit 1
fi

if ss -tln 2>/dev/null | grep -q ":\${PORT} "; then
  echo "already_listening"
  exit 0
fi

if [[ -x "\${REMOTE_SCRIPT}" || -f "\${REMOTE_SCRIPT}" ]]; then
  cd "\$(dirname "\${REMOTE_SCRIPT}")"
  CKPT=${CKPT} GPU_IDS=${GPU_IDS} UNNORM_KEY=${UNNORM_KEY} BASE_PORT=\${PORT} \\
    setsid nohup bash "\${REMOTE_SCRIPT}" > /tmp/starvla_launch_\${PORT}.log 2>&1 < /dev/null &
  echo \$! > "\${PIDF}"
  disown
  echo "started_wrapper_pid=\$(cat \${PIDF})"
else
  cd "\${STARVLA_ROOT}"
  mkdir -p logs
  export PYTHONPATH="\${STARVLA_ROOT}:\${PYTHONPATH:-}"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} nohup ${REMOTE_PYTHON} deployment/model_server/server_policy.py \\
    --ckpt_path "\${CKPT_PATH}" \\
    --port \${PORT} \\
    --unnorm_key ${UNNORM_KEY} \\
    --use_bf16 \\
    --idle_timeout -1 \\
    > "\${LOG}" 2>&1 < /dev/null &
  echo \$! > "\${PIDF}"
  disown
  echo "started_pid=\$(cat \${PIDF})"
fi
REMOTE
  wait_remote_server
}

start_tunnel() {
  if local_port_listening; then
    log "本机 port ${POLICY_PORT} 已在监听（可能已有隧道）"
    return 0
  fi
  log "建立 SSH 隧道 localhost:${POLICY_PORT} → ${SERVER_HOST}:127.0.0.1:${POLICY_PORT}"
  ssh -N -f -L "${POLICY_PORT}:127.0.0.1:${POLICY_PORT}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    "${SERVER_USER}@${SERVER_HOST}"
  sleep 1
  local_port_listening || die "SSH 隧道建立失败"
  TUNNEL_PID="$(pgrep -f "ssh -N -f -L ${POLICY_PORT}:127.0.0.1:${POLICY_PORT} ${SERVER_USER}@${SERVER_HOST}" | head -1 || true)"
  log "隧道已建立${TUNNEL_PID:+ (pid ${TUNNEL_PID})}"
}

test_server_connection() {
  log "测试 WebSocket 连通性..."
  export STARVLA_ROOT
  export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
  python3 - <<'PY'
import os, sys
root = os.environ["STARVLA_ROOT"]
sys.path.insert(0, root)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
host = os.environ.get("POLICY_HOST", "127.0.0.1")
port = int(os.environ.get("POLICY_PORT", "5694"))
client = WebsocketClientPolicy(host, port)
meta = client.get_server_metadata()
print("server_metadata:", meta)
client.close()
print("WEBSOCKET_OK")
PY
}

check_robot_stack() {
  source /opt/ros/humble/setup.bash
  source "${ARX_X5_ROOT}/ROS2/X5_ws/install/setup.bash"
  if ros2 topic list 2>/dev/null | grep -q '/arm_status'; then
    return 0
  fi
  return 1
}

stop_remote_server() {
  log "停止远程 Policy Server (port ${POLICY_PORT})..."
  ssh -o BatchMode=yes "${SERVER_USER}@${SERVER_HOST}" bash -s <<REMOTE || true
PORT=${POLICY_PORT}
pkill -f "server_policy.py.*--port \${PORT}" 2>/dev/null || true
rm -f /tmp/starvla_policy_\${PORT}.pid
REMOTE
}

cleanup() {
  if [[ "${STOP_SERVER_ON_EXIT}" == "1" ]]; then
    stop_remote_server
  fi
  if [[ "${KEEP_TUNNEL}" != "1" && -n "${TUNNEL_PID}" ]]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SKIP_SERVER=0
SKIP_TUNNEL=0
KEEP_TUNNEL=0
DRY_RUN=1
MAX_STEPS=20
TASK="pick up the mug"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-only) MODE="server-only"; shift ;;
    --full) MODE="full"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --real) DRY_RUN=0; shift ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --stop-server) STOP_SERVER_ON_EXIT=1; shift ;;
    --skip-server) SKIP_SERVER=1; shift ;;
    --skip-tunnel) SKIP_TUNNEL=1; shift ;;
    --keep-tunnel) KEEP_TUNNEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EVAL_ARGS+=("$@"); break ;;
    *) EVAL_ARGS+=("$1"); shift ;;
  esac
done

[[ -d "${STARVLA_ROOT}/deployment" ]] || die "STARVLA_ROOT 无效: ${STARVLA_ROOT}"
[[ -d "${ARX_X5_ROOT}/lewm_x5_grasp" ]] || die "ARX_X5_ROOT 无效: ${ARX_X5_ROOT}"

python3 -c "import websockets, msgpack" 2>/dev/null || {
  log "安装 client 依赖: pip3 install --user websockets msgpack"
  pip3 install --user websockets msgpack -q
}

export POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
export POLICY_PORT
export STARVLA_ROOT
export ARX_X5_ROOT

if [[ "${SKIP_SERVER}" == "0" ]]; then
  start_remote_server
else
  log "跳过远程 server 启动 (--skip-server)"
  remote_port_listening || die "远程 port ${POLICY_PORT} 未在监听"
fi

if [[ "${SKIP_TUNNEL}" == "0" ]]; then
  start_tunnel
else
  export POLICY_HOST="${SERVER_HOST}"
  log "跳过隧道，直连 POLICY_HOST=${POLICY_HOST}"
fi

test_server_connection

if [[ "${MODE}" == "server-only" ]]; then
  KEEP_TUNNEL=1
  log "server-only 完成：远程推理服务可用（隧道保持开启）"
  exit 0
fi

if ! check_robot_stack; then
  die "机械臂 ROS2 未就绪。请先另开终端执行：
  1) cd ${ARX_X5_ROOT}/ARX_CAN/arx_can && bash arx_can1.sh
  2) ros2 launch arx_x5_controller open_single_arm.launch.py
  或先用: bash $(basename "$0") --server-only"
fi

EVAL_CMD=(
  bash "${ARX_X5_ROOT}/lewm_x5_grasp/scripts/run_eval_starvla_libero_x5.sh"
  --max-steps "${MAX_STEPS}"
  --task "${TASK}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  EVAL_CMD+=(--dry-run)
fi
if [[ ${#EVAL_ARGS[@]} -gt 0 ]]; then
  EVAL_CMD+=("${EVAL_ARGS[@]}")
fi

log "启动本机 eval client: ${EVAL_CMD[*]}"
exec "${EVAL_CMD[@]}"
