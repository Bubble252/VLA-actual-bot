#!/usr/bin/env bash
# 测笔记本 ↔ GPU Policy Server 通信/推理时延（经 SSH 隧道）。
# Usage:
#   MODEL=1p bash bench_policy_latency.sh
#   MODEL=3p N=40 bash bench_policy_latency.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 避免 shell 里残留的 POLICY_PORT/HOST 污染 MODEL 默认值
unset POLICY_PORT POLICY_HOST UNNORM_KEY ACTION_SPACE FRAME_ALIGN CKPT_STEP 2>/dev/null || true

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/models.env"

export STARVLA_ROOT="${STARVLA_ROOT:-$HOME/project/starVLA}"
# 公网直连常被拦；时延测试固定走本机隧道
export POLICY_HOST="127.0.0.1"
N="${N:-30}"

PORT="${POLICY_PORT}" bash "${SCRIPT_DIR}/open_tunnel.sh"

exec python3 "${SCRIPT_DIR}/bench_policy_latency.py" \
  --host "${POLICY_HOST}" \
  --port "${POLICY_PORT}" \
  --unnorm-key "${UNNORM_KEY}" \
  --starvla-root "${STARVLA_ROOT}" \
  --n "${N}" \
  "$@"
