#!/usr/bin/env bash
# 本机建立到 101 Policy Server 的本地隧道（端口 5697，避开 0813 的 5696）。
# Usage: bash open_tunnel.sh
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

PORT="${POLICY_PORT}"
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "[tunnel] local :${PORT} already listening — skip"
  exit 0
fi

ssh -p "${SERVER_PORT}" -N -f -L "${PORT}:127.0.0.1:${PORT}" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  "${SERVER_USER}@${SERVER_HOST}"
echo "[tunnel] localhost:${PORT} → ${SERVER_HOST}:127.0.0.1:${PORT}"
