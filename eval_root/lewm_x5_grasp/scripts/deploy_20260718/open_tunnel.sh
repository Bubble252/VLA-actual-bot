#!/usr/bin/env bash
# 为本机建立到 GPU 的 Policy 端口隧道。
# Usage: PORT=5794 bash open_tunnel.sh
set -eo pipefail
PORT="${PORT:-5794}"
SERVER_HOST="${SERVER_HOST:-115.190.90.101}"
SERVER_PORT="${SERVER_PORT:-27219}"
SERVER_USER="${SERVER_USER:-root}"

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "[tunnel] local :${PORT} already listening — skip"
  exit 0
fi

ssh -p "${SERVER_PORT}" -N -f -L "${PORT}:127.0.0.1:${PORT}" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  "${SERVER_USER}@${SERVER_HOST}"
echo "[tunnel] localhost:${PORT} → ${SERVER_HOST}:127.0.0.1:${PORT}"
