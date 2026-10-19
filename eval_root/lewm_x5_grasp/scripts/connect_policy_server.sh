#!/usr/bin/env bash
# Quick connectivity check: laptop -> StarVLA Policy Server (WebSocket).
set -eo pipefail

POLICY_HOST="${POLICY_HOST:-115.190.90.101}"
POLICY_PORT="${POLICY_PORT:-5694}"
STARVLA_ROOT="${STARVLA_ROOT:-$HOME/project/starVLA}"

echo "[connect] target ws://${POLICY_HOST}:${POLICY_PORT}"
echo "[connect] laptop public IP: $(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo unknown)"

if ! timeout 3 bash -c "echo >/dev/tcp/${POLICY_HOST}/${POLICY_PORT}" 2>/dev/null; then
  echo "[connect] ERROR: TCP ${POLICY_HOST}:${POLICY_PORT} unreachable"
  exit 1
fi
echo "[connect] TCP port open"

export STARVLA_ROOT
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.environ["STARVLA_ROOT"])
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

host = os.environ.get("POLICY_HOST", "115.190.90.101")
port = int(os.environ.get("POLICY_PORT", "5694"))
print(f"[connect] WebSocket handshake -> {host}:{port}")
client = WebsocketClientPolicy(host, port)
meta = client.get_server_metadata()
print("server_metadata:", meta)
client.close()
print("WEBSOCKET_OK")
PY
