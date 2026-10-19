#!/usr/bin/env bash
# Pull full ARX_X5_clean repo from GPU server 46 into a separate local directory.
set -eo pipefail

SERVER="${SERVER:-zhangjh@172.18.127.46}"
REMOTE_ROOT="${REMOTE_ROOT:-/9950backfile/zhangjh/wh/ARX_X5_clean}"
DEST="${DEST:-$HOME/project/ARX_X5_clean_server46}"

echo "[pull] ${SERVER}:${REMOTE_ROOT} -> ${DEST}"

ssh -o ConnectTimeout=10 -o BatchMode=yes "${SERVER}" "test -d '${REMOTE_ROOT}'"

mkdir -p "${DEST}"

rsync -av --progress --no-owner --no-group \
  --exclude '.git/' \
  --exclude '**/__pycache__/' \
  --exclude '**/build/' \
  --exclude '**/install/' \
  --exclude '**/log/' \
  --exclude 'lewm_x5_grasp/data/' \
  "${SERVER}:${REMOTE_ROOT}/" \
  "${DEST}/"

echo "[pull] done -> ${DEST}"
