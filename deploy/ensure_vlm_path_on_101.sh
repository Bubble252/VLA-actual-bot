#!/usr/bin/env bash
# 在 101 上：1) SpikeBrain 路径软链  2) 补齐 transformers remote-code 缓存（避免缺 ulysses.py）
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

SSH=(ssh -p "${SERVER_PORT}" -o BatchMode=yes -o ConnectTimeout=20 "${SERVER_USER}@${SERVER_HOST}")

# 注意：远程命令用单条 ssh 字符串，避免 `ssh host bash -lc` 在部分环境下触发空 -c
"${SSH[@]}" "set -e
mkdir -p \"\$(dirname '${VLM_CONFIG_PATH}')\"
if [[ ! -d '${VLM_REAL}' ]]; then
  echo \"ERROR: missing real VLM dir: ${VLM_REAL}\" >&2
  exit 1
fi
ln -sfn '${VLM_REAL}' '${VLM_CONFIG_PATH}'
test -f '${VLM_CONFIG_PATH}/config.json'
ls -ld '${VLM_CONFIG_PATH}'

# HF dynamic module cache: keep empty package __init__.py; copy remote-code deps
CACHE=/root/.cache/huggingface/modules/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL
mkdir -p \"\$CACHE\"
for f in ulysses.py attn.py moba_attn.py sse.py sse_swa.py configuration_spb2_vl.py modeling_spb2_vl.py processing_spb2_vl.py video_processing_qwen3_vl.py; do
  if [[ -f '${VLM_REAL}/'\$f ]]; then cp -a '${VLM_REAL}/'\$f \"\$CACHE/\$f\"; fi
done
: > \"\$CACHE/__init__.py\"
rm -rf \"\$CACHE/__pycache__\"
test -f \"\$CACHE/ulysses.py\"
echo OK: SpikeBrain path + HF cache ready
"
