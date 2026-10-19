#!/usr/bin/env bash
# GPU 4,5：扫描 DiT chunk_size，检查是否因大 chunk 导致 NaN
# repeat=6 → DiT 有效 batch=192；chunk=192 为单次整批 forward（最激进）
set -euo pipefail
STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

# 固定 4,5，避免继承训练环境的 CUDA_VISIBLE_DEVICES=6,7
export CUDA_VISIBLE_DEVICES=4,5
export DEBUG_BS="${DEBUG_BS:-32}"
export DEBUG_GRAD_ACCUM="${DEBUG_GRAD_ACCUM:-2}"
export DEBUG_REPEAT="${DEBUG_REPEAT:-6}"
export DEBUG_OPT_STEPS="${DEBUG_OPT_STEPS:-4}"
# 空格分隔的 chunk 列表，默认 128 192（64 已在 rep6 长跑中验证）
CHUNK_LIST="${CHUNK_LIST:-128 192}"

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
SUMMARY="${LOG_DIR}/probe_chunk_nan_dual45_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "=== chunk NaN probe GPU ${CUDA_VISIBLE_DEVICES} repeat=${DEBUG_REPEAT} bs=${DEBUG_BS} $(date -Iseconds) ===" | tee "${SUMMARY}"

for CHUNK in ${CHUNK_LIST}; do
  export DEBUG_CHUNK_SIZE="${CHUNK}"
  LOG="${LOG_DIR}/probe_chunk_nan_dual45_chunk${CHUNK}_$(date +%Y%m%d_%H%M%S).log"
  echo "" | tee -a "${SUMMARY}"
  echo "--- chunk_size=${CHUNK} log=${LOG} ---" | tee -a "${SUMMARY}"

  set +e
  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port 29751 \
    --num_processes 2 \
    examples/LIBERO/train_files/debug_nan_bs32_dual.py \
    2>&1 | tee "${LOG}"
  RC=$?
  set -e

  if grep -qE 'action_loss NaN|finite=False|FORWARD_ERR' "${LOG}"; then
    echo "RESULT chunk=${CHUNK}: FAIL (NaN/error)" | tee -a "${SUMMARY}"
  elif grep -qE 'finished optimizer step' "${LOG}"; then
    PEAK=$(grep -oE 'peak_cuda_alloc=[0-9.]+' "${LOG}" | tail -1 || true)
    echo "RESULT chunk=${CHUNK}: OK exit=${RC} ${PEAK}" | tee -a "${SUMMARY}"
  else
    echo "RESULT chunk=${CHUNK}: UNKNOWN exit=${RC}" | tee -a "${SUMMARY}"
  fi
done

echo "" | tee -a "${SUMMARY}"
echo "=== done summary=${SUMMARY} ===" | tee -a "${SUMMARY}"
