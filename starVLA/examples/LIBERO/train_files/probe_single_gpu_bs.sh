#!/usr/bin/env bash
# 单卡 per_device_batch_size OOM 探测（默认 GPU 6）
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled

BASE_VLM="${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL"
CONFIG_YAML="${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/bs_probe_gpu${CUDA_VISIBLE_DEVICES}_summary.txt"
echo "=== BS probe GPU ${CUDA_VISIBLE_DEVICES} $(date -Iseconds) ===" | tee "${SUMMARY}"

for BS in 4 2 1; do
  RUN_ID="bs_probe_gpu${CUDA_VISIBLE_DEVICES}_bs${BS}"
  LOG_FILE="${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
  PORT=$((29520 + BS))

  echo "" | tee -a "${SUMMARY}"
  echo "--- per_device_batch_size=${BS} log=${LOG_FILE} ---" | tee -a "${SUMMARY}"

  set +e
  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port "${PORT}" \
    --num_processes 1 \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --framework.name QwenGR00T \
    --framework.qwenvl.base_vlm "${BASE_VLM}" \
    --datasets.vla_data.per_device_batch_size "${BS}" \
    --datasets.vla_data.data_mix libero_all \
    --trainer.max_train_steps 3 \
    --trainer.logging_frequency 1 \
    --trainer.gradient_accumulation_steps 64 \
    --run_root_dir ./results/Checkpoints \
    --run_id "${RUN_ID}" \
    2>&1 | tee "${LOG_FILE}"
  EC=${PIPESTATUS[0]}
  set -e

  if grep -qiE 'CUDA out of memory|OutOfMemoryError|OOM' "${LOG_FILE}"; then
    RESULT="OOM"
  elif grep -q 'action_dit_loss' "${LOG_FILE}" && [[ "${EC}" -eq 0 ]]; then
    RESULT="OK"
  elif [[ "${EC}" -ne 0 ]]; then
    RESULT="FAIL(exit=${EC})"
  else
    RESULT="UNKNOWN"
  fi

  PEAK=$(grep -oE 'allocated=[0-9.]+ GiB|reserved=[0-9.]+ GiB' "${LOG_FILE}" 2>/dev/null | tail -4 | tr '\n' ' ' || true)
  echo "RESULT bs=${BS}: ${RESULT} ${PEAK}" | tee -a "${SUMMARY}"

  sleep 5
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo "" | tee -a "${SUMMARY}"
echo "=== done ===" | tee -a "${SUMMARY}"
