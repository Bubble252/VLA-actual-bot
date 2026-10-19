#!/usr/bin/env bash
# 双卡 ZeRO-2 per_device_batch_size OOM 探测（默认 GPU 6,7，与 run_libero_train 一致）
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled

BASE_VLM="${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL"
CONFIG_YAML="${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/bs_probe_dual_gpu67_summary.txt"
echo "=== Dual-GPU BS probe CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ZeRO-2 num_processes=2 $(date -Iseconds) ===" | tee "${SUMMARY}"

for BS in 4 2 1; do
  RUN_ID="bs_probe_dual67_bs${BS}"
  LOG_FILE="${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
  PORT=$((29620 + BS))

  echo "" | tee -a "${SUMMARY}"
  echo "--- per_device_batch_size=${BS} log=${LOG_FILE} ---" | tee -a "${SUMMARY}"

  set +e
  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port "${PORT}" \
    --num_processes 2 \
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

  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "${LOG_FILE}"; then
    RESULT="OOM"
  elif grep -q 'action_dit_loss' "${LOG_FILE}" && [[ "${EC}" -eq 0 ]]; then
    STEPS=$(grep -cE 'Step [0-9]+, Loss' "${LOG_FILE}" 2>/dev/null || echo 0)
    if grep -qE 'Total optimization steps = 3' "${LOG_FILE}" && [[ "${STEPS}" -ge 1 ]]; then
      RESULT="OK"
    else
      RESULT="PARTIAL"
    fi
  elif [[ "${EC}" -ne 0 ]]; then
    RESULT="FAIL(exit=${EC})"
  else
    RESULT="UNKNOWN"
  fi

  PEAK=$(grep -oE 'cuda/max_allocated_gb.: [0-9.]+|GPU[0-9]: used [0-9.]+ /' "${LOG_FILE}" 2>/dev/null | tail -6 | tr '\n' ' ' || true)
  echo "RESULT bs=${BS}: ${RESULT} ${PEAK}" | tee -a "${SUMMARY}"

  sleep 10
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo "" | tee -a "${SUMMARY}"
echo "=== done ===" | tee -a "${SUMMARY}"
