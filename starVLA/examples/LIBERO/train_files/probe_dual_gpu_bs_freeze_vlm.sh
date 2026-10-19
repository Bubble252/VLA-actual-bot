#!/usr/bin/env bash
# 双卡 ZeRO-2 + freeze qwen_vl_interface：探测可承受的 per_device_batch_size
# 使用 examples/LIBERO/train_files/starvla_cotrain_libero.yaml（含 state_dim=7、freeze_modules）
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

CONFIG_YAML="${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
BASE_VLM="${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL"
LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/bs_probe_freeze_vlm_dual67_summary.txt"
echo "=== Freeze VLM + dual GPU 6,7 | $(date -Iseconds) ===" | tee "${SUMMARY}"
echo "config: ${CONFIG_YAML}" | tee -a "${SUMMARY}"
grep -E 'state_dim|freeze_modules|repeated_diffusion|action_type|per_device_batch_size' "${CONFIG_YAML}" | head -8 | tee -a "${SUMMARY}"

# 从高到低试，尽快找到上限
for BS in 8 4 2 1; do
  RUN_ID="bs_probe_freeze67_bs${BS}"
  LOG_FILE="${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
  PORT=$((29720 + BS))

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
    --trainer.freeze_modules qwen_vl_interface \
    --trainer.max_train_steps 2 \
    --trainer.logging_frequency 1 \
    --trainer.gradient_accumulation_steps 16 \
    --run_root_dir ./results/Checkpoints \
    --run_id "${RUN_ID}" \
    2>&1 | tee "${LOG_FILE}"
  EC=${PIPESTATUS[0]}
  set -e

  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "${LOG_FILE}"; then
    RESULT="OOM"
  elif grep -qiE 'illegal memory access|CUDA error' "${LOG_FILE}"; then
    RESULT="CUDA_ERROR"
  elif grep -q 'action_dit_loss' "${LOG_FILE}" && [[ "${EC}" -eq 0 ]]; then
    if grep -qE 'Step 1, Loss|1/2.*100%|2/2' "${LOG_FILE}"; then
      RESULT="OK(>=2steps)"
    else
      RESULT="OK(partial)"
    fi
  elif [[ "${EC}" -ne 0 ]]; then
    RESULT="FAIL(exit=${EC})"
  else
    RESULT="UNKNOWN"
  fi

  MEM=$(grep -oE 'cuda/max_allocated_gb.: [0-9.]+|GPU[0-9]: used [0-9.]+ /' "${LOG_FILE}" 2>/dev/null | tail -8 | tr '\n' ' ' || true)
  TRAINABLE=$(grep -oE 'Trainable.*' "${LOG_FILE}" 2>/dev/null | head -1 || true)
  echo "RESULT bs=${BS}: ${RESULT} | ${TRAINABLE} | ${MEM}" | tee -a "${SUMMARY}"

  sleep 10
  python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo "" | tee -a "${SUMMARY}"
echo "=== done ===" | tee -a "${SUMMARY}"
