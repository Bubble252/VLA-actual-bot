#!/usr/bin/env bash
# 双卡 GPU 6,7 复现/定位 bs=32 DiT NaN
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export DEBUG_BS="${DEBUG_BS:-32}"
export DEBUG_GRAD_ACCUM="${DEBUG_GRAD_ACCUM:-2}"
export DEBUG_OPT_STEPS="${DEBUG_OPT_STEPS:-2}"

LOG="${STARVLA_ROOT}/results/Checkpoints/train_logs/debug_nan_bs32_dual67_$(date +%Y%m%d_%H%M%S).log"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} DEBUG_BS=${DEBUG_BS} DEBUG_GRAD_ACCUM=${DEBUG_GRAD_ACCUM} DEBUG_OPT_STEPS=${DEBUG_OPT_STEPS}"
echo "log=${LOG}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29740 \
  --num_processes 2 \
  examples/LIBERO/train_files/debug_nan_bs32_dual.py \
  2>&1 | tee "${LOG}"
