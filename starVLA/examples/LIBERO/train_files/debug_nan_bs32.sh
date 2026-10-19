#!/usr/bin/env bash
# 单卡复现/定位 bs=32 DiT NaN（不占双卡训练）
set -euo pipefail
STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"
export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 训练占 6,7 时，可改成 0 或等训练结束
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

LOG="${STARVLA_ROOT}/results/Checkpoints/train_logs/debug_nan_bs32_$(date +%Y%m%d_%H%M%S).log"
echo "log=${LOG}"
python examples/LIBERO/train_files/debug_nan_bs32.py 2>&1 | tee "${LOG}"
