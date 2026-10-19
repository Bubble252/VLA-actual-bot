#!/usr/bin/env bash
# GPU 0+7 双卡 bs=2 + VLM packing：用 varlen 单次 forward 代替 batched VLM（对齐 lmms-engine）
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
export NCCL_TIMEOUT=1800 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
mkdir -p "${TRITON_CACHE_DIR}"
# shellcheck source=examples/LIBERO/train_files/setup_starvla_cuda_cache.sh
source "${STARVLA_ROOT}/examples/LIBERO/train_files/setup_starvla_cuda_cache.sh"
setup_starvla_cuda_cache

export CUDA_VISIBLE_DEVICES=0,7
NUM_PROCESSES=2
PER_DEVICE_BS=2
TARGET_GLOBAL_BS=8
PROBE_STEPS=5
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
run_id="probe_packing_gpu07_$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${run_id}.log"

echo "[probe packing] GPU=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES}"
echo "[probe packing] per_device_bs=${PER_DEVICE_BS} global_bs=${TARGET_GLOBAL_BS} grad_accum=$((TARGET_GLOBAL_BS/(PER_DEVICE_BS*NUM_PROCESSES)))"
echo "[probe packing] vlm_packing=true max_seqlen=8192 steps=${PROBE_STEPS}"
echo "[probe packing] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29522 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.vlm_packing true \
  --framework.qwenvl.vlm_packing_max_seqlen 8192 \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 1 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.skip_non_finite_loss true \
  --trainer.max_grad_skips 50 \
  --trainer.max_grad_norm_skip_threshold 1000 \
  --trainer.enable_gradient_checkpointing true \
  --framework.qwenvl.rmsnorm_fp32_forward true \
  --framework.qwenvl.verify_fp32_after_prepare true \
  --framework.qwenvl.vlm_hidden_collapse_row_ratio 0.3 \
  --trainer.freeze_modules "" \
  --run_root_dir ./results/Checkpoints \
  --run_id ${run_id} \
  > "${LOG}" 2>&1 &

echo "[probe packing] started pid=$! log=${LOG}"
echo "[probe packing] tail -f ${LOG}"
