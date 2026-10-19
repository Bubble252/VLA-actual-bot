#!/usr/bin/env bash
# GPU 7 单卡 bs=2 全训探测：vlm_full_train_sample_wise（逐样本 VLM+DiT，避免 batched NaN）
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
mkdir -p "${TRITON_CACHE_DIR}"
# shellcheck source=examples/LIBERO/train_files/setup_starvla_cuda_cache.sh
source "${STARVLA_ROOT}/examples/LIBERO/train_files/setup_starvla_cuda_cache.sh"
setup_starvla_cuda_cache

export CUDA_VISIBLE_DEVICES=7
NUM_PROCESSES=1
PER_DEVICE_BS=2
TARGET_GLOBAL_BS=4
PROBE_STEPS=${PROBE_STEPS:-25}
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_RESUME_CKPT:-${STARVLA_ROOT}/results/Checkpoints/20260608_1842_libero_all_spb2_gr00t_full_spb2align/checkpoints/steps_250_pytorch_model.pt}

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
run_id="probe_bs2_gpu7_samplewise_$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${run_id}.log"

echo "[probe gpu7 sample-wise] GPU=${CUDA_VISIBLE_DEVICES} per_device_bs=${PER_DEVICE_BS}"
echo "[probe gpu7 sample-wise] global_bs=${TARGET_GLOBAL_BS} grad_accum=$((TARGET_GLOBAL_BS/(PER_DEVICE_BS*NUM_PROCESSES))) steps=${PROBE_STEPS}"
echo "[probe gpu7 sample-wise] vlm_full_train_sample_wise=true ckpt=${pretrained_ckpt}"
echo "[probe gpu7 sample-wise] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29527 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.vlm_full_train_sample_wise true \
  --framework.qwenvl.vlm_packing false \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 1 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.is_resume false \
  --trainer.pretrained_checkpoint "${pretrained_ckpt}" \
  --trainer.resume_step 0 \
  --trainer.skip_non_finite_loss true \
  --trainer.max_grad_skips 50 \
  --trainer.enable_gradient_checkpointing true \
  --framework.qwenvl.rmsnorm_fp32_forward true \
  --framework.action_model.repeated_diffusion_steps 4 \
  --trainer.freeze_modules "" \
  --run_root_dir ./results/Checkpoints \
  --run_id ${run_id} \
  > "${LOG}" 2>&1 &

echo "[probe gpu7 sample-wise] started pid=$! log=${LOG}"
echo "[probe gpu7 sample-wise] tail -f ${LOG}"
