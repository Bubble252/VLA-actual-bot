#!/usr/bin/env bash
# GPU 6,7：从 steps_21000 checkpoint resume，bs=32
set -euo pipefail
STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_API_KEY="4f3ad82c189f15c159dc9f3e38395ef7a7051584"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="starVLA"
export WANDB_ENTITY="jianingliu"
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

export CUDA_VISIBLE_DEVICES=6,7
NUM_PROCESSES=2
run_id=20260523_0541_libero_all_spb2_gr00t_freeze
config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_ROOT}/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_21000_pytorch_model.pt

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_libero_train_resume_bs32_$(date +%Y%m%d_%H%M%S).log"

echo "[resume bs32] run_id=${run_id} GPU=${CUDA_VISIBLE_DEVICES} ckpt=${pretrained_ckpt} log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29502 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 32 \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size 128 \
  --trainer.epochs 12 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 200 \
  --trainer.is_resume true \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --run_root_dir ./results/Checkpoints \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jianingliu \
  --trainer.freeze_modules qwen_vl_interface \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "[resume bs32] started with nohup, pid=${TRAIN_PID}"
echo "[resume bs32] follow log: ${LOG}"
