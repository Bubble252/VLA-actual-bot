#!/usr/bin/env bash
# 从 steps_1000 恢复 stage2 全量微调；学习率降为原来的 1/4；NaN/Inf 跳过该步
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

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
NUM_PROCESSES=6
PER_DEVICE_BS=1
TARGET_GLOBAL_BS=126

# LR: yaml 默认 base=2.5e-5, qwen=1e-5, action=1e-4 → 降为 1/4
LR_BASE=6.25e-06
LR_QWEN=2.5e-06
LR_ACTION=2.5e-05

run_id=20260606_1533_libero_all_spb2_gr00t_stage2_full_from6k
run_root_dir=./results/Checkpoints
output_dir=${run_root_dir}/${run_id}
resume_ckpt=${STARVLA_ROOT}/${output_dir}/checkpoints/steps_1000_pytorch_model.pt
resume_step=1000

if [ ! -f "${resume_ckpt}" ]; then
  echo "[stage2 resume lowlr] ERROR: checkpoint not found: ${resume_ckpt}"
  exit 1
fi

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_libero_stage2_full_resume_lowlr_$(date +%Y%m%d_%H%M%S).log"

echo "[stage2 resume lowlr] run_id=${run_id}"
echo "[stage2 resume lowlr] GPU=${CUDA_VISIBLE_DEVICES} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[stage2 resume lowlr] per_device_bs=${PER_DEVICE_BS} target_global_bs=${TARGET_GLOBAL_BS} grad_accum=$((TARGET_GLOBAL_BS / (PER_DEVICE_BS * NUM_PROCESSES)))"
echo "[stage2 resume lowlr] resume_ckpt=${resume_ckpt}"
echo "[stage2 resume lowlr] resume_step=${resume_step}"
echo "[stage2 resume lowlr] LR base=${LR_BASE} qwen=${LR_QWEN} action=${LR_ACTION} (1/4 of yaml default)"
echo "[stage2 resume lowlr] save_interval=500 skip_non_finite_loss=true"
echo "[stage2 resume lowlr] output_dir=${STARVLA_ROOT}/${output_dir}"
echo "[stage2 resume lowlr] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29503 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 12 \
  --trainer.save_interval 500 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base ${LR_BASE} \
  --trainer.learning_rate.qwen_vl_interface ${LR_QWEN} \
  --trainer.learning_rate.action_model ${LR_ACTION} \
  --trainer.is_resume false \
  --trainer.pretrained_checkpoint ${resume_ckpt} \
  --trainer.resume_step ${resume_step} \
  --trainer.skip_non_finite_loss true \
  --trainer.enable_gradient_checkpointing false \
  --trainer.freeze_modules "" \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jianingliu \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "[stage2 resume lowlr] started with nohup, pid=${TRAIN_PID}"
echo "[stage2 resume lowlr] follow log: tail -f ${LOG}"
