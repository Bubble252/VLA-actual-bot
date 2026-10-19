#!/usr/bin/env bash
# GPU 0+7 双卡 bs=2 全训探测（带 numerical_probe 日志），不影响 1-6 卡正式训练。
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

# 仅占用物理 GPU 0 和 7
export CUDA_VISIBLE_DEVICES=0,7
NUM_PROCESSES=2
PER_DEVICE_BS=2
TARGET_GLOBAL_BS=44   # 2×2×11，与六卡 bs=2 的 grad_accum 比例一致
PROBE_STEPS=15
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_ROOT}/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
run_id="probe_bs2_gpu07_numprobe_$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/${run_id}.log"

echo "[probe gpu07] GPU=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES}"
echo "[probe gpu07] per_device_bs=${PER_DEVICE_BS} global_bs=${TARGET_GLOBAL_BS} probe_steps=${PROBE_STEPS}"
echo "[probe gpu07] numerical_probe enabled via yaml; GC=false; full VLM train"
echo "[probe gpu07] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29521 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 1 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.is_resume false \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.resume_step 0 \
  --trainer.enable_gradient_checkpointing false \
  --framework.qwenvl.vlm_forward_fp32 false \
  --trainer.freeze_modules "" \
  --run_root_dir ./results/Checkpoints \
  --run_id ${run_id} \
  > "${LOG}" 2>&1 &

echo "[probe gpu07] started pid=$! log=${LOG}"
echo "[probe gpu07] tail -f ${LOG}"
