#!/usr/bin/env bash
# LIBERO 全训 — 物理 GPU 0+7 双卡（X5A 占 1-6 时使用）
# bs=1 为全训 VLM 数值稳定上限；vlm_packing 需 bs>2 且 step1 backward 仍会 NaN（见 probe_packing_gpu07_20260608）
# MFU 优化：repeat 6→4、eval 间隔加大、独占 0/7 避免 NCCL/illegal memory access
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

kill_line_py() {
  local pid
  for pid in $(pgrep -f "python.*line\.py" 2>/dev/null || true); do
    echo "[gpu07 spb2align] kill line.py pid=${pid}"
    kill "${pid}" 2>/dev/null || true
  done
  sleep 3
}
if [ "${STARVLA_KILL_LINE_PY:-1}" != "0" ]; then
  kill_line_py
fi

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
export NCCL_TIMEOUT=1800 TORCH_NCCL_BLOCKING_WAIT=1 TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
mkdir -p "${TRITON_CACHE_DIR}"

# shellcheck source=examples/LIBERO/train_files/setup_starvla_cuda_cache.sh
source "${STARVLA_ROOT}/examples/LIBERO/train_files/setup_starvla_cuda_cache.sh"
setup_starvla_cuda_cache

export CUDA_VISIBLE_DEVICES=0,7
NUM_PROCESSES=2
PER_DEVICE_BS=1
TARGET_GLOBAL_BS=${TARGET_GLOBAL_BS:-84}
REPEAT_STEPS=${REPEAT_STEPS:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}

run_date=$(date +%Y%m%d)
run_time=$(date +%H%M)
run_id=${STARVLA_RUN_ID:-${run_date}_${run_time}_libero_all_spb2_gr00t_gpu07_spb2align}

RESUME_CKPT="${STARVLA_RESUME_CKPT:-}"
RESUME_STEP="${STARVLA_RESUME_STEP:-0}"
RESUME_ARGS=()
if [ -n "${RESUME_CKPT}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[gpu07 spb2align] ERROR: STARVLA_RESUME_CKPT not found: ${RESUME_CKPT}"
    exit 1
  fi
  RESUME_ARGS=(
    --trainer.is_resume false
    --trainer.pretrained_checkpoint "${RESUME_CKPT}"
    --trainer.resume_step "${RESUME_STEP}"
  )
  echo "[gpu07 spb2align] resume: ${RESUME_CKPT} step=${RESUME_STEP}"
fi

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
run_root_dir=./results/Checkpoints
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_libero_gpu07_spb2align_$(date +%Y%m%d_%H%M%S).log"

grad_accum=$((TARGET_GLOBAL_BS / (PER_DEVICE_BS * NUM_PROCESSES)))
echo "[gpu07 spb2align] GPU=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES}"
echo "[gpu07 spb2align] per_device_bs=${PER_DEVICE_BS} target_global_bs=${TARGET_GLOBAL_BS} grad_accum=${grad_accum}"
echo "[gpu07 spb2align] repeated_diffusion_steps=${REPEAT_STEPS} eval_interval=${EVAL_INTERVAL}"
echo "[gpu07 spb2align] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29523 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.vlm_packing false \
  --framework.action_model.repeated_diffusion_steps ${REPEAT_STEPS} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 12 \
  --trainer.save_interval 250 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval ${EVAL_INTERVAL} \
  --trainer.skip_non_finite_loss true \
  --trainer.max_grad_skips 50 \
  --trainer.max_grad_norm_skip_threshold 1000 \
  --framework.action_model.dit_pred_abs_max 50 \
  --framework.action_model.dit_pred_error_abs_max 100 \
  --trainer.use_spb2vl_lr_groups true \
  --trainer.learning_rate.spb2_lm 5e-5 \
  --trainer.learning_rate.spb2_merge 5e-5 \
  --trainer.learning_rate.spb2_ve 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.optimizer.weight_decay 0.1 \
  --trainer.optimizer.betas "[0.9,0.99]" \
  --trainer.optimizer.eps 1e-5 \
  --trainer.freeze_modules "" \
  --trainer.enable_gradient_checkpointing true \
  --framework.qwenvl.rmsnorm_fp32_forward true \
  --framework.qwenvl.verify_fp32_after_prepare true \
  --framework.qwenvl.vlm_hidden_collapse_row_ratio 0.3 \
  "${RESUME_ARGS[@]}" \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jianingliu \
  > "${LOG}" 2>&1 &

echo "[gpu07 spb2align] started pid=$! log=${LOG}"
echo "[gpu07 spb2align] tail -f ${LOG}"
