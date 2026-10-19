#!/usr/bin/env bash
# 第二阶段：从 steps_6000 全量微调（不冻结 VLM）
# GPU 1-6，单卡 bs=1，global=126（1×6×21=126；bs≥2 全训会 VLM hidden NaN，见 docs/STAGE2_VLM_NAN_INVESTIGATION.md）
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

# SpikeBrain HF 模块缓存（与 run_libero_train_all_ljn.sh 一致）
SPB2_HF_MODULES_CACHE="${HOME}/.cache/huggingface/modules/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL"
if [ -d "${SPB2_HF_MODULES_CACHE}" ]; then
  echo "[stage2 full] 清除 SpikeBrain HF 模块缓存: ${SPB2_HF_MODULES_CACHE}"
  rm -rf "${SPB2_HF_MODULES_CACHE}"
fi
# 单进程预热 HF 模块缓存，避免 6 rank 并行写缓存导致 SPB2VLConfig 损坏
python3.10 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL', trust_remote_code=True); print('[stage2 full] HF modules cache warmed')"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
NUM_PROCESSES=6
PER_DEVICE_BS=1
TARGET_GLOBAL_BS=126

run_date=$(date +%Y%m%d)
run_time=$(date +%H%M)
run_id=${run_date}_${run_time}_libero_all_spb2_gr00t_stage2_full_from6k
run_root_dir=./results/Checkpoints

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_ROOT}/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_libero_stage2_full_from6k_$(date +%Y%m%d_%H%M%S).log"

echo "[stage2 full] run_id=${run_id}"
echo "[stage2 full] GPU=${CUDA_VISIBLE_DEVICES} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[stage2 full] per_device_bs=${PER_DEVICE_BS} target_global_bs=${TARGET_GLOBAL_BS} grad_accum=$((TARGET_GLOBAL_BS / (PER_DEVICE_BS * NUM_PROCESSES)))"
echo "[stage2 full] ckpt=${pretrained_ckpt}"
echo "[stage2 full] output_dir=${STARVLA_ROOT}/${output_dir}"
echo "[stage2 full] log=${LOG}"
echo "[stage2 full] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[stage2 full] VLM: 全训；fp32敏感层+GDN；Liger仅RoPE/SwiGLU；NaN即停训；GC=false"

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
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 200 \
  --trainer.is_resume false \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.resume_step 0 \
  --trainer.enable_gradient_checkpointing false \
  --trainer.freeze_modules "" \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jianingliu \
  > "${LOG}" 2>&1 &

TRAIN_PID=$!
echo "[stage2 full] started with nohup, pid=${TRAIN_PID}"
echo "[stage2 full] follow log: tail -f ${LOG}"
