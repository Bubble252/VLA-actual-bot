#!/usr/bin/env bash
# LIBERO 单阶段全训：SpikeBrain-2.0-VL 预训练权重 + DiT 随机初始化，VLM+DiT 同时训练
# 对齐 SPB2-VL SFT 超参/数值策略（DeepSpeed ZeRO-2）；不加载 starVLA stage1/stage2 ckpt
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
STARVLA_CONDA_ENV="/9950backfile/zhangjh/.conda/envs/starvla_cu124"
cd "${STARVLA_ROOT}"

stop_own_starvla_train() {
  local starvla_bin="${STARVLA_CONDA_ENV}/bin"
  local libero_cfg="${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
  local pid cmd
  is_own_libero_train_cmd() {
    local c="$1"
    echo "${c}" | grep -q "${starvla_bin}" \
      && echo "${c}" | grep -q "train_starvla.py" \
      && echo "${c}" | grep -q "${libero_cfg}"
  }
  for pid in $(pgrep -f "train_starvla.py" 2>/dev/null || true); do
    cmd=$(ps -p "${pid}" -o cmd= 2>/dev/null || true)
    if is_own_libero_train_cmd "${cmd}"; then
      echo "[full spb2align] 停止本仓库 LIBERO worker pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in $(pgrep -f "accelerate launch.*train_starvla.py" 2>/dev/null || true); do
    cmd=$(ps -p "${pid}" -o cmd= 2>/dev/null || true)
    if is_own_libero_train_cmd "${cmd}"; then
      echo "[full spb2align] 停止本仓库 LIBERO launcher pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  done
  sleep 2
}
stop_own_starvla_train

# --- CUDA 环境：PATH / LD_LIBRARY_PATH 与 PyTorch 自带 CUDA 对齐 ---
# 默认 12.1（本机 torch 2.5.1+cu121）；可 export CUDA_HOME=/usr/local/cuda-11.7 覆盖
setup_cuda_env() {
  local cuda_home="${CUDA_HOME:-}"
  if [ -z "${cuda_home}" ]; then
    for candidate in /usr/local/cuda-12.1 /usr/local/cuda-11.7 /usr/local/cuda; do
      if [ -d "${candidate}/bin" ] && [ -d "${candidate}/lib64" ]; then
        cuda_home="${candidate}"
        break
      fi
    done
  fi
  if [ -z "${cuda_home}" ] || [ ! -d "${cuda_home}/bin" ]; then
    echo "[full spb2align] WARN: CUDA_HOME not found; skip PATH/LD_LIBRARY_PATH patch"
    return 0
  fi
  export CUDA_HOME="${cuda_home}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  echo "[full spb2align] CUDA_HOME=${CUDA_HOME}"
  echo "[full spb2align] nvcc: $(nvcc --version 2>/dev/null | grep 'release' || echo 'nvcc not in PATH')"
}
setup_cuda_env

# starVLA 专用 NVIDIA JIT 缓存 → /9950backfile/zhangjh/wh/.nv/ComputeCache（不动 ~/.nv）
# shellcheck source=examples/LIBERO/train_files/setup_starvla_cuda_cache.sh
source "${STARVLA_ROOT}/examples/LIBERO/train_files/setup_starvla_cuda_cache.sh"
setup_starvla_cuda_cache

export PATH="${STARVLA_CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_API_KEY="4f3ad82c189f15c159dc9f3e38395ef7a7051584"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="starVLA"
export WANDB_ENTITY="jianingliu"
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
# 须在 import torch 之前生效；accelerate 子进程会继承
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
mkdir -p "${TRITON_CACHE_DIR}"

SPB2_HF_MODULES_CACHE="${HOME}/.cache/huggingface/modules/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL"
if [ -d "${SPB2_HF_MODULES_CACHE}" ]; then
  echo "[full spb2align] 清除 SpikeBrain HF 模块缓存: ${SPB2_HF_MODULES_CACHE}"
  rm -rf "${SPB2_HF_MODULES_CACHE}"
fi
"${STARVLA_CONDA_ENV}/bin/python" -c "from transformers import AutoConfig; AutoConfig.from_pretrained('${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL', trust_remote_code=True); print('[full spb2align] HF modules cache warmed')"

export CUDA_VISIBLE_DEVICES=${STARVLA_CUDA_DEVICES:-1,2,3,4,5,6}
NUM_PROCESSES=${STARVLA_NUM_PROCESSES:-6}
PER_DEVICE_BS=1
# global_bs 只影响 grad_accum，不降低单卡峰值显存（per_device_bs=1 时）。
# 可通过环境变量覆盖，例如 TARGET_GLOBAL_BS=84 bash ...
TARGET_GLOBAL_BS=${TARGET_GLOBAL_BS:-84}

run_date=$(date +%Y%m%d)
run_time=$(date +%H%M)
run_id=${STARVLA_RUN_ID:-${run_date}_${run_time}_libero_all_spb2_gr00t_full_spb2align}

# 默认：仅 SPB2-VL 预训练权重 + DiT 随机初始化（不加载 starVLA steps_* ckpt）。
# 续训示例：
#   STARVLA_RUN_ID=20260608_1842_libero_all_spb2_gr00t_full_spb2align \
#   STARVLA_RESUME_CKPT=.../steps_250_pytorch_model.pt STARVLA_RESUME_STEP=250 \
#   bash run_libero_train_all_ljn_full_spb2align.sh
RESUME_CKPT="${STARVLA_RESUME_CKPT:-}"
RESUME_STEP="${STARVLA_RESUME_STEP:-0}"
RESUME_ARGS=()

if [ -n "${RESUME_CKPT}" ]; then
  if [ ! -f "${RESUME_CKPT}" ]; then
    echo "[full spb2align] ERROR: STARVLA_RESUME_CKPT not found: ${RESUME_CKPT}"
    exit 1
  fi
  RESUME_ARGS=(
    --trainer.is_resume false
    --trainer.pretrained_checkpoint "${RESUME_CKPT}"
    --trainer.resume_step "${RESUME_STEP}"
  )
  echo "[full spb2align] starVLA ckpt resume: ${RESUME_CKPT} step=${RESUME_STEP}"
else
  echo "[full spb2align] init: SpikeBrain-2.0-VL pretrained weights + DiT random init (no starVLA ckpt)"
fi
run_root_dir=./results/Checkpoints

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run_libero_full_spb2align_$(date +%Y%m%d_%H%M%S).log"

echo "[full spb2align] run_id=${run_id}"
echo "[full spb2align] base_vlm=${base_vlm}"
echo "[full spb2align] GPU=${CUDA_VISIBLE_DEVICES} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[full spb2align] per_device_bs=${PER_DEVICE_BS} target_global_bs=${TARGET_GLOBAL_BS} grad_accum=$((TARGET_GLOBAL_BS / (PER_DEVICE_BS * NUM_PROCESSES)))"
echo "[full spb2align] SPB2 LR: lm/merge=5e-5 ve=1e-5 action=1e-4 wd=0.1 betas=0.9/0.99"
echo "[full spb2align] gradient_checkpointing=true (cold-start forward 更稳; 首步 backward 慢, timeout=1800s)"
echo "[full spb2align] VLM bf16 + rmsnorm_fp32_forward; DiT fp32 compute island; skip_non_finite_loss=true"
echo "[full spb2align] grad skip: non-finite OR grad_norm>1000; DiT extreme pred resample; save_interval=250"
echo "[full spb2align] NCCL/TORCH timeout=1800s (InitProcessGroupKwargs in train_starvla.py)"
echo "[full spb2align] CUDA_HOME=${CUDA_HOME:-unset} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
echo "[full spb2align] output_dir=${STARVLA_ROOT}/${output_dir}"
echo "[full spb2align] log=${LOG}"

nohup accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29509 \
  --num_processes ${NUM_PROCESSES} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix libero_all \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 12 \
  --trainer.save_interval 250 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 50 \
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
  --trainer.grad_log.enabled false \
  --trainer.numerical_probe.rmsnorm_hook_check false \
  --trainer.numerical_probe.layer_hidden_check false \
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

TRAIN_PID=$!
echo "[full spb2align] started with nohup, pid=${TRAIN_PID}"
echo "[full spb2align] follow log: tail -f ${LOG}"
