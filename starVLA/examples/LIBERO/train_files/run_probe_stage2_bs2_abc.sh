#!/usr/bin/env bash
# 六卡 bs=2 探测：复用 train_starvla + DeepSpeed，快速跑几步后解析 VLM guard。
#
# A 已结论：bs=2 + freeze → VLM 无 NaN（见 probe_stage2_bs2_expA_20260605_222329.log）
# B: bs=2 + 全训 + gradient_checkpointing=false
# C: bs=2 + 全训 + use_liger_kernel=false
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

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
NUM_PROCESSES=6
PER_DEVICE_BS=2
TARGET_GLOBAL_BS=132   # 2×6×11=132，与正式 bs=2 全训一致
PROBE_STEPS=3
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_ROOT}/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt

LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/probe_stage2_bs2_bc_6gpu_$(date +%Y%m%d_%H%M%S).log"

echo "[probe 6gpu] GPU=${CUDA_VISIBLE_DEVICES} processes=${NUM_PROCESSES}" | tee "${MASTER_LOG}"
echo "[probe 6gpu] per_device_bs=${PER_DEVICE_BS} global_bs=${TARGET_GLOBAL_BS} grad_accum=1 probe_steps=${PROBE_STEPS}" | tee -a "${MASTER_LOG}"
echo "[probe 6gpu] skip A (already done: freeze+bs2 -> 0 VLM guard)" | tee -a "${MASTER_LOG}"
echo "[probe 6gpu] master_log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"

summarize_vlm_guard() {
  local logfile="$1"
  local exp="$2"
  local count
  count=$(grep -c "VLM guard" "${logfile}" 2>/dev/null || true)
  echo "" | tee -a "${MASTER_LOG}"
  echo "========== ${exp} summary ==========" | tee -a "${MASTER_LOG}"
  if [ "${count}" -eq 0 ]; then
    echo "[${exp}] VLM guard: 0 warnings -> PASS (no NaN hidden)" | tee -a "${MASTER_LOG}"
  else
    echo "[${exp}] VLM guard: ${count} warnings -> FAIL (NaN hidden detected)" | tee -a "${MASTER_LOG}"
    grep "VLM guard" "${logfile}" | head -5 | tee -a "${MASTER_LOG}"
    if [ "${count}" -gt 5 ]; then
      echo "  ... (${count} total, showing first 5)" | tee -a "${MASTER_LOG}"
    fi
  fi
  if grep -q "illegal memory access\|CUDA error" "${logfile}" 2>/dev/null; then
    echo "[${exp}] CUDA crash detected!" | tee -a "${MASTER_LOG}"
    grep -E "illegal memory access|CUDA error" "${logfile}" | head -3 | tee -a "${MASTER_LOG}"
  fi
}

run_exp() {
  local exp="$1"
  local freeze="$2"
  local grad_ckpt="$3"
  local use_liger="$4"
  local port="$5"
  local desc="$6"

  local run_id
  run_id="probe_stage2_bs2_exp${exp}_$(date +%Y%m%d_%H%M%S)"
  local exp_log="${LOG_DIR}/${run_id}.log"

  echo "" | tee -a "${MASTER_LOG}"
  echo "################################################################" | tee -a "${MASTER_LOG}"
  echo "# Experiment ${exp}: ${desc}" | tee -a "${MASTER_LOG}"
  echo "# freeze_modules=${freeze} grad_ckpt=${grad_ckpt} use_liger=${use_liger}" | tee -a "${MASTER_LOG}"
  echo "# log=${exp_log}" | tee -a "${MASTER_LOG}"
  echo "################################################################" | tee -a "${MASTER_LOG}"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port "${port}" \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name QwenGR00T \
    --framework.qwenvl.base_vlm "${base_vlm}" \
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
    --datasets.vla_data.data_mix libero_all \
    --trainer.target_global_batch_size "${TARGET_GLOBAL_BS}" \
    --trainer.epochs 1 \
    --trainer.save_interval 999999 \
    --trainer.eval_interval 999999 \
    --trainer.logging_frequency 1 \
    --trainer.is_resume false \
    --trainer.pretrained_checkpoint "${pretrained_ckpt}" \
    --trainer.resume_step 0 \
    --trainer.enable_gradient_checkpointing "${grad_ckpt}" \
    --framework.qwenvl.use_liger_kernel "${use_liger}" \
    --trainer.freeze_modules "${freeze}" \
    --run_root_dir ./results/Checkpoints \
    --run_id "${run_id}" \
    > "${exp_log}" 2>&1

  local rc=$?
  echo "[probe 6gpu] ${exp} finished rc=${rc}" | tee -a "${MASTER_LOG}"
  summarize_vlm_guard "${exp_log}" "${exp}"
  return "${rc}"
}

run_exp B "" false true 29511 "full train, GC off" || true
run_exp C "" true false 29512 "full train, liger off" || true

echo "" | tee -a "${MASTER_LOG}"
echo "[probe 6gpu] B/C done. master_log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"
