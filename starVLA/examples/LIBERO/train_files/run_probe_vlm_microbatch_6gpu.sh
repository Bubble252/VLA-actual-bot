#!/usr/bin/env bash
# 对比 bs=1 vs bs=2+VLM micro-batch（六卡，使用正式训练的 grad_accum，不用 grad_accum=1）
#
# bs=1: global=126, grad_accum=21  (1×6×21)
# bs=2: global=132, grad_accum=11  (2×6×11)
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
PROBE_STEPS=5
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"

config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
pretrained_ckpt=${STARVLA_ROOT}/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt
LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/probe_vlm_microbatch_6gpu_$(date +%Y%m%d_%H%M%S).log"

echo "[microbatch probe] GPU=${CUDA_VISIBLE_DEVICES} steps=${PROBE_STEPS}" | tee "${MASTER_LOG}"
echo "[microbatch probe] 仅使用正式训练 grad_accum（126/21 与 132/11）" | tee -a "${MASTER_LOG}"

summarize_run() {
  local tag="$1"
  local logfile="$2"
  local per_device_bs="$3"
  local global_bs="$4"
  local grad_accum=$((global_bs / (per_device_bs * NUM_PROCESSES)))

  echo "" | tee -a "${MASTER_LOG}"
  echo "========== ${tag} ==========" | tee -a "${MASTER_LOG}"
  echo "per_device_bs=${per_device_bs} global_bs=${global_bs} grad_accum=${grad_accum}" | tee -a "${MASTER_LOG}"

  local vlm_guard vlm_micro
  vlm_guard=$(grep -c "VLM guard" "${logfile}" 2>/dev/null || true)
  vlm_micro=$(grep -c "VLM micro-batch" "${logfile}" 2>/dev/null || true)
  echo "VLM guard warnings: ${vlm_guard}" | tee -a "${MASTER_LOG}"
  echo "VLM micro-batch log lines: ${vlm_micro}" | tee -a "${MASTER_LOG}"

  python3 - "${logfile}" "${PROBE_STEPS}" "${global_bs}" <<'PY' | tee -a "${MASTER_LOG}"
import re, sys
log_path, steps, global_bs = sys.argv[1:4]
steps, global_bs = int(steps), int(global_bs)
text = open(log_path, encoding="utf-8", errors="ignore").read()
model_times = [float(x) for x in re.findall(r"'model_time': ([0-9.]+)", text)]
model_times = model_times[:steps]
if not model_times:
    print("model_time: N/A")
    sys.exit(0)
avg = sum(model_times) / len(model_times)
total = sum(model_times)
samples = steps * global_bs
tput = samples / total if total > 0 else 0.0
print(f"model_time avg={avg:.3f}s/optimizer_step step_total={total:.3f}s over {len(model_times)} steps")
print(f"throughput≈{tput:.3f} samples/s ({samples} global samples in {total:.1f}s)")
bad = re.findall(r"'vlm_hidden_bad_elems': ([0-9]+)", text)
if bad:
    print(f"vlm_hidden_bad_elems per optimizer step: {bad[:steps]}")
PY
}

run_case() {
  local tag="$1"
  local per_device_bs="$2"
  local global_bs="$3"
  local micro_bs="$4"
  local port="$5"

  local run_id="probe_microbatch_${tag}_$(date +%Y%m%d_%H%M%S)"
  local exp_log="${LOG_DIR}/${run_id}.log"

  echo "" | tee -a "${MASTER_LOG}"
  echo ">>> Running ${tag} (per_device_bs=${per_device_bs}, global_bs=${global_bs}, vlm_micro_batch_size=${micro_bs})" | tee -a "${MASTER_LOG}"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port "${port}" \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name QwenGR00T \
    --framework.qwenvl.base_vlm "${base_vlm}" \
    --framework.qwenvl.vlm_micro_batch_size "${micro_bs}" \
    --datasets.vla_data.per_device_batch_size "${per_device_bs}" \
    --datasets.vla_data.data_mix libero_all \
    --trainer.target_global_batch_size "${global_bs}" \
    --trainer.epochs 1 \
    --trainer.save_interval 999999 \
    --trainer.eval_interval 999999 \
    --trainer.logging_frequency 1 \
    --trainer.is_resume false \
    --trainer.pretrained_checkpoint "${pretrained_ckpt}" \
    --trainer.resume_step 0 \
    --trainer.enable_gradient_checkpointing true \
    --trainer.freeze_modules "" \
    --run_root_dir ./results/Checkpoints \
    --run_id "${run_id}" \
    > "${exp_log}" 2>&1 || true

  summarize_run "${tag}" "${exp_log}" "${per_device_bs}" "${global_bs}"
}

run_case "bs1_global126" 1 126 0 29522
run_case "bs2_global132_micro" 2 132 1 29523

echo "" | tee -a "${MASTER_LOG}"
echo "[microbatch probe] done. master_log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"
