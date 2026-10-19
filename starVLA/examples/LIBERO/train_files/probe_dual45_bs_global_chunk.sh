#!/usr/bin/env bash
# GPU 4,5：扫描 per_device_bs × target_global_batch × action_forward_chunk_size
# 目标：在显存允许下提高吞吐（samples/s），缩短 12 epoch 总墙钟时间
set -euo pipefail

STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
NUM_PROCESSES=2
PROBE_STEPS="${PROBE_STEPS:-25}"
export STARVLA_PROBE_MAX_STEPS="${PROBE_STEPS}"
WARMUP_SKIP="${WARMUP_SKIP:-8}"

CONFIG_YAML="${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
BASE_VLM="${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL"
LOG_DIR="${STARVLA_ROOT}/results/Checkpoints/train_logs"
mkdir -p "${LOG_DIR}"

SUMMARY="${LOG_DIR}/probe_dual45_bs_global_chunk_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "=== dual45 bs/global/chunk probe | GPU=${CUDA_VISIBLE_DEVICES} steps=${PROBE_STEPS} | $(date -Iseconds) ===" | tee "${SUMMARY}"
echo "baseline(ref): per_device=32 global=128 chunk=128 grad_accum=2 dit_B=192 chunks=2" | tee -a "${SUMMARY}"
echo "known: bs64 OOM; bs48+global96 OK(~57GB); chunk192@bs32 NaN" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

# 格式: name|per_device_bs|global_bs|chunk_size
CASES=(
  "base32|32|128|128"
  "bs48_g96_c128|48|96|128"
  "bs48_g192_c128|48|192|128"
  "bs48_g192_c192|48|192|192"
  "bs48_g192_c288|48|192|288"
  "bs48_g384_c288|48|384|288"
  "bs40_g160_c240|40|160|240"
  "bs44_g176_c264|44|176|264"
)

PORT_BASE=29840
CASE_IDX=0

parse_metrics() {
  local log="$1"
  local avg_mt peak_gb steps_per_epoch grad_accum dit_chunks
  avg_mt=$(grep -oE "'model_time': [0-9.]+" "${log}" 2>/dev/null | tail -n 12 | awk -F': ' '{s+=$2; n++} END{if(n>0) printf "%.3f", s/n; else print "NA"}')
  peak_gb=$(grep -oE "'cuda/max_allocated_gb': [0-9.]+" "${log}" 2>/dev/null | awk -F': ' '{if($2+0>m)m=$2+0} END{printf "%.2f", m+0}')
  steps_per_epoch=$(grep -oE "steps/epoch=[0-9]+" "${log}" 2>/dev/null | head -1 | grep -oE "[0-9]+" || echo "NA")
  grad_accum=$(grep -oE "grad_accum\([0-9]+\)" "${log}" 2>/dev/null | head -1 | grep -oE "[0-9]+" || echo "NA")
  dit_chunks=$(grep -oE "'dit_num_chunks': [0-9]+" "${log}" 2>/dev/null | tail -1 | grep -oE "[0-9]+$" || echo "NA")
  echo "${avg_mt}|${peak_gb}|${steps_per_epoch}|${grad_accum}|${dit_chunks}"
}

for ENTRY in "${CASES[@]}"; do
  IFS='|' read -r NAME PER_BS GLOBAL_BS CHUNK <<< "${ENTRY}"
  CASE_IDX=$((CASE_IDX + 1))
  PORT=$((PORT_BASE + CASE_IDX))
  RUN_ID="probe45_${NAME}_$(date +%H%M%S)"
  LOG="${LOG_DIR}/${RUN_ID}.log"

  echo "" | tee -a "${SUMMARY}"
  echo "--- ${NAME}: per_device=${PER_BS} global=${GLOBAL_BS} chunk=${CHUNK} ---" | tee -a "${SUMMARY}"
  echo "log=${LOG}" | tee -a "${SUMMARY}"

  set +e
  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_port "${PORT}" \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --framework.name QwenGR00T \
    --framework.qwenvl.base_vlm "${BASE_VLM}" \
    --framework.action_model.action_forward_chunk_size "${CHUNK}" \
    --datasets.vla_data.per_device_batch_size "${PER_BS}" \
    --datasets.vla_data.data_mix libero_all \
    --trainer.target_global_batch_size "${GLOBAL_BS}" \
    --trainer.num_warmup_steps 0 \
    --trainer.save_interval 100000 \
    --trainer.eval_interval 100000 \
    --trainer.logging_frequency 5 \
    --trainer.freeze_modules qwen_vl_interface \
    --run_root_dir ./results/Checkpoints \
    --run_id "${RUN_ID}" \
    2>&1 | tee "${LOG}"
  EC=${PIPESTATUS[0]}
  set -e

  if grep -qiE 'CUDA out of memory|OutOfMemoryError' "${LOG}"; then
    RESULT="OOM"
  elif grep -qiE 'action_loss NaN|non-finite|skip_non_finite' "${LOG}" && grep -qiE 'NaN|non-finite' "${LOG}"; then
    RESULT="NaN"
  elif grep -qE 'action_dit_loss' "${LOG}" && [[ "${EC}" -eq 0 ]]; then
    RESULT="OK"
  elif [[ "${EC}" -ne 0 ]]; then
    RESULT="FAIL(exit=${EC})"
  else
    RESULT="UNKNOWN"
  fi

  IFS='|' read -r AVG_MT PEAK_GB STEPS_EPOCH GRAD_ACCUM DIT_CHUNKS <<< "$(parse_metrics "${LOG}")"
  # 吞吐粗估: global_bs / (avg_model_time * grad_accum) samples/s（每 optimizer step）
  if [[ "${AVG_MT}" != "NA" && "${GRAD_ACCUM}" != "NA" && "${GLOBAL_BS}" != "NA" ]]; then
    THROUGHPUT=$(python3 - <<PY
gb=${GLOBAL_BS}; ga=${GRAD_ACCUM}; mt=${AVG_MT}
if mt > 0 and ga > 0:
    print(f"{gb/(mt*ga):.2f}")
else:
    print("NA")
PY
)
    EPOCH_HOURS=$(python3 - <<PY
import math
samples=405876; gb=${GLOBAL_BS}; mt=${AVG_MT}; ga=${GRAD_ACCUM}
if mt > 0:
    steps=math.ceil(samples/gb)
    sec=steps*mt*ga
    print(f"{sec/3600:.1f}")
else:
    print("NA")
PY
)
  else
    THROUGHPUT="NA"
    EPOCH_HOURS="NA"
  fi

  echo "RESULT ${NAME}: ${RESULT} | avg_model_time=${AVG_MT}s grad_accum=${GRAD_ACCUM} peak_alloc=${PEAK_GB}GB dit_chunks=${DIT_CHUNKS} throughput~${THROUGHPUT} samp/s epoch~${EPOCH_HOURS}h" | tee -a "${SUMMARY}"

  sleep 8
  python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
done

echo "" | tee -a "${SUMMARY}"
echo "=== done | summary=${SUMMARY} ===" | tee -a "${SUMMARY}"
