#!/bin/bash
# Run official Qwen baseline on all 4 LIBERO suites (Spatial / Object / Goal / Long=libero_10).
#
# Prerequisite: download checkpoint once, e.g.
#   cd /9950backfile/zhangjh/wh/starvla_wh/starVLA
#   huggingface-cli download StarVLA/Qwen3-VL-GR00T-LIBERO-4in1 \
#     --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-GR00T-LIBERO-4in1
#
# Terminal 1 (starvla env, from repo root):
#   CKPT_PATH=playground/Pretrained_models/StarVLA/Qwen3-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt \
#   GPU_IDS=5 bash examples/LIBERO/eval_files/run_policy_server.sh
#
# Terminal 2 (libero env, from repo root):
#   bash examples/LIBERO/eval_files/eval_libero_baseline.sh
#
# Override CKPT_PATH / STEPS_NUM / RENDER_GPU_IDS / GPU_IDS as needed.

set -euo pipefail

STARVLA_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${STARVLA_ROOT}"

# Qwen3-VL-GR00T matches README table & same QwenGR00T head as SPB2 training.
# Local fallback: trash/hf_models/StarVLA-Qwen2.5-VL-GR00T-LIBERO-4in1 (older VLM, not ideal).
DEFAULT_QWEN3_GR00T="${STARVLA_ROOT}/playground/Pretrained_models/StarVLA/Qwen3-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
DEFAULT_QWEN25_GR00T="${STARVLA_ROOT}/trash/hf_models/StarVLA-Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"

if [[ -n "${CKPT_PATH:-}" ]]; then
  baseline_ckpt="${CKPT_PATH}"
elif [[ -f "${DEFAULT_QWEN3_GR00T}" ]]; then
  baseline_ckpt="${DEFAULT_QWEN3_GR00T}"
elif [[ -f "${DEFAULT_QWEN25_GR00T}" ]]; then
  echo "[eval_libero_baseline.sh] Qwen3 ckpt not found, using local Qwen2.5-VL-GR00T"
  baseline_ckpt="${DEFAULT_QWEN25_GR00T}"
else
  echo "[eval_libero_baseline.sh] ERROR: no baseline checkpoint found."
  echo "  Download: huggingface-cli download StarVLA/Qwen3-VL-GR00T-LIBERO-4in1 \\"
  echo "    --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-GR00T-LIBERO-4in1"
  exit 1
fi

steps_num="${STEPS_NUM:-$(basename "${baseline_ckpt}" | sed -n 's/^steps_\([0-9]*\)_pytorch_model\.pt$/\1/p')}"
steps_num="${steps_num:-30000}"

echo "[eval_libero_baseline.sh] ckpt: ${baseline_ckpt}"
echo "[eval_libero_baseline.sh] steps label: ${steps_num}"

SUITES=(libero_spatial libero_object libero_goal libero_10)
for suite in "${SUITES[@]}"; do
  echo "========== ${suite} =========="
  CKPT_PATH="${baseline_ckpt}" \
  TASK_SUITE_NAME="${suite}" \
  STEPS_NUM="${steps_num}" \
  UNNORM_KEY="${UNNORM_KEY:-franka}" \
  RENDER_GPU_IDS="${RENDER_GPU_IDS:-7}" \
  bash examples/LIBERO/eval_files/eval_libero.sh
done

echo "[eval_libero_baseline.sh] All 4 suites finished."
