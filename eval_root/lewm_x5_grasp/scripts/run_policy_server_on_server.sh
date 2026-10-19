#!/usr/bin/env bash
set -eo pipefail

# Run on the GPU server (not the laptop connected to the robot).
#
# Usage examples:
#   CKPT=4in1 bash run_policy_server_on_server.sh
#   CKPT=ljn82500 GPU_IDS=0 bash run_policy_server_on_server.sh
#   CKPT_PATH=/path/to/model.pt UNNORM_KEY=franka bash run_policy_server_on_server.sh

STARVLA_ROOT="${STARVLA_ROOT:-/9950backfile/zhangjh/wh/starvla_wh/starVLA}"
CKPT="${CKPT:-4in1}"

case "${CKPT}" in
  4in1)
    DEFAULT_CKPT="${STARVLA_ROOT}/playground/Pretrained_models/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
    ;;
  ljn82500)
    DEFAULT_CKPT="${STARVLA_ROOT}/results/ljn/v1/steps_82500/steps_82500_pytorch_model.pt"
    ;;
  *)
    DEFAULT_CKPT="${CKPT}"
    ;;
esac

export CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
export UNNORM_KEY="${UNNORM_KEY:-franka}"
export GPU_IDS="${GPU_IDS:-0}"
export BASE_PORT="${BASE_PORT:-5694}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

echo "Starting policy server"
echo "  CKPT_PATH=${CKPT_PATH}"
echo "  UNNORM_KEY=${UNNORM_KEY}"
echo "  BASE_PORT=${BASE_PORT}"
echo "  GPU_IDS=${GPU_IDS}"

cd "${STARVLA_ROOT}"
bash examples/LIBERO/eval_files/run_policy_server.sh
