#!/usr/bin/env bash
# Shared checkpoint resolver for LIBERO / LIBERO-plus / LIBERO-PRO eval scripts.
# Usage: source examples/LIBERO/eval_files/_resolve_eval_ckpt.sh && resolve_eval_ckpt

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

DEFAULT_QWEN25_GR00T="${STARVLA_ROOT}/playground/Pretrained_models/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt"
DEFAULT_SPB2_PT="${STARVLA_ROOT}/results/ljn/v1/steps_82500/steps_82500_pytorch_model.pt"
# Config for ljn/v1 layout: run_dir = parent of steps_82500/
DEFAULT_SPB2_CONFIG_RUN="${STARVLA_ROOT}/results/Checkpoints/20260608_1842_libero_all_spb2_gr00t_full_spb2align"

prepare_spb2_ckpt_configs() {
  local ckpt_path="$1"
  local run_dir
  run_dir="$(cd "$(dirname "${ckpt_path}")/.." && pwd)"
  local config_src="${SPB2_CONFIG_RUN:-${DEFAULT_SPB2_CONFIG_RUN}}"

  if [[ ! -f "${config_src}/config.yaml" ]]; then
    echo "[resolve_ckpt] ERROR: SPB2 config run not found: ${config_src}"
    exit 1
  fi

  for f in config.yaml dataset_statistics.json; do
    if [[ ! -f "${run_dir}/${f}" ]]; then
      echo "[resolve_ckpt] copy ${f} -> ${run_dir}/"
      cp "${config_src}/${f}" "${run_dir}/${f}"
    fi
  done
}

resolve_eval_ckpt() {
  local model="${MODEL:-custom}"

  if [[ -n "${CKPT_PATH:-}" ]]; then
    :
  elif [[ "${model}" == "baseline" || "${model}" == "qwen" ]]; then
    CKPT_PATH="${DEFAULT_QWEN25_GR00T}"
    STEPS_NUM="${STEPS_NUM:-30000}"
  elif [[ "${model}" == "spb2" || "${model}" == "ljn" ]]; then
    CKPT_PATH="${DEFAULT_SPB2_PT}"
    STEPS_NUM="${STEPS_NUM:-82500}"
    prepare_spb2_ckpt_configs "${CKPT_PATH}"
  else
    echo "[resolve_ckpt] Set CKPT_PATH=... or MODEL=baseline|spb2"
    exit 1
  fi

  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[resolve_ckpt] ERROR: checkpoint not found: ${CKPT_PATH}"
    exit 1
  fi

  STEPS_NUM="${STEPS_NUM:-$(basename "${CKPT_PATH}" | sed -n 's/^steps_\([0-9]*\)_pytorch_model\.pt$/\1/p')}"
  UNNORM_KEY="${UNNORM_KEY:-franka}"

  echo "[resolve_ckpt] MODEL=${model}"
  echo "[resolve_ckpt] CKPT_PATH=${CKPT_PATH}"
  echo "[resolve_ckpt] STEPS_NUM=${STEPS_NUM}"
}
