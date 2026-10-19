#!/usr/bin/env bash
# starVLA 专用 NVIDIA JIT 计算缓存（不占用 ~/.nv，避免影响同账号其他任务）
#
# CUDA 默认写 ~/.nv/ComputeCache；通过 CUDA_CACHE_PATH 改到 wh 目录下。
# 用法（在 import torch / accelerate 之前）：
#   source examples/LIBERO/train_files/setup_starvla_cuda_cache.sh
#   setup_starvla_cuda_cache
#
# 仅清 starVLA 缓存（不动 ~/.nv）：
#   STARVLA_CLEAR_NV_CACHE=1 setup_starvla_cuda_cache
# 跳过清理：
#   STARVLA_CLEAR_NV_CACHE=0 setup_starvla_cuda_cache

setup_starvla_cuda_cache() {
  local cache_root="${STARVLA_NV_CACHE_ROOT:-/9950backfile/zhangjh/wh/.nv}"
  export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${cache_root}/ComputeCache}"
  mkdir -p "${CUDA_CACHE_PATH}"

  if [ "${STARVLA_CLEAR_NV_CACHE:-1}" != "0" ] && [ -d "${CUDA_CACHE_PATH}" ]; then
    echo "[starvla cuda cache] 清除 starVLA 专用缓存: ${CUDA_CACHE_PATH}"
    rm -rf "${CUDA_CACHE_PATH:?}"/*
  fi

  echo "[starvla cuda cache] CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
}
