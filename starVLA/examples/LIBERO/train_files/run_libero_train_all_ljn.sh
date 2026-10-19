#!/usr/bin/env bash
set -euo pipefail

###########################################################################################
# 备忘文档：examples/LIBERO/docs/SPB2_LIBERO_TRAINING_NOTES.md
# （SPB2 环境 / Liger / 排错 / 路径索引）
###########################################################################################
# 环境与工作目录（每次运行本脚本会自动设置，无需手动 export/cd）
###########################################################################################
STARVLA_ROOT="/9950backfile/zhangjh/wh/starvla_wh/starVLA"
cd "${STARVLA_ROOT}"

export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"

export WANDB_API_KEY="4f3ad82c189f15c159dc9f3e38395ef7a7051584"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_PROJECT="starVLA"
export WANDB_ENTITY="jianingliu"
export WANDB_MODE="online"

# 须在 import torch 之前生效。部分 PyTorch/CUDA 不支持 expandable_segments（log 会提示可忽略）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DeepSpeed Triton autotune 缓存（项目内，便于查找/清理；项目在 NFS 上时 log 可能仍有 NFS 提示）
export TRITON_CACHE_DIR="/data/local/zhangjh/.cache/triton_autotune"
mkdir -p "${TRITON_CACHE_DIR}"

echo "[run_libero_train] STARVLA_ROOT=${STARVLA_ROOT}"
echo "[run_libero_train] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[run_libero_train] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[run_libero_train] python=$(command -v python)"

###########################################################################################

# SpikeBrain 使用 trust_remote_code，会把 SpikeBrain-2.0-VL/*.py 复制到 HF 缓存再 import。
# 若缓存里是旧版自定义代码，会与磁盘上 spikebrain/SpikeBrain-2.0-VL 不一致（例如加载类、forward）。
# 每次启动训练前清一次，强制从当前本地模型目录重新同步。（稳定后可注释掉本段）
SPB2_HF_MODULES_CACHE="${HOME}/.cache/huggingface/modules/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL"
if [ -d "${SPB2_HF_MODULES_CACHE}" ]; then
  echo "[run_libero_train] 清除 SpikeBrain HF 模块缓存: ${SPB2_HF_MODULES_CACHE}"
  rm -rf "${SPB2_HF_MODULES_CACHE}"
else
  echo "[run_libero_train] 未发现 SpikeBrain HF 模块缓存，跳过清除"
fi

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list='qwen_vl_interface'
# base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
base_vlm=${STARVLA_ROOT}/spikebrain/SpikeBrain-2.0-VL
config_yaml=${STARVLA_ROOT}/examples/LIBERO/train_files/starvla_cotrain_libero.yaml
run_root_dir=./results/Checkpoints
data_mix=libero_all
# 输出目录按启动日期命名，同一天多次训练用时分区分
run_date=$(date +%Y%m%d)
run_time=$(date +%H%M)
run_id=${run_date}_${run_time}_${data_mix}_spb2_gr00t_freeze
# === End of environment variable configuration ===
###########################################################################################

export PYTHONNOUSERSITE=1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=online

output_dir=${run_root_dir}/${run_id}
echo "[run_libero_train] run_id=${run_id}"
echo "[run_libero_train] output_dir=${STARVLA_ROOT}/${output_dir}"
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
# GPU 数可变；全局 batch / grad_accum / max_train_steps 由 train_starvla.py 自动推算
# GPU4/5 探测最优（probe_dual45 20260523）：bs40 global160 chunk240，吞吐~9 samp/s，单 epoch~12.5h
NUM_PROCESSES=${NUM_PROCESSES:-2}
PER_DEVICE_BS=${PER_DEVICE_BS:-40}
TARGET_GLOBAL_BS=${TARGET_GLOBAL_BS:-160}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-240}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_port 29502 \
  --num_processes ${NUM_PROCESSES}  \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_forward_chunk_size ${ACTION_CHUNK_SIZE} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.target_global_batch_size ${TARGET_GLOBAL_BS} \
  --trainer.epochs 12 \
  --trainer.save_interval 3000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 200 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jianingliu \
  --trainer.freeze_modules ${freeze_module_list}
  # --is_debug True
  # --trainer.max_train_steps 500000 \



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####