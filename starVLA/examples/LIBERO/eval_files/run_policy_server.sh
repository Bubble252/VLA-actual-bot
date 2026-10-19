#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python="${star_vla_python:-/9950backfile/zhangjh/.conda/envs/starvla/bin/python}"

NUM_SERVERS=1
GPU_IDS="${GPU_IDS:-5}"

# MACA runtime
# export MACA_HOME=/opt/maca-3.3.0
# export MACA_PATH=/opt/maca-3.3.0
# export LD_LIBRARY_PATH=/opt/maca-3.3.0/mxgpu_llvm/lib:$LD_LIBRARY_PATH
# export PATH=/opt/maca-3.3.0/bin:/opt/maca-3.3.0/tools/cu-bridge/bin:$PATH
# export PYTHONNOUSERSITE=1

your_ckpt=${CKPT_PATH:-/9950backfile/zhangjh/wh/starvla_wh/starVLA/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_29000_pytorch_model.pt}
base_port=${BASE_PORT:-5694}

if command -v ss >/dev/null 2>&1 && ss -tln | grep -q ":${base_port} "; then
  echo "ERROR: port ${base_port} 已被占用（可能已有 policy server 在跑）"
  echo "  查看: ss -tln | grep ${base_port}"
  echo "  结束旧进程: pkill -f 'server_policy.py.*--port ${base_port}'"
  echo "  不要重复启动，否则新进程会绑定失败并导致脚本退出（显示 Stopping policy server）"
  exit 1
fi

# NUM_SERVERS=1 for single-GPU eval; NUM_SERVERS=4 GPU_IDS="0,1,2,3" for parallel eval
NUM_SERVERS=${NUM_SERVERS:-1}
GPU_IDS=${GPU_IDS:-0}

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
server_pids=()

cleanup() {
  echo "Stopping policy server(s)..."
  for pid in "${server_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((base_port + i))
  gpu_id=${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}
  log_file="logs/policy_server_${port}.log"
  mkdir -p logs

  echo "Starting policy server on GPU ${gpu_id}, port ${port} (log: ${log_file})"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --unnorm_key ${UNNORM_KEY:-franka} \
    --use_bf16 \
    --idle_timeout -1 \
    > "${log_file}" 2>&1 &
  server_pids+=($!)
done

echo "Launched ${NUM_SERVERS} policy server(s) on port(s) ${base_port}-$((base_port + NUM_SERVERS - 1))"
echo "PIDs: ${server_pids[*]}"
echo "Waiting for server(s) to load model and listen (this can take several minutes per server)..."
echo "  tail -f logs/policy_server_${base_port}.log   # watch load progress"

ready=0
for ((i=0; i<NUM_SERVERS; i++)); do
  port=$((base_port + i))
  for ((t=0; t<600; t++)); do
    if grep -q "server running" "logs/policy_server_${port}.log" 2>/dev/null \
       || (command -v ss >/dev/null && ss -tln | grep -q ":${port} "); then
      echo "✅ Port ${port} is ready"
      ready=$((ready + 1))
      break
    fi
    sleep 2
  done
  if ! grep -q "server running" "logs/policy_server_${port}.log" 2>/dev/null \
     && ! (command -v ss >/dev/null && ss -tln | grep -q ":${port} "); then
    echo "⚠️  Port ${port} not ready after 20 min — check logs/policy_server_${port}.log"
  fi
done

echo "Ready: ${ready}/${NUM_SERVERS} server(s). Press Ctrl+C to stop all servers."
wait
