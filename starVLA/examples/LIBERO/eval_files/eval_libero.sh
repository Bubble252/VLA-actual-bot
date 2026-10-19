#!/bin/bash
set -euo pipefail

# cd /mnt/petrelfs/yejinhui/Projects/starVLA
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/9950backfile/zhangjh/wh/starvla_wh/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero   # must contain config.yaml with paths on THIS machine
export LIBERO_Python=/9950backfile/zhangjh/.conda/envs/libero/bin/python

# Auto-fix config.yaml if it still points to another machine (e.g. /data2/liguoqi/...)
${LIBERO_Python} -c "
import os, yaml
cfg = os.path.join(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml')
libero_pkg = os.path.join(os.environ['LIBERO_HOME'], 'libero', 'libero')
with open(cfg) as f:
    c = yaml.safe_load(f)
if not os.path.isdir(c.get('init_states', '')):
    from libero.libero import set_libero_default_path
    set_libero_default_path(libero_pkg)
    print('[eval_libero.sh] Regenerated', cfg)
else:
    print('[eval_libero.sh] LIBERO config OK:', cfg)
" 2>&1 || true

export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}" # let eval_libero find the LIBERO tools
export PYTHONPATH="$(pwd):${PYTHONPATH}" # let LIBERO find the websocket tools from main repo

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 # This is important for loading the checkpoint with missing keys (e.g., unnorm_key) without error.

# MuJoCo offscreen rendering (robosuite / LIBERO)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Set to your GPU id for simulation rendering (policy server can use another GPU)
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}

host="${HOST:-127.0.0.1}"
base_port="${BASE_PORT:-5694}"
unnorm_key="${UNNORM_KEY:-franka}"
your_ckpt="${CKPT_PATH:-/9950backfile/zhangjh/wh/starvla_wh/starVLA/results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_29000_pytorch_model.pt}"
export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}
EVAL_LOG_FILE="${LOG_DIR}/eval_libero.log"

task_suite_name="${TASK_SUITE_NAME:-libero_goal}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
steps_num="${STEPS_NUM:-29000}" # 不要忘了改
video_out_path="results/${task_suite_name}_steps_${steps_num}/${folder_name}"

# --- Parallel evaluation (optional) ---
# Start policy server FIRST in another terminal; wait until log shows "server running ...":
#   NUM_SERVERS=1 GPU_IDS=0 bash examples/LIBERO/eval_files/run_policy_server.sh
#   tail -f logs/policy_server_5694.log
num_parallel_workers="${NUM_PARALLEL_WORKERS:-1}"
policy_ports="${POLICY_PORTS:-${base_port}}"
render_gpu_ids="${RENDER_GPU_IDS:-5}"   # e.g. "4,5,6,7" — one physical GPU id per parallel worker
max_tasks="${MAX_TASKS:--1}" # set MAX_TASKS=1/2... for quick smoke debugging

# Example for 4-way parallel (needs 4 GPUs + 4 policy servers):
# num_parallel_workers=4
# policy_ports="${base_port},$((base_port+1)),$((base_port+2)),$((base_port+3))"
# render_gpu_ids="4,5,6,7"

echo "[eval_libero.sh] eval log: ${EVAL_LOG_FILE}"
echo "[eval_libero.sh] policy ports: ${policy_ports}, render gpus: ${render_gpu_ids}"

set +e
${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key "${unnorm_key}" \
    --args.host "$host" \
    --args.port ${base_port} \
    --args.policy-ports "${policy_ports}" \
    --args.num-parallel-workers ${num_parallel_workers} \
    --args.max-tasks "${max_tasks}" \
    --args.render-gpu-ids "${render_gpu_ids}" \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${EVAL_LOG_FILE}"
eval_exit_code=${PIPESTATUS[0]}
set -e

echo "[eval_libero.sh] eval finished with exit code: ${eval_exit_code}"
echo "[eval_libero.sh] eval log saved at: ${EVAL_LOG_FILE}"
exit ${eval_exit_code}
