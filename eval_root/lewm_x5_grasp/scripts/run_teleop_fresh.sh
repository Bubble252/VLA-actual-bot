#!/usr/bin/env bash
# 全新遥操作采集：备份旧 HDF5 → 重启 teleop_record
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEMO_DIR="${PROJECT}/data/teleop_demos"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [extra args for run_teleop_record.sh]

1. 备份 data/teleop_demos 下已有 episode_*.hdf5
2. 停止旧 teleop_record.py
3. 启动全新遥操作采集会话

CAN + ros2 launch 需已在其他终端运行。
EOF
  exit 0
fi

shopt -s nullglob
existing=("${DEMO_DIR}"/episode_*.hdf5)
if ((${#existing[@]} > 0)); then
  stamp="$(date +%Y-%m-%d_%H%M%S)"
  backup="${DEMO_DIR}/backup_${stamp}"
  mkdir -p "${backup}"
  echo "[fresh] 备份 ${#existing[@]} 个 HDF5 -> ${backup}"
  mv "${DEMO_DIR}"/episode_*.hdf5 "${backup}/" 2>/dev/null || true
  mv "${DEMO_DIR}"/.replay_*.parquet "${backup}/" 2>/dev/null || true
else
  echo "[fresh] 无旧 episode，直接启动"
fi

exec bash "${SCRIPT_DIR}/run_teleop_record.sh" "$@"
