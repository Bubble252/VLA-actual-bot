#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR="${PROJECT}/data/teach_demos"
RUNS_ROOT="${PROJECT}/data/arx_x5_real/runs"
OUTPUT_DIR=""
FPS=20
DATED_RUN=1
EXTRA_ARGS=()

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: $(basename "$0") [input_dir] [output_dir] [--fps N] [--no-dated-run] [convert.py args]

Convert episode_*.hdf5 -> LeRobot v2.1 (StarVLA arx_x5_real).

Default (each run creates a new dated folder):
  input : ${PROJECT}/data/teach_demos
  output: ${RUNS_ROOT}/YYYY-MM-DD_HHMMSS/

Examples:
  bash scripts/run_convert_to_lerobot.sh
  bash scripts/run_convert_to_lerobot.sh data/teach_demos data/arx_x5_real/runs/my_run
  bash scripts/run_convert_to_lerobot.sh --no-dated-run
EOF
  exit 0
fi

if [[ -n "${1:-}" && "${1}" != -* ]]; then
  INPUT_DIR="${1}"
  shift
fi
if [[ -n "${1:-}" && "${1}" != -* ]]; then
  OUTPUT_DIR="${1}"
  DATED_RUN=0
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-dated-run)
      DATED_RUN=0
      shift
      ;;
    --fps)
      FPS="${2:?}"
      shift 2
      ;;
    --fps=*)
      FPS="${1#*=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${DATED_RUN}" -eq 1 ]]; then
    RUN_ID="$(date +%Y-%m-%d_%H%M%S)"
    OUTPUT_DIR="${RUNS_ROOT}/${RUN_ID}"
  else
    OUTPUT_DIR="${PROJECT}/data/arx_x5_real"
  fi
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED="${PIP_TRUSTED:-pypi.tuna.tsinghua.edu.cn}"

python3 -c "import pyarrow, pandas" 2>/dev/null || {
  echo "[convert] installing pyarrow pandas (~60MB, use Tsinghua mirror)..."
  python3 -m pip install --user \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    --timeout 120 \
    pyarrow pandas
}

python3 -c "import imageio.v3" 2>/dev/null || {
  echo "[convert] installing imageio imageio-ffmpeg (bundled ffmpeg)..."
  python3 -m pip install --user \
    -i "${PIP_INDEX}" \
    --trusted-host "${PIP_TRUSTED}" \
    imageio imageio-ffmpeg
}

echo "============================================================"
echo "LeRobot convert"
echo "  input : ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR}"
echo "============================================================"

python3 "${SCRIPT_DIR}/convert_hdf5_to_lerobot.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fps "${FPS}" \
  "${EXTRA_ARGS[@]}"

if [[ "${DATED_RUN}" -eq 1 && -d "${OUTPUT_DIR}" ]]; then
  mkdir -p "${RUNS_ROOT}"
  ln -sfn "${OUTPUT_DIR}" "${RUNS_ROOT}/latest"
  echo ""
  echo "============================================================"
  echo "LeRobot dataset saved:"
  echo "  ${OUTPUT_DIR}"
  echo "  latest -> ${RUNS_ROOT}/latest"
  echo "============================================================"
fi
