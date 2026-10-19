#!/usr/bin/env bash
# One-time Python deps for lewm_x5_grasp (record + convert).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED="${PIP_TRUSTED:-pypi.tuna.tsinghua.edu.cn}"

echo "Installing lewm_x5_grasp deps from ${PIP_INDEX} ..."

python3 -m pip install --user \
  -i "${PIP_INDEX}" \
  --trusted-host "${PIP_TRUSTED}" \
  --timeout 120 \
  -r "${PROJECT}/requirements.txt"

python3 - <<'PY'
import importlib
for m in ("numpy", "h5py", "pyrealsense2", "pyarrow", "pandas"):
    importlib.import_module(m)
import numpy
import cv2
if int(cv2.__version__.split(".")[0]) < 4:
    raise SystemExit("cv2 too old; need pip opencv-python-headless>=4.10")
print(f"numpy {numpy.__version__}, cv2 {cv2.__version__}")
print("All deps OK")
PY
