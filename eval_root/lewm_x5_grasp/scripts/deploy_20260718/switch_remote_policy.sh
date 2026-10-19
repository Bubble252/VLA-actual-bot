#!/usr/bin/env bash
# 换权重一键脚本：停掉同端口旧 Policy Server，再按 MODEL/CKPT_STEP 拉起新的。
#
# Usage:
#   MODEL=3p CKPT_STEP=5000 bash switch_remote_policy.sh
#   MODEL=3p CKPT_STEP=mid   bash switch_remote_policy.sh
#   MODEL=1p                 bash switch_remote_policy.sh   # 省略 = 默认最后一步
#
# 不碰本机 CAN / ROS2 / client；换完后请 Ctrl+C 旧 client 再重跑 run_local_client.sh。
#
# ---------- 常用 CKPT_STEP（防遗忘）----------
# MODEL | 含义              | 默认 step        | mid / 中间建议 | 其它
# ------|-------------------|------------------|----------------|------
# 3p    | 关节 20260721-1616| latest（最大步） | 自己指定数字   | —
# 3p_old| 旧 7.17 关节      | 10000            | mid → 5000     | 2500, 7500
# 1p    | EEF libero 续训   | 98000            | mid → 90250    | 85000
# 2p    | EEF scratch       | 7750             | mid → 4000     | 2000, 6000
#
# 现场最常用（新模型）:
#   MODEL=3p bash switch_remote_policy.sh                 # 自动最新 pt
#   MODEL=3p CKPT_STEP=xxxxx bash switch_remote_policy.sh
# ------------------------------------------------
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORCE_RESTART=1
exec bash "${SCRIPT_DIR}/start_remote_policy.sh"
