# ARX X5 · ljn pick stage2 90k 本机部署迁移仓

本仓库是 **真机 client 侧** 的可迁移打包（Policy Server 仍跑在 101，**本仓不含权重 / 不含 101 训练代码**）。

来源配方：`arx_x5_ljn_pick_stage2_90000`（师姐 ljn，action 7D，连续夹爪，H=20，state T=2）。

---

## 目录结构

```text
arx_x5_ljn_deploy/
├── README.md                 ← 本文件
├── docs/
│   ├── MIGRATION.md          ← 换机 / 推 GitHub 步骤
│   └── ROTVEC_FIX.md         ← 旋转语义修复说明（为何必须 rotvec）
├── deploy/                   ← 启动脚本（日常从这里跑）
│   ├── env.sh                ← 路径与端口（默认相对本仓根）
│   ├── run_local_client.sh
│   ├── start_remote_policy.sh
│   ├── open_tunnel.sh
│   ├── x5_play_params.yaml
│   ├── README.md             ← 配方细节
│   └── 01_怀疑点排查清单.md
├── eval_root/                ← EVAL_ROOT / ARX_X5_ROOT（含 rotvec 修复）
│   ├── lewm_x5_grasp/        ← client + IK
│   ├── ARX_CAN/
│   └── ROS2/X5_ws/src/       ← 需在新机 colcon build
├── getdata_root/             ← GETDATA_ROOT（RTC + play-viz）
│   ├── RTC/
│   └── lewm_x5_grasp/scripts/tool/
└── starVLA/                  ← STARVLA_ROOT（ModelClient）
```

**关键**：控臂逻辑只认 `eval_root/lewm_x5_grasp/`（已含 `rotvec_delta_to_euler_increment`）。不要用其它树里的旧 `eef_to_joint_ik.py`。

---

## 新机最短路径

### 0. 系统依赖

- Ubuntu + ROS2 Humble  
- RealSense（腕部 D405）+ 桌面相机  
- 能 SSH：`ssh -p 27219 root@115.190.90.101`  
- Python3：`numpy`、`opencv-python`、`pyrealsense2`（及 ROS2 Python）

### 1. 拿到本仓

```bash
# 方式 A：GitHub（推送后）
git clone <你的私有仓 URL> ~/arx_x5_ljn_deploy
cd ~/arx_x5_ljn_deploy

# 方式 B：本机已有目录则直接用
cd /path/to/arx_x5_ljn_deploy
```

路径已由 `deploy/env.sh` 按仓库根自动解析，一般**不用改**。若目录改名，仍以「仓根」为准。

### 2. 编译 ROS2（首次）

```bash
source /opt/ros/humble/setup.bash
cd eval_root/ROS2/X5_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. 启动（101 不动）

```bash
cd deploy

# A. 101 上拉 policy（SSH），端口 5697
FORCE_RESTART=1 bash start_remote_policy.sh
bash open_tunnel.sh

# B. CAN + 单臂（另开终端）
cd ../eval_root
bash ARX_CAN/arx_can/arx_can1.sh
source /opt/ros/humble/setup.bash
source ROS2/X5_ws/install/setup.bash
ros2 launch arx_x5_controller open_single_arm.launch.py

# C. client（先 dry-run）
cd ../deploy
bash run_local_client.sh --dry-run --max-steps 20 --task "pick and place the can"

# D. 真机
RECORD_DIR=./rollouts CONTROL_HZ=20 bash run_local_client.sh \
  --max-steps 800 --task "pick and place the can" \
  --rtc-mode track_b --rtc-execution-horizon 15
```

日志里应出现：`rot_delta_semantics=rotvec`、`ik_fk=eef_link`。改代码后**必须重启 client**。

---

## 配方要点（勿与 0813 混用）

| 项 | 本配方 |
|---|---|
| `unnorm_key` | `franka` |
| policy 端口 | `5697` |
| state 历史 | T=2，**不**归一化 |
| 夹爪 | `continuous_openpi` |
| 旋转语义 | **`rotvec`**（世界系轴角增量 → 再转欧拉给 IK） |
| IK 参考点 | **`eef_link`**（tip，非 link6 法兰） |

详见 `deploy/README.md`、`docs/ROTVEC_FIX.md`。

---

## 本仓不含什么

- 101 上的 Starvla 训练框架 / SpikeBrain / ckpt（仍走 SSH + 隧道）  
- 训练数据、rollouts 录像  
- ROS2 `build/` / `install/`（新机编译）

---

## 自检

```bash
# 确认修复函数在迁移仓内
python3 -c "
import sys
from pathlib import Path
root = Path('.').resolve() / 'eval_root' / 'lewm_x5_grasp' / 'src'
sys.path.insert(0, str(root))
from lewm_x5_grasp.eef_to_joint_ik import rotvec_delta_to_euler_increment
print('OK', rotvec_delta_to_euler_increment)
"
```

更多迁移 / 推 GitHub 步骤见 [docs/MIGRATION.md](docs/MIGRATION.md)。
