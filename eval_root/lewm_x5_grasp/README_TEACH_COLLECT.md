# 单臂拖动示教数据采集说明

本文档说明如何在 `ARX_X5_clean` 中，通过**手拖着机械臂**的方式采集抓取示教数据（kinesthetic teaching）。

适用场景：一只手拿着机械臂，拖到目标物体附近，合拢夹爪，完成一次抓取示范，并同步录制腕部相机图像与关节轨迹。

设计参考了师兄师姐提到的 [AC_One](https://github.com/Vertax42/AC_One) 项目，但针对本仓库硬件做了精简：

- **单臂**（非 AC_One 双臂）
- **1 路 Intel RealSense D405** 腕部相机（非三相机）
- **键盘**触发录制（非物理手柄）

---

## 一、改了哪些文件

### 1. ROS2 控制层（示教模式）

| 文件 | 说明 |
|------|------|
| `ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/config/single_arm_teach.yaml` | 单臂 `remote_master` 配置：CAN=`can1`，话题 `/arm_status` + `/arm_cmd`，启动后进入重力补偿，可手拖 |
| `ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/launch/x5_v1/open_single_arm_teach.launch.py` | 示教模式 launch 文件 |

与常规 `open_single_arm.launch.py`（`normal` 模式，电机锁位）不同，示教模式使用 `remote_master`，电机处于重力补偿状态，允许人手拖动。

### 2. 采集脚本（`lewm_x5_grasp`）

| 文件 | 说明 |
|------|------|
| `lewm_x5_grasp/scripts/record_teach.py` | 采集主程序：订阅 `/arm_status`、读取 D405 图像、键盘控制 episode、写入 HDF5 或目录格式 |
| `lewm_x5_grasp/scripts/run_record_teach.sh` | 一键启动脚本（自动加载 ROS2 环境与 `PYTHONPATH`） |
| `lewm_x5_grasp/requirements.txt` | 新增依赖 `h5py`（HDF5 写入） |

### 3. 未改动的复用模块

| 模块 | 作用 |
|------|------|
| `lewm_x5_grasp/src/lewm_x5_grasp/arm_ros2.py` | `X5ArmBridge`，订阅 `/arm_status` |
| `lewm_x5_grasp/src/lewm_x5_grasp/image_source.py` | `RealSenseD405ImageSource`，D405 取帧 |
| `lewm_x5_grasp/src/lewm_x5_grasp/schemas.py` | `ArmState` / `ArmCommand` 数据结构 |

---

## 二、硬件与环境要求

- Ubuntu 22.04 + ROS 2 Humble
- ARX X5 单臂 + USB-CAN（`can1`）
- Intel RealSense D405（腕部安装）
- Python 3，已安装本目录依赖

安装 Python 依赖：

```bash
cd ARX_X5_clean/lewm_x5_grasp
python3 -m pip install --user -r requirements.txt
```

首次使用示教 launch 前，需重新编译 ROS2 工作区（让新 yaml/launch 生效）：

```bash
source /opt/ros/humble/setup.bash
cd ARX_X5_clean/ROS2/X5_ws
colcon build --packages-select arx_x5_controller
source install/setup.bash
```

---

## 三、使用流程

需要**三个终端**，按顺序启动。

### 终端 1：启动 CAN

```bash
cd ARX_X5_clean/ARX_CAN/arx_can
bash arx_can1.sh
```

确认 `can1` 已 UP：

```bash
ip link show can1
```

### 终端 2：启动示教模式控制节点

```bash
source /opt/ros/humble/setup.bash
cd ARX_X5_clean/ROS2/X5_ws
source install/setup.bash
ros2 launch arx_x5_controller open_single_arm_teach.launch.py
```

保持此终端运行。此时机械臂应处于**重力补偿**状态，可以用手拖动。

验证状态话题：

```bash
ros2 topic echo /arm_status
```

拖动机械臂时，`joint_pos` 和 `end_pos` 应实时变化。

### 终端 3：启动采集程序

```bash
cd ARX_X5_clean/lewm_x5_grasp
bash scripts/run_record_teach.sh
```

默认输出目录：`lewm_x5_grasp/data/teach_demos/`。

指定输出目录和任务名：

```bash
bash scripts/run_record_teach.sh /tmp/my_demos --task "pick up the red bottle"
```

---

## 四、采集操作说明

采集程序的终端需要**保持焦点**（键盘输入在该窗口生效）。

| 按键 | 功能 |
|------|------|
| `0` 或 `空格` | 开始录制当前 episode（记录此刻关节角为初始位姿） |
| `s` 或 `Enter` | 手动停止并保存当前 episode |
| `d` | 删除上一个已保存的 episode |
| `q` | 退出采集程序 |

### 推荐示教步骤（以抓瓶子为例）

1. 将机械臂和夹爪摆到**起始姿态**（夹爪张开），**不要**按录制键。
2. 按 `0` 或空格 → 开始录制。
3. 用手拖着机械臂移到瓶子附近。
4. 合拢夹爪（若重力补偿下夹爪可手动捏合，会自动记录 `joint_pos[6]`）。
5. （可选）抬起机械臂。
6. 将臂拖回起始位姿附近并保持约 2 秒 → **自动停止**；或按 `s` 手动停止。
7. 重复 1–6 采集更多 episode；坏的 episode 按 `d` 删除。

### 自动停止逻辑

录制开始 2 秒之后，若关节角与按键时刻的初始位姿相差均 ≤ 0.05 rad，并持续 2 秒，则自动结束当前 episode。逻辑参考 AC_One 的 `collect.py`。

---

## 五、输出数据格式

### 默认：HDF5（`--format hdf5`）

每个 episode 一个文件：

```text
data/teach_demos/
  episode_0.hdf5
  episode_1.hdf5
  ...
```

文件内字段（单臂 7 维）：

| 路径 | 维度 | 含义 |
|------|------|------|
| `observations/qpos` | (T, 7) | 关节角（含夹爪） |
| `observations/eef` | (T, 7) | 末端位姿 6D + 夹爪 |
| `observations/qvel` | (T, 7) | 关节速度 |
| `observations/effort` | (T, 7) | 关节电流 |
| `observations/images/wrist` | (T, H, W, 3) | D405 RGB，`uint8` |
| `action` | (T, 7) | 示教动作（= 当前 `qpos`） |
| `action_eef` | (T, 7) | 末端动作（= 当前 `eef`） |
| `time` | (T,) | 主机时间戳 |
| `arm_stamp` | (T,) | `/arm_status` 时间戳 |
| `attrs["task"]` | 字符串 | 任务描述，如 `"grab the bottle"` |

加 `--use-depth` 可额外保存 `observations/images_depth/wrist`。

### 备选：目录 + jsonl（`--format dir`）

```bash
bash scripts/run_record_teach.sh /tmp/demos --format dir
```

```text
episode_000/
  meta.json
  samples.jsonl
  images/
    frame_000000.png
    ...
```

---

## 六、常用参数

直接调用 `record_teach.py` 可覆盖默认参数：

```bash
source /opt/ros/humble/setup.bash
cd ARX_X5_clean/ROS2/X5_ws && source install/setup.bash
export PYTHONPATH=ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH

python3 ARX_X5_clean/lewm_x5_grasp/scripts/record_teach.py \
  --output-dir /tmp/teach_demos \
  --task "grab the bottle" \
  --hz 20 \
  --max-steps 600 \
  --num-episodes 20 \
  --format hdf5
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--output-dir` | （必填） | 输出目录 |
| `--task` | `grab the bottle` | 任务描述，写入 HDF5 `attrs` |
| `--hz` | `20` | 采样频率（Hz） |
| `--max-steps` | `600` | 单条 episode 最大帧数 |
| `--num-episodes` | `-1` | 采集条数上限，`-1` 不限制 |
| `--format` | `hdf5` | `hdf5` 或 `dir` |
| `--use-depth` | 关闭 | HDF5 中保存深度图 |
| `--no-auto-stop` | 关闭 | 禁用回初始位自动停止 |
| `--enter-gravity` | 关闭 | 若用 `open_single_arm.launch.py`（normal 模式），按录制键时发送 `mode=3` |

---

## 七、与常规控制模式的区别

| 项目 | `open_single_arm.launch.py` | `open_single_arm_teach.launch.py` |
|------|----------------------------|-----------------------------------|
| 控制类型 | `normal` | `remote_master` |
| 电机状态 | 位置锁定，不可拖 | 重力补偿，可手拖 |
| 用途 | 自动执行、画圆 demo、推理 eval | 拖动示教采集 |
| 话题 | `/arm_status`, `/arm_cmd` | 相同 |

**注意**：示教采集时不要同时运行推理脚本（如 `eval_starvla_libero_x5.py`）或画圆 demo，避免多个节点争抢 `/arm_cmd`。

---

## 八、硬件参数调整

若机械臂无法连接或状态异常，编辑：

`ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/config/single_arm_teach.yaml`

```yaml
/arm:
  ros__parameters:
    arm_can_id: can1      # 按实际 CAN 口修改
    arm_control_type: remote_master
    arm_pub_topic_name: arm_status
    arm_sub_topic_name: arm_cmd
    arm_end_type: 0       # 0=标准末端, 1=master, 2=2025新版
```

修改后重新 `colcon build` 并 `source install/setup.bash`。

---

## 九、常见问题

### 9.1 拖不动机械臂

- 确认使用的是 `open_single_arm_teach.launch.py`，不是 `open_single_arm.launch.py`。
- 确认 `can1` 正常、`/arm_status` 有数据。

### 9.2 采集程序报 `Timed out waiting for /arm_status`

- 终端 2 的控制节点未启动或已退出。
- 未 `source install/setup.bash`。

### 9.3 D405 打不开

```bash
export PYTHONPATH=ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH
python3 ARX_X5_clean/lewm_x5_grasp/scripts/run_d405_camera.py
```

先单独确认相机正常，再跑采集。

### 9.4 夹爪捏不动 / `joint_pos[6]` 不变

重力补偿下部分夹爪仍由电机保持力矩。若无法手动捏合，需要后续在采集脚本中增加键盘触发夹爪指令（`--enter-gravity` 场景下通过 `/arm_cmd` 发送 `gripper`）。当前版本假设夹爪可随手动操作变化并被记录。

### 9.5 键盘无响应

采集终端必须处于前台焦点；不要在采集运行时切换到其他窗口输入。

---

## 十、团队目标与 LeRobot 对接

根据当前项目安排，优先级如下：

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P0** | 手摁着采集真机数据 | 本仓库 `record_teach.py` |
| **P0** | 机械臂固定 + 相机安装 | 硬件前置，否则无法稳定采集 |
| **P1** | 转为 **LeRobot** 格式 | 供 StarVLA 真机训练 |
| 暂缓 | 零样本 / sim2real 直接测试 | 真机与仿真结构差异大，需先采真机数据 |

**原则：采集时尽量多存。** 即使某些字段暂时用不到 LeRobot 转换，也应在原始 HDF5 中保留，便于后续排查和扩展。当前脚本已记录：

| 字段 | 来源 | 用途 |
|------|------|------|
| `qpos[7]` | 关节角 + 夹爪 | LeRobot `observation.state` / `action` |
| `eef[7]` | 末端 xyz + rpy + 夹爪 | 位姿分析、后续 delta eef 实验 |
| `qvel[7]` | 关节速度 | 可选训练特征 |
| `effort[7]` | 关节电流 | 夹爪/接触分析 |
| `images/wrist` | D405 RGB | LeRobot `observation.images.cam_wrist` |
| `time` / `arm_stamp` | 时间戳 | 同步与排查 |

其中 `eef[3:6]` 即末端 **roll / pitch / yaw**（弧度），对应聊天记录里提到的「位置、俯仰角等尽量都采」。

### LeRobot 目标格式

StarVLA 真机训练示例在：

`starvla_wh/starVLA/examples/Arx5_ljn/`

| 项目 | 值 |
|------|-----|
| 动作语义 | **abs_qpos**：7D = 6 关节角 (rad) + 夹爪 |
| 状态 | 7D，与 action 同结构 |
| 相机键名 | `observation.images.cam_wrist` |
| 数据注册名 | `arx_x5_real` |
| modality 定义 | `examples/Arx5_ljn/train_files/modality.json` |

LeRobot 目录布局见：`starvla_wh/starVLA/examples/Arx5_ljn/arx5_2lerobot/README.md`

**转换流程（采集完成后）：**

```bash
# 1. 将 HDF5 episode 转为 LeRobot 数据集（转换脚本待补，输入为本仓库 episode_*.hdf5）
#    目标目录示例：playground/Datasets/arx_x5_real/

# 2. 写入 modality.json
python starvla_wh/starVLA/examples/Arx5_ljn/train_files/prepare_arx5_dataset.py \
  --data-root playground/Datasets/arx_x5_real

# 3. 验证 dataloader 能读
python starvla_wh/starVLA/starVLA/dataloader/lerobot_datasets.py \
  --config_yaml starvla_wh/starVLA/examples/Arx5_ljn/train_files/starvla_train_arx5_real.yaml

# 4. 训练
bash starvla_wh/starVLA/examples/Arx5_ljn/train_files/run_arx5_train.sh
```

> **说明**：`arx5_2lerobot` 目前只有格式说明，**HDF5 → LeRobot 的转换脚本尚未实现**。采完第一批 `episode_0.hdf5` 后，按样例补 `convert_hdf5_to_lerobot.py` 即可。转换是否成功，是数据有效性的第一道检验。

### 双相机

聊天记录提到需要 **2 个摄像头**（例如腕部 + 第三视角）。当前脚本仅支持 **1 路 D405（wrist）**。硬件装好后需扩展：

1. `image_source.py`：支持多路 D405 / RealSense
2. `record_teach.py`：每帧写入 `images/wrist` + `images/cam_fixed`（或 LeRobot 对应键名）
3. `modality.json`：增加第二路 video 字段

在第二路相机就绪前，可先用单腕相机完成采集与 LeRobot 流程验证。

---

## 十一、数据有效性检查

按团队讨论，建议按以下顺序验收：

### 11.1 格式检查（自动）

- [ ] HDF5 能正常打开，帧数 > 0
- [ ] `qpos` / `eef` / `images/wrist` shape 一致
- [ ] 能成功转为 LeRobot 并被 `lerobot_datasets.py` 加载

### 11.2 逐条人工检查（必做）

每条 episode 至少看一遍：

- [ ] **图像**：方向正确、无严重模糊/过曝、能看到任务相关物体
- [ ] **关节角范围**：`qpos[0:6]` 是否在合理弧度范围，无突变尖峰
- [ ] **夹爪**：`qpos[6]` 在「张开 → 闭合」过程中有变化
- [ ] **末端轨迹**：`eef[0:3]` 位移方向符合任务（例如确实靠近瓶子）
- [ ] **俯仰/姿态**：`eef[3:6]`（rpy）变化是否合理

### 11.3 统计检查（脚本可辅助）

```bash
# 示例：用 Python 快速看一条 episode 的数值范围
python3 - <<'PY'
import h5py, numpy as np, sys
path = sys.argv[1] if len(sys.argv) > 1 else "lewm_x5_grasp/data/teach_demos/episode_0.hdf5"
with h5py.File(path) as f:
    q = f["observations/qpos"][()]
    e = f["observations/eef"][()]
    print("frames:", len(q))
    print("qpos min/max:", q.min(0).round(3), q.max(0).round(3))
    print("eef  xyz range:", e[:, :3].min(0).round(3), e[:, :3].max(0).round(3))
    print("eef  rpy range:", e[:, 3:6].min(0).round(3), e[:, 3:6].max(0).round(3))
    print("gripper range:", q[:, 6].min().round(3), q[:, 6].max().round(3))
PY
```

### 11.4 归一化检查（转换 LeRobot 后）

- [ ] 训练 config 中 `unnorm_key=arx_x5_real` 与数据集匹配
- [ ] action / state 分布无 NaN、无全零 episode
- [ ] 可问森威（仿真 arxx5 环境经验）核对 action 语义是否为 **绝对关节角**

---

## 十二、后续步骤

1. **今天硬件**：固定机械臂、装好相机（目标 2 路，至少先 1 路 D405 可用）。
2. **采集**：按本文第三节跑通，采 5–10 条试数据。
3. **验收**：按第十一节逐条检查。
4. **转换**：补 HDF5 → LeRobot 脚本，跑通 `prepare_arx5_dataset.py` + dataloader 验证。
5. **训练**：`run_arx5_train.sh` 真机微调（非零样本 sim2real）。
6. **扩展**：第二路相机、更多任务与物体位置多样性。

---

## 十三、相关文档

- 机械臂基础运行：`ARX_X5_clean/README_X5_RUN.md`
- 抓取与 D405 联调：`lewm_x5_grasp/README.md`
- StarVLA 真机训练（LeRobot）：`starvla_wh/starVLA/examples/Arx5_ljn/README.md`
- LeRobot 格式说明：`starvla_wh/starVLA/examples/Arx5_ljn/arx5_2lerobot/README.md`
- StarVLA LIBERO eval（仿真权重，非当前优先）：`lewm_x5_grasp/scripts/eval_starvla_libero_x5.py`
