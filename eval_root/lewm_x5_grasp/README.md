# LeWM X5 Grasp

这个目录提供一个独立的 X5 机械臂抓取项目骨架，目标是把 `lucas-maes/le-wm` 的世界模型方法接到现有 ARX X5 驱动上。

当前实现分成四层：

1. `scripts/run_grasp.py`
   ROS2 执行入口。订阅 `/arm_status`，发布 `/arm_cmd`，读取图像并执行抓取状态机。
2. `src/lewm_x5_grasp/policy.py`
   策略适配层。默认包含一个可直接联调的 `ScriptedPickPolicy`，并预留 `ImportedPolicyAdapter` 用于接入真实 LeWM 推理模块。
3. `src/lewm_x5_grasp/image_source.py`
   图像输入层。支持文件图像和 Intel RealSense D405，D405 模式可实时显示 RGB/depth 并输出最新 RGB 帧。
4. `scripts/collect_episode.py`
   采集机械臂状态和外部图像路径，生成后续训练 LeWM 所需的数据基础。

## 目录

```text
lewm_x5_grasp/
├── config/
│   ├── grasp_task.d405.json
│   ├── grasp_task.example.json
│   └── grasp_task.wrist_click.json
├── memory.md
├── README.md
├── requirements.txt
├── scripts/
│   ├── collect_episode.py
│   ├── run_d405_camera.py
│   ├── run_grasp.py
│   └── run_wrist_click.sh
└── src/
    └── lewm_x5_grasp/
        ├── __init__.py
        ├── arm_ros2.py
        ├── image_source.py
        ├── policy.py
        ├── schemas.py
        └── state_machine.py
```

## 设计说明

`le-wm` 官方仓库的重点是：

- LeWM 模型结构与训练目标
- 依赖 `stable-worldmodel` 做 planning / evaluation
- 使用像素序列而不是单帧规则控制

但你当前这套 X5 代码库已经有可工作的 ROS2 控制接口，所以这里不直接改底层驱动，而是新建一个外接项目层：

- 底层执行继续走 `/arm_cmd`
- 感知输入支持图像文件，也支持 Intel RealSense D405 直接接入
- LeWM 作为“动作提议器 / 目标位姿生成器”
- 抓取闭环由安全状态机托底，避免模型直接裸控夹爪和末端

这样可以先完成真实机械臂联调，再逐步替换为训练好的 LeWM 推理模块。

## 当前硬件状态

当前已按 D405 主摄像头路径完成接入验证：

- RealSense SDK 能识别 `Intel RealSense D405`
- 已验证 D405 RGB 流输出形状为 `(480, 640, 3)`
- 已验证 D405 depth 流输出形状为 `(480, 640)`
- 项目内 `RealSenseD405ImageSource` 已验证可读取图像并保存 `/tmp/latest_frame.npy`

注意：D405 已进入控制主循环的 observation。当前可用 `click_depth_grasp` 做真机点击抓取：在实时图像中点击目标物体，程序用 depth 和腕载相机外参生成抓取位姿。要达到稳定自动抓取，还需要完成精确手眼外参、工作空间和夹爪姿态标定。

## 运行前提

1. 已按仓库现有流程启动 X5：
   `/home/casia/project/ARX_X5_clean/README_X5_RUN.md`
2. ROS2 控制节点已经运行，并持续发布 `/arm_status`
3. Python 环境包含：
   - `numpy`
   - `rclpy`
4. 如果需要从图片文件输入观察，还需要：
   - `Pillow`
5. 如果使用 Intel RealSense D405 作为主摄像头，还需要：
   - `pyrealsense2`
   - `opencv-python`
   - 系统能看到 `/dev/video*` 或 RealSense SDK 能正常打开 D405

安装 Python 依赖：

```bash
python3 -m pip install --user -r /home/casia/project/ARX_X5_clean/lewm_x5_grasp/requirements.txt
```

## 快速开始

### 1. 启动机械臂底层

按仓库现有说明执行：

```bash
cd /home/casia/project/ARX_X5_clean/ARX_CAN/arx_can
bash arx_can1.sh
```

```bash
source /opt/ros/humble/setup.bash
cd /home/casia/project/ARX_X5_clean/ROS2/X5_ws
source install/setup.bash
ros2 launch arx_x5_controller open_single_arm.launch.py
```

### 2. 运行抓取项目

```bash
source /opt/ros/humble/setup.bash
cd /home/casia/project/ARX_X5_clean/ROS2/X5_ws
source install/setup.bash
export PYTHONPATH=/home/casia/project/ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH
python3 /home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_grasp.py \
  --config /home/casia/project/ARX_X5_clean/lewm_x5_grasp/config/grasp_task.example.json
```

默认策略为 `scripted_pick`，用于验证：

- ROS2 控制链可用
- 状态机逻辑正确
- 夹爪和末端位姿命令能闭环执行

### 2.1 使用 D405 作为主摄像头并实时显示

先单独检查 D405 图像：

```bash
export PYTHONPATH=/home/casia/project/ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH
python3 /home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_d405_camera.py \
  --save-path /tmp/latest_frame.npy
```

确认 RGB 和 depth 窗口正常显示后，再启动机械臂抓取主循环：

```bash
source /opt/ros/humble/setup.bash
cd /home/casia/project/ARX_X5_clean/ROS2/X5_ws
source install/setup.bash
export PYTHONPATH=/home/casia/project/ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH
python3 /home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_grasp.py \
  --config /home/casia/project/ARX_X5_clean/lewm_x5_grasp/config/grasp_task.d405.json
```

`grasp_task.d405.json` 会在主循环中直接打开 D405，实时显示 `D405 color` 和 `D405 depth`，并把 RGB 帧保存到 `/tmp/latest_frame.npy`。外部 `imported` 策略会在 `observation["image"]` 中收到 RGB 图像，在 `observation["depth_image"]` 中收到对齐到彩色图的 depth。

注意：默认 `scripted_pick` 仍然按配置文件里的固定 `pregrasp_xyzrpy/grasp_xyzrpy/lift_xyzrpy` 执行，不会自动根据图像找物体。真实视觉抓取需要把 `policy.type` 改成 `imported`，并在你的策略里使用 `image/depth_image` 计算目标位姿。

### 2.2 D405 数据流

D405 模式下，主循环每个控制周期会生成：

- `observation["image"]`: RGB 图像，`uint8`，形状通常为 `[480, 640, 3]`
- `observation["depth_image"]`: depth 图，`uint16`，形状通常为 `[480, 640]`
- `observation["arm_state"]`: 当前机械臂末端位姿、关节位置、速度和电流

这三类信息会一起传给 `ImportedPolicyAdapter` 中的外部策略。策略输出仍然保持为：

- `target_xyzrpy`: 长度 6 的末端目标位姿
- `gripper`: 夹爪目标
- `phase`: 可选阶段控制

### 2.3 腕载 D405 点击抓取

这个模式用于先让真机完成真实世界抓取闭环。启动后会出现一个实时图像窗口：左侧是 RGB，右侧是 depth 伪彩色图。机械臂在没有点击目标前保持当前位姿；在 `D405 grasp target` 窗口中左键点击 RGB 或 depth 中的目标物体后，程序会：

1. 在点击像素附近取 depth 中值
2. 用 D405 内参反投影成相机坐标点
3. 用 `T_base_ee * T_ee_camera` 转成机械臂基坐标
4. 自动执行 `approach -> descend -> close -> lift`

运行命令：

```bash
/home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_wrist_click.sh
```

这个脚本会自动加载 ROS2 Humble、X5 workspace 和本项目 `PYTHONPATH`。如果要调整最大步数，可以传一个参数：

```bash
/home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/run_wrist_click.sh 900
```

必须先校准 `config/grasp_task.wrist_click.json` 里的腕载外参：

```json
"camera_in_ee_xyzrpy": [x, y, z, roll, pitch, yaw]
```

这里表示从末端执行器坐标系到 D405 彩色光学坐标系的变换，单位是米和弧度。默认值是全 0，只能用于检查软件链路，不能保证真实抓取落点准确。

建议第一次真机测试时：

- 把 `workspace.min_xyz[2]` 设得偏高，避免撞桌
- 把 `grasp_z_offset` 设为正值，例如 `0.02`
- 先点击桌面上方较高、较大的物体
- 手放在急停附近，确认方向正确后再降低抓取高度
- 如果机械臂不动，先看终端日志里的 `phase/current_xyz/target_xyz/dist/gripper`。未点击时 `phase=wait` 是正常现象；点击后应看到 `[click_depth_grasp] target ...` 和 `phase=approach`。

### 3. 采集训练数据

```bash
source /opt/ros/humble/setup.bash
cd /home/casia/project/ARX_X5_clean/ROS2/X5_ws
source install/setup.bash
export PYTHONPATH=/home/casia/project/ARX_X5_clean/lewm_x5_grasp/src:$PYTHONPATH
python3 /home/casia/project/ARX_X5_clean/lewm_x5_grasp/scripts/collect_episode.py \
  --output-dir /tmp/lewm_x5_episode \
  --image-path /tmp/latest_frame.npy
```

## 如何接入真实 LeWM

当前目录没有直接复制 `le-wm` 代码，也没有伪造它的内部 API。原因是官方仓库主要提供训练与评估框架，而真实上线到 X5 机械臂还需要你自己的：

- 相机输入
- 机械臂数据集
- 任务定义
- checkpoint 导出方式
- 推理包装

这里已经预留了接入点：把你的 LeWM 推理包装成一个 Python 类，并实现：

```python
class YourLeWMPolicy:
    def __init__(self, config: dict):
        ...

    def predict(self, observation: dict) -> dict:
        return {
            "target_xyzrpy": [...],   # 长度 6
            "gripper": 0.0,           # 浮点数
            "phase": "approach",      # 可选
        }
```

然后在配置文件中改成：

```json
{
  "policy": {
    "type": "imported",
    "module_path": "/abs/path/to/your_lewm_policy.py",
    "class_name": "YourLeWMPolicy",
    "init": {
      "checkpoint": "/abs/path/to/model.ckpt"
    }
  }
}
```

状态机会负责：

- 限幅
- 末端位姿安全裁剪
- 抓取阶段切换
- lift 完成判断

## 后续建议

1. 先用 `run_d405_camera.py` 确认 D405 画面稳定。
2. 用 `scripted_pick` 验证 X5 控制链、夹爪和状态机。
3. 标定腕载 D405 的 `camera_in_ee_xyzrpy`。
4. 用 `click_depth_grasp` 完成真实物体点击抓取。
5. 把点击抓取数据保存为 episode，再整理成 LeWM 训练需要的时间序列样本。
6. 训练或微调 LeWM checkpoint 后，用 `ImportedPolicyAdapter` 替换点击策略。

## 参考

- `le-wm` GitHub: https://github.com/lucas-maes/le-wm
- `stable-worldmodel` GitHub: https://github.com/galilai-group/stable-worldmodel
