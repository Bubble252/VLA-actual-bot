# Memory

## 目标

在 `ARX_X5_clean` 内维护一个独立项目目录，使用 LeWorldModel (`le-wm`) 的方法作为机械臂抓取任务的模型层，并保持和现有 X5 ROS2 驱动解耦。当前已把腕载 Intel RealSense D405 接入为主摄像头输入，并新增了点击深度抓取策略，后续重点转向手眼标定、数据集格式和 LeWM 微调。

## 当前实现逻辑

### 1. 不改底层驱动

原因：

- 现有仓库已经能通过 ROS2 话题 `/arm_cmd` 和 `/arm_status` 控制单臂
- 直接改底层控制器风险高，容易破坏已有联调环境
- LeWM 适合作为上层策略/规划模块，而不是替换电机级驱动

所以当前方案是：

- X5 原始驱动继续保留
- 新项目只做“D405/图像文件感知 -> 模型/策略 -> 抓取状态机 -> ROS2 命令发布”

### 2. 已确认的接口

- `RobotCmd.msg`
  - `end_pos[6]`
  - `joint_pos[6]`
  - `gripper`
  - `mode`
- `RobotStatus.msg`
  - `end_pos[6]`
  - `joint_pos[7]`
  - `joint_vel[7]`
  - `joint_cur[7]`

从 `X5Controller.cpp` 可以确认：

- `mode=4` 走末端位姿控制
- `gripper` 独立发送到 `setCatch`
- `/arm_status` 提供末端位姿和关节状态反馈

### 3. 为什么没有硬塞官方 le-wm API

官方 `le-wm` 仓库核心内容是：

- LeWM 的 JEPA 模型
- 训练配置
- 依赖 `stable-worldmodel` 进行 planning / evaluation

但官方仓库并没有现成的“X5 机械臂抓取在线推理节点”。如果这里直接伪造一个所谓“官方推理 API”，后续反而会和你的真实训练代码不兼容。

因此当前采取的做法是：

- 把 LeWM 接入点抽象成 `ImportedPolicyAdapter`
- 你后面只要把自己的 LeWM checkpoint 封装成一个 Python 类即可替换

### 4. 当前目录中实现的模块职责

- `arm_ros2.py`
  - 订阅 `/arm_status`
  - 发布 `/arm_cmd`
  - 做基础状态缓存
- `image_source.py`
  - 从 `.npy` / `.png` / `.jpg` 文件读取最新图像
  - 直接打开 Intel RealSense D405
  - 读取 RGB 和 depth
  - 可实时显示 `D405 color` / `D405 depth`
  - 可把最新 RGB 帧保存为 `/tmp/latest_frame.npy`
- `policy.py`
  - `ScriptedPickPolicy` 用于硬件联调
- `ClickDepthGraspPolicy` 用于腕载 D405 真机点击抓取
  - 显示一个 RGB+depth 左右拼接窗口
  - 支持在 RGB 或 depth 区域点击目标
  - 未点击前保持 `wait`，避免程序启动后误动
  - `ImportedPolicyAdapter` 用于接入真实 LeWM 推理类
- `state_machine.py`
  - 抓取阶段控制
  - 安全裁剪
  - 位置容差判断
- `run_grasp.py`
  - 主循环
  - 从配置构建图像源
  - 把 RGB/depth/机械臂状态打包为 observation
- `run_d405_camera.py`
  - 独立 D405 预览和最新帧保存脚本
- `run_wrist_click.sh`
  - 一键加载 ROS2 Humble、X5 workspace、本项目 `PYTHONPATH`
  - 运行腕载 D405 点击抓取，避免用户手动 source 时遇到 `No module named rclpy`
- `collect_episode.py`
  - 采集状态与图像路径，为后续训练做数据沉淀

### 5. D405 接入状态

当前已完成：

- 安装并验证 `pyrealsense2`
- RealSense SDK 能识别 `Intel RealSense D405`
- 设备序列号：`323622272722`
- 已验证 RGB 帧形状：`(480, 640, 3)`
- 已验证 depth 帧形状：`(480, 640)`
- 已验证项目内 `RealSenseD405ImageSource` 能读取图像并保存 `/tmp/latest_frame.npy`
- 新增配置：`config/grasp_task.d405.json`
- 新增点击抓取配置：`config/grasp_task.wrist_click.json`

D405 模式下，`ImportedPolicyAdapter` 可从 `observation.to_dict()` 收到：

- `image`: RGB `numpy.ndarray`
- `image_shape`
- `depth_image`: depth `numpy.ndarray`
- `depth_shape`
- `arm_state`
- `phase`
- `step_index`
- `history`

### 6. 现实限制

当前还没有实现以下内容：

- 真正的 LeWM 训练脚本落地到 X5 数据集
- 腕载 D405 到末端执行器的精确手眼外参标定
- 自动识别物体并生成抓取位姿；当前可用鼠标点击目标物体
- 从真实相机 ROS topic 直接取图；当前 D405 是 Python SDK 直连
- 把采集数据转成 `stable-worldmodel` / `le-wm` 的标准数据格式
- 针对 X5 的安全碰撞约束和工作空间标定
- 默认 `scripted_pick` 仍然只按固定配置位姿执行，不使用图像做决策

这些内容需要后续继续补齐。

## 后续迭代建议

### A. 先做相机标定和任务坐标

现在 D405 已经能稳定提供 RGB/depth，点击抓取策略已经具备基本真机闭环。下一步应完成：

- 固定 D405 安装位置
- 标定 D405 相机坐标系到末端执行器坐标系的外参 `camera_in_ee_xyzrpy`
- 明确桌面/工作空间平面
- 把 depth 像素点转换为机械臂可达的 `target_xyzrpy`

### B. 再补视觉策略

当前已经实现了一个非 LeWM 的 `click_depth_grasp` 策略：

- 用户在 `D405 grasp target` 窗口点击目标；窗口左侧为 RGB，右侧为 depth 伪彩色图
- 策略用点击像素附近 depth 反投影为相机坐标
- 再用 `T_base_ee * T_ee_camera` 转成基坐标
- 输出 `target_xyzrpy` 和 `gripper`
- `state_machine.py` 负责 `approach -> descend -> close -> lift`
- 主循环会打印 `phase/current_xyz/target_xyz/dist/gripper`，用于排查机械臂不动问题

后续自动化策略可以沿用同样的输出接口。建议先实现一个简单 `imported` 策略，而不是一开始直接上完整 LeWM：

- 输入 `image` / `depth_image`
- 做目标检测或颜色/区域分割
- 用 depth 估计目标三维点
- 输出 `target_xyzrpy` 和 `gripper`
- 继续让 `state_machine.py` 负责阶段切换和安全裁剪

### C. 再补数据格式

LeWM 需要时间序列数据。建议每个 episode 至少保存：

- 图像序列
- depth 序列
- 机械臂末端位姿
- 关节状态
- 夹爪动作
- 抓取成功标签

### D. 最后接真实模型

接入真实 LeWM 时，建议模型输出不要直接是底层关节量，而是：

- `target_xyzrpy`
- `gripper`
- 可选 `phase`

这样更容易受状态机保护，也更适合和现有 X5 控制接口对接。
