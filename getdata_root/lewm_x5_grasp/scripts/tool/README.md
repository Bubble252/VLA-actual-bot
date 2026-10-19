# Play 坐标系 / 图像可视化工具

边跑策略 client 边看：**MuJoCo + URDF 3D**、基座轴、**关节目标角 FK 复现的末端**、双相机侧窗、HDF5 采集鬼臂/参考轴、帧总线。

共享显示参数（如 `eef_xyz_offset_m`）见同级目录：
[`../x5_play_params.yaml`](../x5_play_params.yaml)（改这一处即可同步日志与 MuJoCo SDK 轴）。


## 文档

- [01 · 项目背景](doc/01_项目背景.md)
- [02 · 技术栈与技术点](doc/02_技术栈与技术点.md)
- [03 · 功能实现计划](doc/03_功能实现计划.md)
- [04 · 真机启动命令（含 UI）](doc/04_真机启动命令_含UI.md) ← **日常开跑看这份**

## 目录

```text
scripts/
├── x5_play_params.yaml   ← 共享参数（eef 偏移等）
└── tool/
    ├── run_viz.sh
    ├── requirements.txt
    ├── doc/ 01..04
    ├── assets/
    └── x5_play_viz/
```

## 快速开始

```bash
cd lewm_x5_grasp/scripts/tool
pip3 install -r requirements.txt
bash run_viz.sh --demo                 # MuJoCo + 相机侧窗（demo 灌图）
bash run_viz.sh --demo --headless --steps 60
bash run_viz.sh --bus                  # 等 client --play-viz；无图时 waiting 占位
bash run_viz.sh --ros                  # 仅跟 /arm_status
# HDF5 鬼臂 + 采集参考（P4）
bash run_viz.sh --demo --hdf5 ../../data/pour_foam_from_cup/episode_2026-07-14_201032.hdf5 --ref-frame 10
bash run_viz.sh --bus --hdf5 .../episode.hdf5 --ref-frame 0   # 无总线图时侧窗显示采集帧
```

## 范围摘要

| 项 | 决定 |
|----|------|
| 优先 | 策略 client 边跑边看 |
| 回放 | HDF5 `r/p/v` 只留 `Hdf5ReplaySource` 接口 |
| 末端 | **主**：`q_cmd` → MuJoCo FK；SDK 可选 `--show-sdk` |
| 截图 | 不做 |

实现进度见 `doc/03_功能实现计划.md`。
