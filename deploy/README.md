# ARX X5 LJN Pick Stage2（90k）真机部署 — 本机 + 101

> **迁移仓入口**：仓库根目录 [../README.md](../README.md)（换机 / GitHub 先看那里）  
> 旋转修复说明：[../docs/ROTVEC_FIX.md](../docs/ROTVEC_FIX.md) · 疑点清单：[01_怀疑点排查清单.md](01_怀疑点排查清单.md)
>
> 适用产物：`115.190.90.101` 上  
> `/vepfs-mlp2/c20250405/400040/transfer/arx_x5_ljn_pick_stage2_90000_transfer/arx_x5_ljn_pick_stage2_90000_transfer/`  
> 权重：`checkpoints/steps_90000/pytorch_model.pt`（另有 50k / 60k）  
> 来源：师姐（ljn）在岩山服务器训练，已迁移至 101  
> 配方：**SpikeBrain-2.0-VL + DiT**，ARX 实机，**双阶段**（stage1 冻 VLM 10k → stage2 全量 90k），**仅 pick 任务**  
> 本机 client 根：`../eval_root`（`EVAL_ROOT`，含 rotvec 修复）；**不是**旧 server46 `src/` 副本

---

## 0. 这次部署 vs scratch0813 的核心区别

| 项 | scratch0813 | **本次 ljn_pick_stage2_90k** | 部署侧影响 |
|---|---|---|---|
| action_dim | 8（含 pad/+g/−g） | **7**（x,y,z,roll,pitch,yaw,gripper） | client action 解析改 7 维 |
| **gripper** | binary {0,1}，阈值化 | **连续 openpi_closedness** | **不能二值化**，需映射回 qpos |
| **action_horizon H** | 8 | **20** | chunk 更长，RTC 参数重调 |
| 归一化 | min_max | min_max | 一样 |
| frame_align | none | none | 一样 |
| pos/rot scale | 1.0 / 1.0 | 1.0 / 1.0 | 一样 |
| state_dim | 8 | 8 | 一样 |
| **state history** | T=16 | **T=2**（state_indices=[-1,0]） | client 送 2 帧，不是 16 |
| 图像增广 | 无 | 训练时 0.5 概率 crop/旋转/color（部署关） | 部署无影响（eval 模式） |
| vlm_packing | true | false | 部署无影响 |
| 训练阶段 | 单阶段 | 双阶段 | 部署无影响 |
| 任务范围 | pick + stack | **仅 pick** | 只能跑 pick |
| robot_type | `arx_x5_to_franka` | **`arx_x5_real_ljn`**（101 已注册 ✅） | dataloader 自动解析 |
| **unnorm_key** | `new_embodiment` | **`franka`** | server 启动必须传 `franka` |
| data_mix | `arx_x5_finetune_base20_h16_8d` | `arx_x5_pick_delta_cmdgrip_aug` | **101 未注册 ❌，需补** |
| 推理步 | FM 4 步 | FM 4 步 | 一样 |
| Policy 端口 | 5696 | **5697**（避开 0813） | 隧道端口改 5697 |

**一句话**：坐标系 / scale / 归一化和 0813 完全一样（none / 1.0 / min_max），**差异集中在夹爪连续化 + H=20 + state 历史 T=2 + unnorm_key=franka**。

---

## 1. 夹爪映射（最关键改动）

0813 是 binary：`grip_raw >= 0.5 → 开 else 关`。

本次是**连续 openpi_closedness**：
- 模型输出 action 第 7 维 ∈ [0, 1]，1 = 完全闭合，0 = 完全张开（OpenPI 约定）
- 映射回 ARX 原生 qpos（来自 action_contract.json）：
  - `qpos_open = 4.491874016821384`
  - `qpos_closed = -0.06599569314857945`
  - `qpos = qpos_closed + closedness * (qpos_open - qpos_closed)`
- 下发给真机的是这个连续 qpos，**不再阈值 0/1**

```python
def map_gripper_continuous(closedness: float) -> float:
    """closedness ∈ [0,1]: 0=全开, 1=全闭（OpenPI 约定）→ ARX qpos"""
    qpos_open = 4.491874016821384
    qpos_closed = -0.06599569314857945
    c = float(np.clip(closedness, 0.0, 1.0))
    return qpos_closed + c * (qpos_open - qpos_closed)
```

> ⚠️ 注意 OpenPI closedness 方向：1=闭、0=开。若示教采集时开≈4.5、关≈0.0，则 closedness=0 → qpos=4.49（开），closedness=1 → qpos=−0.066（关）。方向自洽。

---

## 2. 部署前置（必须先解决）

### 2.1 补注册 data_mix `arx_x5_pick_delta_cmdgrip_aug`（101）

`PolicyNormProcessor` 从 ckpt config 读 `data_mix=arx_x5_pick_delta_cmdgrip_aug`，再反查 `DATASET_NAMED_MIXTURES` 得 robot_type。101 上该 data_mix **未注册**，启动会 `KeyError`。

需在 101 的 `starVLA/dataloader/gr00t_lerobot/mixtures.py` 加条目，指向 `arx_x5_real_ljn`。具体 dataset 路径 / 权重需师姐提供，或按 `pick_delta_cmdgrip` 单条构造。

### 2.2 SpikeBrain 软链

复用 0813 的 `ensure_vlm_path_on_101.sh`（VLM 路径相同）。

---

## 3. 部署流程（6 步，与 0813 同构，参数不同）

```bash
cd ~/project/ARX_X5_clean_server46\(7.12get-data\)/lewm_x5_grasp/scripts/deploy_arxx5_ljn_pick_stage2_90k

# 步骤 1：101 启动 Policy Server（端口 5697，unnorm_key=franka，gpu=1）
FORCE_RESTART=1 CKPT_STEP=90000 bash start_remote_policy.sh
# 就绪标志：日志出现 "server running"，ss -tln | grep 5697

# 步骤 2：本机开隧道（5697）
bash open_tunnel.sh
# 等价: ssh -p 27219 -N -f -L 5697:127.0.0.1:5697 root@115.190.90.101

# 步骤 3：CAN（另开终端）
cd ~/project/ARX_X5_clean\(7.12eval\)
bash ARX_CAN/arx_can/arx_can1.sh

# 步骤 4：ROS2 单臂
cd ~/project/ARX_X5_clean\(7.12eval\)
source /opt/ros/humble/setup.bash
cd ROS2/X5_ws && source install/setup.bash
ros2 launch arx_x5_controller open_single_arm.launch.py
# 确认: ros2 topic hz /arm_status 约 100Hz

# 步骤 5（可选）：Play UI
cd ~/project/ARX_X5_clean_server46\(7.12get-data\)/lewm_x5_grasp/scripts/tool
bash run_viz.sh --bus

# 步骤 6：Client（先 dry-run 再真机）
cd ~/project/ARX_X5_clean_server46\(7.12get-data\)/lewm_x5_grasp/scripts/deploy_arxx5_ljn_pick_stage2_90k

# 6a. dry-run
bash run_local_client.sh --dry-run --max-steps 20 --task "pick and place the can"

# 6b. 真机（H=20，RTC 参数待调）
CONTROL_HZ=10 bash run_local_client.sh --max-steps 300 \
  --task "pick and place the can" \
  --rtc-mode track_b
# 注意：连续夹爪，不再传 --gripper-open/--gripper-close 二值参数
```

---

## 4. RTC 参数待调（H=20 vs 0813 的 H=8）

0813 用 `CONTROL_HZ=5` + `track_b`。H=20 chunk 长 2.5 倍，可适当提高 `CONTROL_HZ`（如 10），或加大 replan 间隔。**需真机实测调参**，先从低 Hz 起步。

---

## 5. 离线测评参考（已跑）

| 空间 | MSE | MAE | max |pred−gt| |
|---|---|---|---|
| 归一化 [-1,1] | 0.00304 | 0.0292 | 2.036（gripper） |
| 物理（min_max 反归一化） | 7.23e-5 | 0.00196 | 0.980（gripper） |

物理各维 max：Δx≈1.4cm, Δy≈1.0cm, Δz≈3.1cm, Δroll≈0.10rad, Δpitch≈0.06rad, Δyaw≈0.07rad, gripper≈0.98。
质量不错，可上真机。

---

## 6. 本目录文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `README.md` | 本文 | ✅ |
| `env.sh` | 路径/端口/unnorm/scale/H/T/gripper 常量 | ✅ |
| `start_remote_policy.sh` | 本机 SSH 拉起 101 Policy Server（5697，含 `STARVLA_DEPLOY=1`） | ✅ |
| `open_tunnel.sh` | 本机端口隧道（5697） | ✅ |
| `ensure_vlm_path_on_101.sh` | 101 上 SpikeBrain 软链（复制自 0813，VLM 路径相同） | ✅ |
| `x5_play_params.yaml` | frame_align=none, scale=1.0, remap=off（与 0813 同，坐标系/scale 一致） | ✅ |
| `gripper_map.py` | 连续夹爪 closedness↔qpos 映射（自检通过，方向已修正） | ✅ |
| `run_local_client.sh` | 按本配方调 7.12eval client（T=2 + no-normalize + openpi state + continuous gripper） | ✅ |

---

## 7. 进度记录（A 类：101 服务器侧）

| 步骤 | 状态 | 说明 |
|---|---|---|
| A0 层级修复 | ✅ 完成 | ckpt 迁移包 config.yaml/dataset_statistics.json 在 `metadata/`，但 `read_mode_config` 用 `ckpt.parents[1]`（=checkpoints/）找。已在 `checkpoints/` 下软链到 `../metadata/` |
| A1 data_mix 注册 | ✅ 完成（101） | `arx_x5_pick_delta_cmdgrip_aug = [("pick_delta_cmdgrip", 1.0, "arx_x5_real_ljn")]` 已写入 101 `mixtures.py`，验证可查 |
| A1 本地同步 | ✅ 完成 | 本地 `migrate_bundle/.../mixtures.py` 已加同条目（注释标注 arx_x5_real_ljn 待同步 data_config） |
| A2 SpikeBrain 软链 | ✅ 完成 | 复用 0813 `ensure_vlm_path_on_101.sh`，软链 + HF cache 就绪 |
| A3 PolicyServer 启动验证 | ✅ 完成 | 方案 2：`ArxX5RealLjnDataConfig.transform()` 加 `STARVLA_DEPLOY=1` 开关，部署时复用 Q99 干净 transform，绕过 video metadata。验证 PolicyNormProcessor 构造成功 |

## 8. 阻塞问题：PolicyNormProcessor video metadata（已解决 ✅）

### 根因
- `ArxX5RealLjnDataConfig` 继承 `ArxX5ToFrankaAugDataConfig`，其 `transform()` 含 video aug（`VideoResizeWithPad`/`VideoColorJitter` 等，来自 `_openpi_style_real_robot_video_aug`）
- `PolicyNormProcessor._build_dataset_metadata` 只构造 state/action metadata，**video metadata 返回空 `{}`**
- `self._transform.set_metadata(ds_meta)` 时，video transform 找 `primary_image` key 找不到 → `ValueError`
- **0813 无此问题**：`Libero4in1DataConfig.transform()` 的 video transforms 全部注释掉，不触发 video metadata 检查

### 本质
训练-部署代码路径不兼容：训练时 dataloader 造完整 metadata（含 video），部署时 `PolicyNormProcessor` 只造 state/action，video 给空 → 崩。**不是模型问题，不影响最终效果**（部署时 `PolicyNormProcessor` 只用 `apply_actions`/`unapply_actions` 处理 action，从不调用 video transform）。

### 采用方案：方案 2（deploy 专用 transform 开关）

在 101 `data_config.py` 的 `ArxX5RealLjnDataConfig` 加 `transform()` 重写，用环境变量 `STARVLA_DEPLOY=1` 切换：
- **训练**（不设 env）：用 `ArxX5ToFrankaAugDataConfig.transform()`（含 video aug + state noise/dropout）—— 不影响训练
- **部署**（设 `STARVLA_DEPLOY=1`）：复用 `ArxX5ToFrankaQ99DataConfig.transform()`（干净版，只 state gripper→openpi + action min_max，无 video aug）—— 让 set_metadata 通过

备份：`data_config.py.bak_20261010_044048`

### 验证结果
设 `STARVLA_DEPLOY=1` 构造 `PolicyNormProcessor(ckpt, unnorm_key="franka")`：
- ✅ 构造成功，不再报 `Video key primary_image not found`
- ✅ `robot_type=arx_x5_real_ljn`，`unnorm_key=franka`
- ✅ action_keys 7 维（x,y,z,roll,pitch,yaw,gripper），state_keys 8 维（含 pad/gripper）

### ⚠️ 注意事项（重要）

1. **部署必须设 `STARVLA_DEPLOY=1`**，否则 server 启动会崩（video metadata）。`start_remote_policy.sh` 里要 `export STARVLA_DEPLOY=1`。
2. **训练绝不能设 `STARVLA_DEPLOY=1`**，否则丢 video aug + state noise/dropout，影响训练效果。训练脚本不要带这个 env。
3. **此改动只影响 101 `data_config.py`**，是框架代码改动（非可移植配置）。本地 `migrate_bundle` 副本无 `ArxX5RealLjnDataConfig`，待同步。
4. **state 不归一化（关键发现，见 §10）**：训练 transform 链里 `StateActionTransform` 只 `apply_to=action_keys`，**state 从头到尾没有 min_max 步骤**。模型实际看到的 state = `_arx_x5_state_gripper_to_openpi()` 转换后的原始值：`[xyz, rotvec, pad=0, gripper=closedness∈0~1]`，不归一化直接喂模型。**client 侧不需要 `--normalize-state` / `--state-stats-json`**，只需复现转换：`pad→0`、`gripper = clip((4.49 - qpos)/4.56, 0, 1)`，然后 history T=2（`[t-1, t]` 两帧）直接送。`dataset_statistics.json` 的 state 部分（±0.036，= 原始×0.008）是旧流程残留，**训练和部署都不用它**。
5. **部署时 video transform 不被调用**：client 送的图像直接喂模型，图像预处理（224 + resize_with_pad 填充）由 client 侧负责，要和训练对齐（见 C7 client 改造）。

### 师姐 real_eval 改进版（未采用，备查）
迁移包 `real_eval/policy_norm_processor.py` 是改进版（深拷贝 data_config + 从 ckpt config 注入 normalization_mode + min_max/q99 clip + 有 normalize_state），但 101 deployment 版不用它也能跑（0813 先例）。如后续发现 state/action 归一化不一致，可考虑用 real_eval 改进版覆盖 101 deployment 版（方案 3）。

## 9. 待办

- [x] A3 解决 video metadata 阻塞（方案 2，已完成）
- [x] **C-state 查清：state 不归一化**（见 §10），client 不需 state stats，只需 pad→0 / grip→closedness 转换
- [ ] 本地 `migrate_bundle/.../data_config.py` 同步 `ArxX5RealLjnDataConfig` + deploy 开关（当前本地无此类）
- [x] B4-B6 生成部署脚本（start_remote_policy 含 `STARVLA_DEPLOY=1` / open_tunnel / x5_play_params / gripper_map）
- [x] **C7-C9 client 改造完成**（见 §11）：`eval_starvla_libero_x5.py` 加 4 参数 + 3 处 gripper 下发分支 + state8 openpi 模式；`run_local_client.sh` 生成
- [ ] 真机 dry-run → 低速真机 → 调 RTC

---

## 10. 关键发现：state 不归一化（2026-10-10 查清）

### 现象
- parquet 落盘 state：`pad ∈ [-0.026, 4.49]`（+g 原始 qpos），`grip ∈ [-4.49, 0.026]`（-g 原始）
- `dataset_statistics.json` state：`pad ∈ [-0.0002, 0.036]`，`grip ∈ [-0.036, 0.0002]` —— **= 原始 ×0.008**（0.036 = 4.39×0.008，用户猜的系数 0.008 正确）
- 两者差 0.008 倍，看似矛盾

### 真相：state 根本不归一化
查 `data_config.py` 的 transform 链（训练用 `ArxX5ToFrankaAugDataConfig`，部署用 `ArxX5ToFrankaQ99DataConfig`）：

```python
transforms = [
    *_arx_x5_state_gripper_to_openpi(),   # state.pad→0, state.gripper→closedness∈[0,1]
    StateActionToTensor(apply_to=action_keys),
    StateActionTransform(
        apply_to=action_keys,              # ← 只对 action 归一化！state 不归一化
        normalization_modes=dict(_ARX_X5_EEF_ACTION_NORM_MODES),
    ),
]
```

`StateActionTransform` 只 `apply_to=action_keys`，**state 从头到尾没有 min_max 步骤**。模型训练时实际看到的 state = 转换后原始值 `[xyz, rotvec, pad=0, gripper=closedness∈0~1]`，不归一化直接喂。

`dataset_statistics.json` 的 state 部分（±0.036 = 原始×0.008）是**旧流程残留**（早期版本 state 可能也归一化，后来改成不归一化但 stats 没删），训练和部署都不用它。

### 验证（从 parquet 重算）
| | pad (idx6) | grip (idx7) |
|---|---|---|
| parquet 原始 | [-0.026, 4.49] | [-4.49, 0.026] |
| stats 文件（×0.008） | [-0.0002, 0.036] | [-0.036, 0.0002] |
| **transform 后（模型输入）** | **0**（清零） | **[0.0003, 0.991]**（closedness） |

### 对部署的影响（简化 C8）
- client **不需要** `--normalize-state` / `--state-stats-json`
- client 只需复现 `_arx_x5_state_gripper_to_openpi()`：`pad→0`、`gripper = clip((4.49 - qpos)/4.56, 0, 1)`
- state history T=2：`[t-1, t]` 两帧，每帧 8D = `[xyz(3), rotvec(3), pad=0, gripper=closedness]`
- 直接喂模型，不做任何 min_max

---

## 11. Client 改造完成（C7-C9，2026-10-10）

### 改的文件
- `~/project/ARX_X5_clean(7.12eval)/lewm_x5_grasp/scripts/eval_starvla_libero_x5.py`（0813 共用 client，加分支不影响 0813）
- `run_local_client.sh`（本目录新生成）

### 新增 4 个参数（argparse 已确认注册）
| 参数 | 默认值 | 作用 |
|---|---|---|
| `--gripper-mode` | `binary`（0813 不变） | `continuous_openpi` 时连续映射，否则二值阈值 |
| `--gripper-qpos-open` | `4.491874016821384` | continuous 模式：closedness=0 → qpos（开） |
| `--gripper-qpos-closed` | `-0.06599569314857945` | continuous 模式：closedness=1 → qpos（闭） |
| `--state-gripper-mode` | `sign`（0813 不变） | `openpi` 时 state[6:8]=[pad=0, grip=closedness]，否则 [+g,-g] |

### 新增辅助函数（line ~395）
```python
def _closedness_to_qpos(c, qo, qc):  # 0=open→qo, 1=closed→qc
    return qpos_open - clip(c,0,1) * (qpos_open - qpos_closed)
def _qpos_to_closedness(q, qo, qc):   # 逆，state 侧用
    return clip((qpos_open - qpos) / (qpos_open - qpos_closed), 0, 1)
```
往返一致性验证通过（diff 0），端点 c=0→4.49开、c=1→-0.066闭、c=0.37→2.80（与训练数据吻合）。

### 改的 3 处 gripper 下发（加 `continuous_openpi` 分支，不影响 binary）
- `joint` 分支（line ~2113）：`gripper = _closedness_to_qpos(grip_raw,...)` or `float(grip_raw)`
- `eef_via_joint` 分支（line ~2148）：`_closedness_to_qpos` or `_binarize+_map_gripper`
- `eef` 分支（line ~2322）：同上

### 改的 state 组装（`_sdk_eef_state8`，line ~860）
加 `state_gripper_mode` 参数：`openpi` 时 `out[6]=0, out[7]=_qpos_to_closedness(g,...)`，否则原 `[+g,-g]`。调用处（line ~1910）透传 `args.state_gripper_mode`。

### run_local_client.sh 关键参数（与 0813 差异）
```bash
--state-history-len 2          # 0813: 16
--no-normalize-state            # 0813: --normalize-state（state 不归一化）
--state-gripper-mode openpi     # 0813: sign
--gripper-mode continuous_openpi # 0813: binary
--gripper-qpos-open 4.491874    # 显式传（默认已对）
--gripper-qpos-closed -0.065996
# 不传 --state-stats-json（state 不归一化）
# 不传 --gripper-open/close（continuous 模式不用）
```

### ⚠️ 发现并修正的方向 bug
初版 `_closedness_to_qpos` 用了 `qc + c*(qo-qc)`（c=0→闭，方向反），测试往返不一致发现。修正为 `qo - c*(qo-qc)`（c=0→开，c=1→闭）。`gripper_map.py` 代码本身是对的，仅修了注释。**OpenPI 约定：0=open, 1=closed**。

---

## 12. 重大发现：rot_delta 旋转语义错位（2026-10-10 真机"乱飘"根因）

> 📋 疑点专项记录（状态速览、两份代码分叉警告、复验步骤、部署 checklist）→ [01_怀疑点排查清单.md](01_怀疑点排查清单.md)

### 现象
真机多次部署：机械臂先追罐子（x 移动正确），随后末端姿态越漂越远（rotvec[1] 漂到 −1.15 rad，J5 被拉到 −66°~−126°），模型输出退化成 ~1mm/步小 delta，永不闭合夹爪。录制轨迹对账发现：**模型输出完全正常**（rot_delta[0]=−5.8mrad/步与 demo 前 100 步统计一致；下降阶段的 dz 命令正确；x 追踪相关 0.46），**但执行层把动作做错了**。

### 根因
- 训练 action = state rotvec 的**逐元素差分**（delta 数据集）→ 模型 rot_delta 语义 = **世界系旋转向量增量**
- client（0813 遗产）把 rot_delta **当 extrinsic-XYZ 欧拉增量**逐元素加到 rpy 上
- 在 yaw≈90° 姿态下，欧拉 roll 增量的物理效果 = 绕世界 **y** 轴旋转（Rz(90°)·Rx(δ)·Rz(−90°)=Ry(δ)）
- → 模型的 roll 命令被执行成 pitch：实测 rotvec[1] 漂移 −5.81 mrad/步 vs 模型 rot_delta[0]=−5.83 mrad/步，数值完美吻合
- 姿态漂移 → state OOD → 模型输出退化 → 恶性循环 = "乱飘"
- 次要 bug：diff-IK Jacobian 用 `full_fk`（link6 法兰）但命令的是 tip（eef_link）增量，0.145m 杠杆臂造成 ~1mm/步级耦合误差（z"命令下降实际上升"的来源）

### 修复
- `eef_to_joint_ik.py` 新增 `rotvec_delta_to_euler_increment()`：R_new = Rv(δ)@R_cur → 欧拉增量（单元测试还原误差 1e-16）
- `eval_starvla_libero_x5.py` 新增 `--rot-delta-semantics {euler,rotvec}`（默认 euler 不影响 0813）和 `--ik-fk {link6,eef_link}`
- `run_local_client.sh` 固定传 `--rot-delta-semantics rotvec --ik-fk eef_link`
- 验证：world+x α → euler pitch −α（yaw=90° 手推一致）；任意倾斜姿态还原均精确

### 排查过程中排除的假设
- ❌ endpoint wrap loss（训练没加也无碍）：action 空间是 EEF delta，无周期量、无 ±π 边界问题；离线 eval MSE 7e-5 无爆炸
- ❌ 颜色/相机问题：录制视频偏蓝是录制器 RGB/BGR 写反（已修），模型输入颜色一直正确
- ❌ 场景没罐子：罐子在视野内且位置在训练分布（demo 罐位 x∈[−0.26,0.29]、y∈[0.13,0.37] 覆盖很广）
- ❌ go_home 未收敛：闭环 settle 修复后 J5 起始 −1°✅

### 师姐 reference 客户端对比（starvla_ljn_code_20260905/deploy）
她的路径：SDK end_pos rpy 逐元素加 delta → **SDK mode=4** 按其自身约定执行（绕开我们自研 IK 的语义错位）。我们保留 mode=5 diff-IK（mode=4 有奇异/跳分支问题），用正确的 rotvec→欧拉转换对齐语义。
