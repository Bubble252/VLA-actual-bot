# 旋转语义修复（rotvec）

## 我们控制的是什么

- **模型 / 训练契约**：世界系 **轴角（rotvec）增量**  
  \( R_{new} = R_v(\delta)\, R_{cur} \)（数据裁决：重构误差 ≈ 0）
- **IK 引擎（差分 Jacobian）**：extrinsic-XYZ **欧拉角增量**  
  Jacobian 旋转行是 \( \partial(\mathrm{rpy})/\partial q \)

部署时必须做一次翻译：`rotvec_delta_to_euler_increment()`（在 `eval_root/lewm_x5_grasp/src/lewm_x5_grasp/eef_to_joint_ik.py`）。

## 旧行为（0813 默认 / 未传参数）

把 `rot_delta` **逐元素加到 rpy**。在 home 附近 yaw≈90° 时，欧拉 roll 增量的物理效果 ≈ 绕世界 **y** 轴旋转 → 姿态持续漂移（「乱飘」）。

## 本仓默认（已写进 `run_local_client.sh`）

```text
--rot-delta-semantics rotvec
--ik-fk eef_link
```

- `rotvec`：走转换后再进 IK  
- `eef_link`：Jacobian 算在 tip（与训练 tip / bake 对齐），避免 link6 法兰 0.145m 杠杆臂伪平移  

**不要**为了「和 0813 一样」改回 `euler` + `link6`，那是对 **同一批 demo 契约** 的错误执行。

## 离线证据（开发机已做）

详见 `deploy/01_怀疑点排查清单.md`：

1. 训练 parquet 语义裁决：rotvec 前置乘精确成立  
2. demo 动作流过执行链：NEW 贴合、OLD 发散  
3. bug 期录像影子回放：旧路径复现漂移  

真机仍需录像复验闭环（相机 / RTC / 夹爪）。
