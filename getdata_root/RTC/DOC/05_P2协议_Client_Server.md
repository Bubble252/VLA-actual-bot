# 05 · P2 协议：Client ↔ Server（轨 B / ΔEEF）

> 日期：2026-08-08  
> 对应计划：`03_执行计划.md` → P2  
> 前置：`04_P1挂载点清单.md`  
> 目标：两边可同时按本表改代码；**旧 client 不传 RTC 字段时行为与现网完全一致**。

---

## 1. 联调目标（写死）

| 项 | 选定 |
|----|------|
| 优先端口 | **5694** |
| 备用端口 | 5695（协议相同，仅权重不同） |
| 权重（5694） | `.../20260716_1738_arx_x5_libero_arx_x5_full_8gpu/checkpoints/steps_98000_pytorch_model.pt` |
| `unnorm_key` | `franka` |
| 动作 | ΔEEF 6 维 + 夹爪 1 维；`action_chunk_size = T = 8` |
| 本期不做 | 5794 关节口 |

连接握手后 server 仍先发 `metadata`（含 `action_chunk_size`、`default_unnorm_key`、`action_keys` 等），与现网一致。

---

## 2. 现网消息形态（兼容基线）

传输：websocket + `msgpack_numpy`（与现网相同）。

### 2.1 现网请求（ModelClient → server）

当前 flat dict（无 `type` 时 server 默认当 `infer`）：

```python
{
    "examples": [example],          # example: image / lang / state...
    "unnorm_key": "franka",
    "do_sample": False,
    "use_ddim": ...,                # 现网可能仍带；RTC 无关，忽略即可
    "num_ddim_steps": ...,
}
```

server：`PolicyServerWrapper.predict_action(**msg)` → 返回物理 `actions`。

### 2.2 现网响应

```python
{
    "status": "ok",
    "ok": True,
    "type": "inference_result",
    "request_id": "default",
    "data": {
        "actions": np.ndarray,      # shape (B, T, 7)，已 unnorm 的物理量
    },
}
```

`actions[..., :6]` = Δ xyzrpy；`actions[..., 6]` = gripper 命令。  
client 再拆成 `world_vector` / `rotation_delta` / `open_gripper`，并可乘 `pos_scale` / `rot_scale`（**部署缩放，不属于训练 norm**）。

---

## 3. RTC 扩展请求字段

在**现网请求上增量增加可选字段**。未出现或显式关闭 → 走现网路径。

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `rtc_enabled` | `bool` | 否 | `false` | `true` 才启用轨 B guidance |
| `rtc_prefix` | `float32` 数组 | 当 enabled 时建议有 | 无 | 旧 chunk leftover，见 §4 |
| `rtc_prefix_space` | `str` | 否 | `"physical"` | `"physical"` \| `"normalized"` |
| `inference_delay` | `int` | enabled 时建议有 | `0` | 已过期步数；见 §6 |
| `execution_horizon` | `int` | 否 | `4` | overlap 强约束长度；须 `1..T`，建议 `≤6` |
| `rtc_max_guidance_weight` | `float` | 否 | `10.0` | 可选覆盖；P3 可先忽略用服务端常量 |

**兼容规则（硬性）：**

1. 无任何 `rtc_*` 字段 → 与现网 bit-identical 路径（不进 guidance）。  
2. `rtc_enabled` 缺省 / `false` / `0` → 忽略其余 RTC 字段。  
3. `rtc_enabled=true` 但 `rtc_prefix` 缺失或长度为 0 → **降级为无 RTC**（打 warning，不报错），保证能跑。  
4. 未知字段忽略，不抛错（现网对多余 kwargs 需在 wrapper 层 pop 掉，避免传入 framework 炸开）。

### 3.1 推荐请求示例（RTC 开）

```python
{
    "examples": [example],
    "unnorm_key": "franka",
    "rtc_enabled": True,
    "rtc_prefix": leftover_physical,   # float32, shape (L, 7) 或 (L, 6)
    "rtc_prefix_space": "physical",
    "inference_delay": 2,
    "execution_horizon": 4,
}
```

### 3.2 Server 侧处理顺序（给 P3）

```text
1. 解析 RTC 字段；pop 出后再调 framework，避免 **kwargs 污染
2. if not rtc_enabled or prefix empty:
       现网：framework.predict_action → unapply → return
3. if rtc_prefix_space == "physical":
       prefix_norm = PolicyNormProcessor.apply_actions(prefix)   # P3 需补此 API
   else:
       prefix_norm = prefix
4. 维掩码：强制 gripper 维不参与 guidance（见 §5）
5. framework.predict_action(..., prev_chunk_left_over=prefix_norm, ...)
6. unapply → 返回物理 actions（响应形状不变）
```

---

## 4. `rtc_prefix` 形状与空间

| 约定 | 要求 |
|------|------|
| dtype | `float32` |
| shape | `(L, 7)` **推荐**；或 `(L, 6)`（仅 ΔEEF，server pad 第 7 维为 0） |
| `L` | `1 ≤ L ≤ T`（T=8）；通常为「旧 chunk 未执行长度」 |
| 时间对齐 | `prefix[0]` = **下一控制步**将要执行的动作（不是已执行完的最后一步） |
| 默认空间 | **`physical`**：与 server 返回的 `data.actions` **同一刻度**（unnorm 后、client `pos_scale/rot_scale` **之前**） |
| 可选空间 | `"normalized"`：已与 DiT 输入同空间；server 不再 apply |

**严禁**把下列量直接当 prefix（除非已逆变换回去）：

- 乘过 `pos_scale` / `rot_scale` 后的增量；  
- 积分后的绝对末端位姿；  
- 夹爪映射后的执行电压/行程（`gripper_open/close`）。

**Client 维护 leftover 的推荐做法：**

```text
每次拿到 data["actions"][0]  # (8,7) 物理
存入 queue，逐步弹出执行
发起新推理前：
  leftover = queue 中尚未执行的行   # (L,7)
  delay = 自发起推理以来又消费的步数（见 §6）
  # 注意：leftover 用原始物理 chunk，不要用 scale 后的值
```

---

## 5. 夹爪约定

| 规则 | 说明 |
|------|------|
| client | `rtc_prefix` 可带第 7 维，但 **不参与** guidance；也可只传 `(L,6)` |
| server | 维掩码固定 `RTC_DIM_MASK = [1,1,1,1,1,1,0]`；第 7 维 guidance 权重强制 0 |
| 输出 | 夹爪仍走模型生成 + 现有 unnorm；client 继续 `_binarize` / `_map_gripper` |
| 语义 | 开合命令，不做连续平滑/对齐 |

与 `dataset_statistics.json` 中 `franka.action.mask` 的 `[T,T,T,T,T,T,F]` 一致。

---

## 6. `inference_delay` 怎么估

含义：从 **发起本次推理** 到 **新 chunk 开始被采用**，中间已经按控制周期消耗的步数。这些步对应的旧动作已执行（或必执行），新 chunk 的前 `delay` 步应视为过期。

### 6.1 简单算法（P4 先用这个）

```text
控制周期 dt_ctrl = 1 / control_hz
t_start = 发起 websocket 请求的时刻
... 等待响应 ...
t_end = 收到响应的时刻
steps_elapsed = floor((t_end - t_start) / dt_ctrl)

# 若异步：推理期间 client 仍在按旧 chunk 往前走
inference_delay = clamp(steps_elapsed, 0, L_leftover_at_send)

# 若在发送前 leftover 已是「当前未执行尾巴」，
# 且发送后到收到前又消费了 d 步，则：
#   送出的 prefix 应是 send 时刻的 leftover
#   inference_delay = d
# server 用 prefix 的前 delay 步做强冻结（权重 1）
```

更稳的实现（推荐）：

1. 发送时记下 `send_index`（全局已执行步计数）和当时的 `leftover`；  
2. 收到后 `inference_delay = current_index - send_index`；  
3. `execution_horizon = min(4, L, T)`（可配置）。

### 6.2 护栏

| 条件 | 行为 |
|------|------|
| `delay < 0` | 置 0 |
| `delay > L` | clamp 到 `L`；若 `L==0` 则关闭 RTC |
| `execution_horizon > T` | clamp 到 `T`（8） |
| `execution_horizon > L` | clamp 到 `L`（与 LeRobot RTCProcessor 一致） |

---

## 7. 响应扩展（可选，兼容）

**响应主路径不变**：仍为 `data.actions` 物理 `(B,T,7)`。

可选调试字段（仅 `rtc_enabled` 且 server 开 debug 时）：

```python
"data": {
    "actions": ...,
    "rtc": {
        "applied": True,
        "inference_delay": 2,
        "execution_horizon": 4,
        "prefix_len": 5,
        "prefix_space_used": "normalized",
        "fell_back": False,          # True 表示降级无 RTC
        "fallback_reason": "",
    },
}
```

旧 client 只读 `data.actions`，忽略 `rtc` 即可。

---

## 8. Wrapper 参数剥离约定（防炸）

`WebsocketPolicyServer` 当前：`predict_action(**msg)`。  
msg 可能含 `type` / `request_id` 等。P3 实现时建议在 `PolicyServerWrapper.predict_action` 入口：

```text
显式取出：
  examples, unnorm_key,
  rtc_enabled, rtc_prefix, rtc_prefix_space,
  inference_delay, execution_horizon, rtc_max_guidance_weight
其余 RTC/路由字段 pop 掉，再调用 framework
```

Framework / action head 只收：

```text
prev_chunk_left_over  # normalized, (B,T,7) 或 (T,7) 由内部 pad
inference_delay       # int
execution_horizon     # int
rtc_enabled           # bool
```

---

## 9. 归一化复用（写进协议，避免分叉）

| 方向 | API | 谁做 |
|------|-----|------|
| normalized → physical | 现有 `PolicyNormProcessor.unapply_actions` | server（现网已有） |
| physical → normalized | **新增** `PolicyNormProcessor.apply_actions`（对称封装 `transform.apply`） | server（推荐） |
| 训练 stats | ckpt 旁 `dataset_statistics.json` + `unnorm_key=franka` | 与 unnorm 同一套 |

Client **默认不自己 norm**；只传 physical leftover（来自上次 `data.actions`）。  
若传 `rtc_prefix_space="normalized"`，必须保证与该 ckpt 的 DiT 空间一致（一般仅调试用）。

---

## 10. 双方改动清单（可开干）

### Server（P3）

- [ ] `PolicyNormProcessor.apply_actions`  
- [ ] `PolicyServerWrapper.predict_action` 解析 RTC 字段 + 默认关闭  
- [ ] `FlowmatchingActionHead.predict_action` Euler guidance  
- [ ] gripper 维掩码强制 0  
- [ ] 失败回退无 RTC + 可选 `data.rtc`  

### Client（P4）

- [ ] chunk 队列；异步推理时保留 physical leftover  
- [ ] 估计 `inference_delay`（§6）  
- [ ] 请求附加 RTC 字段；开关默认关  
- [ ] leftover 取自 unnorm 后、scale 前  
- [ ] 夹爪逻辑不变  

### 本期不改

- 5794 关节协议  
- MuJoCo / 离线验收门槛  
- 强制依赖 `lerobot` pip 包  

---

## 11. 验收口径（P2 完成定义）

1. 本文字段表可供 server/client 对照实现。  
2. 写明：旧请求无 RTC 字段 ⇒ 行为不变。  
3. 写明：prefix 空间、夹爪旁路、delay 估法。  
4. 联调口写死：**5694 优先，5695 备用**。
