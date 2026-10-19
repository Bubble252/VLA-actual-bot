# Stage2 全训 VLM hidden NaN 问题调查总结

> 整理时间：2026-06-06  
> 场景：LIBERO 第二阶段，从 `steps_6000` resume，**全量微调 SpikeBrain-2.0-VL（不冻结）**  
> 目的：记录现象、已做实验与结论，便于请教师兄

---

## 1. 问题一句话

**Stage2 全训 VLM 时，单卡 `per_device_batch_size ≥ 2` 会在 VLM `last_hidden_state` 上出现大面积 NaN（~99% 元素）；`bs=1` 可稳定训练。NaN 出现在 DiT 之前，由 `_sanitize_vlm_hidden` 检测并 clamp。**

---

## 2. 训练背景

| 项目 | 配置 |
|------|------|
| 框架 | `QwenGR00T`（SpikeBrain-2.0-VL + DiT action head） |
| GPU | 6 卡（`CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`） |
| 起点 ckpt | `results/Checkpoints/20260523_0541_libero_all_spb2_gr00t_freeze/checkpoints/steps_6000_pytorch_model.pt` |
| Stage1 | 冻结 VLM，`per_device_bs=32`，稳定 |
| Stage2 目标 | `freeze_modules=""`，全训 VLM + DiT |
| 精度 | bf16 autocast（VLM forward）；DiT fp32 |
| VLM 组件 | flash_attention_2、Liger kernel、gradient_checkpointing、MoBA/SSE-SWA |

### 稳定配置（当前推荐）

```text
per_device_bs = 1
global_bs     = 126  (= 1 × 6 × 21)
grad_accum    = 21
```

日志：`results/Checkpoints/train_logs/run_libero_stage2_full_from6k_20260605_215351.log`  
现象：**无 VLM guard 警告**，可训练参数 ~5743M。

### 出问题配置

```text
per_device_bs = 2
global_bs     = 132  (= 2 × 6 × 11)
grad_accum    = 11
```

日志：`results/Checkpoints/train_logs/run_libero_stage2_full_from6k_20260605_214936.log`  
现象：持续 `VLM guard`，`bad≈3,020,000` / 总元素 `2×594×2560≈3,041,280`（~99% NaN）。

---

## 3. NaN 出现位置（代码路径）

```text
examples → build_qwenvl_inputs(padding=True)
         → qwen_vl_interface.forward()  [bf16, 全训有梯度]
         → last_hidden_state [B, seq_len, 2560]
         → _sanitize_vlm_hidden()      ← NaN 在此检测
         → repeat → DiT
```

关键代码：

- `starVLA/model/framework/QwenGR00T.py`：`forward()`、`_sanitize_vlm_hidden()`
- `starVLA/model/modules/vlm/spikebrain.py`：`build_qwenvl_inputs()`（`padding=True` 拼 batch）

日志特征：

```text
[VLM guard] hidden sanitize (train): bad=3041280 nan=3041280 +inf=0 -inf=0
```

---

## 4. 已做实验与结论

### 4.1 六卡 ABC 探测（`per_device_bs=2`）

脚本：`examples/LIBERO/train_files/run_probe_stage2_bs2_abc.sh`  
汇总日志：`results/Checkpoints/train_logs/probe_stage2_bs2_bc_6gpu_20260605_230516.log`

| 实验 | 配置 | 结论 |
|------|------|------|
| **A** | `bs=2` + **冻结 VLM**（eval+no_grad） | ✅ **0 次 VLM guard** → 不是「batch>1 本身」的问题 |
| **B** | `bs=2` + 全训 + **关 GC** | ❌ step1/2 正常，step3 ~100% NaN |
| **C** | `bs=2` + 全训 + **关 Liger** | ❌ step1 正常，step2/3 ~100% NaN |

**结论：**

- 根因组合是 **`per_device_bs>1` + VLM 全训反传**，不是单独 batch 大或单独全训。
- **关 gradient_checkpointing、关 Liger 均不能根治。**

### 4.2 VLM micro-batch=1 绕路

实现：`QwenGR00T._forward_vlm()`，全训且 `B>1` 时按 `vlm_micro_batch_size=1` 循环 forward，再 `pad_sequence` 拼接。

配置项：`framework.qwenvl.vlm_micro_batch_size: 1`（yaml 已加）

| 探测 / 训练 | grad_accum | 现象 |
|-------------|------------|------|
| `probe_microbatch_bs2_global132`（3 steps） | 11 | step1/2 ✅，step3 ❌ |
| 正式训练 `235845.log` | 11 | **step1 ✅，step2 起持续 NaN**（82+ 次 guard） |

**结论：micro-batch 绕路未能解决长跑 NaN**；step1 正常、第一次 optimizer 更新后 step2 开始炸，与「权重被第一步更新打坏」或「全训反传累积问题」的模式一致。

### 4.3 单卡探测（失败，仅供参考）

早期单卡 probe 出现 `CUDA illegal memory access`，与六卡 DDP+DeepSpeed 环境不一致，**结论以六卡实验为准**。

### 4.4 grad_accum=1 的短探测（已废弃）

`global=6/12, grad_accum=1` 时，`bs=1` 也从 step2 起 NaN，更像 optimizer 更新过激进，**不能代表正式训练**；后续探测已改为 `grad_accum=21/11`。

---

## 5. 已排除 / 已确认

| 假设 | 结果 |
|------|------|
| 显存 OOM 导致 NaN | ❌ `bs=2` 峰值 ~43–53GB，80GB 卡够用 |
| DiT 产生 NaN 回传 | ❌ NaN 在 repeat **之前** 的 VLM hidden 已存在 |
| 忘记解冻（仍冻结 VLM） | ❌ 曾踩过 yaml 默认 freeze；修后 Trainable=5743M |
| `bs>1` 单独导致 NaN | ❌ 冻结 + `bs=2` 完全正常（实验 A） |
| 关 GC 可修复 | ❌ 实验 B 否定 |
| 关 Liger 可修复 | ❌ 实验 C 否定 |
| micro-batch=1 可修复 | ❌ 正式训练 step2 起仍 NaN |
| Stage1 `bs=32` 能跑说明全训 bs=32 也行 | ❌ Stage1 是 **冻结+no_grad**，与全训不是同一路径 |

---

## 6. 当前理解与待请教点

### 6.1 现象模式（较稳定）

1. **冻结 VLM**：`bs=1` / `bs=2` 均正常。  
2. **全训 VLM + `bs=1`**：长跑稳定（`215351.log`）。  
3. **全训 VLM + `bs≥2`**：  
   - 常见 **step1（或前 1–2 个 optimizer step）正常**；  
   - **第一次权重更新后**，VLM hidden 大面积 NaN；  
   - NaN 比例 ~99%，形状仍为 `[2, 594, 2560]`（每 rank）。

### 6.2 推测方向（需师兄确认）

1. **Batched 变长序列 + 全训反传**  
   `build_qwenvl_inputs` 使用 `padding=True`，多条不等长样本拼 batch 后，SpikeBrain 内部 `unpad_input` / SSE-SWA / MoBA / flash-attn 在 **backward** 时数值不稳定（bf16）。

2. **第一步 optimizer 更新后权重恶化**  
   micro-batch 与 `bs=2` 均在 step2 起炸，是否应对 VLM 使用更小 LR、warmup 更长、或 fp32 forward/loss？

3. **是否与 SpikeBrain 自定义算子 + DeepSpeed ZeRO-2 交互有关**  
   六卡 + 全训 5.7B 参数时，是否有已知坑（MoBA `index_reduce` warning 等）？

4. **micro-batch 实现是否应改**  
   当前逐样本 forward 再 pad 拼接，梯度是否仍经有问题路径？是否应改为「完全独立的 B=1 backward + 手动累加」或 VLM 全 fp32？

### 6.3 想请教师兄的问题清单

1. SpikeBrain-2.0-VL 全训时，**推荐的最大 per-device batch** 是否就是 1？是否有官方 batched 训练经验？  
2. `padding=True` 拼 batch 是否是已知雷区？是否应改为 **按长度分桶** 或 **Dynamic batching**？  
3. 全训分支是否建议 **VLM forward 强制 fp32** 或 **只对 GDN/attention 子模块 fp32**？  
4. step1 正常、step2 全 NaN，更像是 **LR/optimizer** 还是 **forward kernel** 问题？  
5. 是否有办法在 backward 中间层 hook，定位 **哪一层** 先出 NaN？

---

## 7. 相关文件与日志索引

### 脚本

| 文件 | 说明 |
|------|------|
| `examples/LIBERO/train_files/run_libero_train_all_ljn_stage2_full.sh` | Stage2 启动脚本（当前为 bs2+micro，建议改回 bs1） |
| `examples/LIBERO/train_files/run_probe_stage2_bs2_abc.sh` | 六卡 ABC 探测 |
| `examples/LIBERO/train_files/run_probe_vlm_microbatch_6gpu.sh` | micro-batch 吞吐对比 |

### 代码改动

| 文件 | 改动 |
|------|------|
| `starVLA/model/framework/QwenGR00T.py` | `_forward_vlm()` + `vlm_micro_batch_size` |
| `starVLA/training/train_starvla.py` | `trainer.resume_step` 支持 LR 从 0 warmup |
| `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | `vlm_micro_batch_size: 1` |

### 关键日志

| 日志 | 说明 |
|------|------|
| `run_libero_stage2_full_from6k_20260605_215351.log` | ✅ **稳定**：bs=1, global=126 |
| `run_libero_stage2_full_from6k_20260605_214936.log` | ❌ bs=2 无 micro，持续 NaN |
| `run_libero_stage2_full_from6k_20260605_235845.log` | ❌ bs=2 + micro-batch，step2 起 NaN |
| `probe_stage2_bs2_expA_20260605_222329.log` | ✅ 冻结+bs=2，0 guard |
| `probe_stage2_bs2_expB_20260605_230516.log` | ❌ 全训+关 GC |
| `probe_stage2_bs2_expC_20260605_231103.log` | ❌ 全训+关 Liger |
| `probe_microbatch_bs2_global132_20260605_234859.log` | micro-batch 短探测 |

---

## 8. 数据流示意（便于对照）

```text
Stage2 每个 micro-batch（每 rank）:
  DataLoader: per_device_bs 条样本
       ↓
  build_qwenvl_inputs(padding=True)  → input_ids, attention_mask, pixel_values ...
       ↓
  [全训] qwen_vl_interface.forward (bf16, 有 grad)
       ↓
  last_hidden [B, L, 2560]   ← NaN 出现在这里（B=2 时常见）
       ↓
  _sanitize_vlm_hidden (nan_to_num clamp ±100)
       ↓
  repeat(repeated_diffusion_steps=6)  → DiT 输入 [B*6, L, 2560]
       ↓
  action_loss → backward → (grad_accum 次后) optimizer.step()
```

**B=1 时**：`L` 为单条序列长度，无跨样本 padding，全训稳定。  
**B=2 时**：两条序列 pad 到同一 `L`，走 batched 变长 attention 路径；全训反传易 NaN。

---

## 9. 当前建议（给师兄看后的暂定方案）

1. **生产训练**：回退 `per_device_bs=1, global=126, grad_accum=21`，已验证稳定。  
2. **若必须提高吞吐**：优先在师兄指导下尝试  
   - VLM 全训 forward **fp32**；或  
   - **按样本 B=1 forward + 显式梯度累加**（非简单 concat hidden）；或  
   - 降低 `qwen_vl_interface` LR / 延长 warmup。  
3. **不建议** 在 NaN 未解决前继续 `bs=2` 长跑（即使加 micro-batch）。

---

## 10. 复现命令（六卡）

```bash
# 稳定 baseline
bash examples/LIBERO/train_files/run_libero_train_all_ljn_stage2_full.sh
# （需将脚本内 PER_DEVICE_BS=1, TARGET_GLOBAL_BS=126, 去掉或保留 micro-batch 均可）

# ABC 探测
bash examples/LIBERO/train_files/run_probe_stage2_bs2_abc.sh

# micro-batch 对比
bash examples/LIBERO/train_files/run_probe_vlm_microbatch_6gpu.sh
```

环境：`/9950backfile/zhangjh/.conda/envs/starvla`，DeepSpeed ZeRO-2，`TRITON_CACHE_DIR=/data/local/zhangjh/.cache/triton_autotune`（本地盘，避免 NFS 警告）。

---

## 11. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-06-06 | 初版：汇总 ABC 探测、micro-batch 试验与稳定/不稳定配置 |
