# SpikeBrain-2.0-VL（SPB2）+ QwenGR00T：LIBERO 训练备忘

> **用途**：记录在 starVLA 上用 **SpikeBrain-2.0-VL** 作 VLM、**QwenGR00T** 在 LIBERO 上训练时的环境、配置、代码改动与排错。  
> **最后更新**：2026-05-23（§20 bs32 显存/加速）  
> **相关对话**：Cursor agent transcript `3b63f479-b0c2-4f7d-b3c9-e730997db6b4`

---

## 更新记录

| 日期 | 内容 |
|------|------|
| 2026-05-22 | 初版：SPB2 接入、Liger、GDN fp32、环境索引、FAQ |
| 2026-05-23 | 补充：`flash_attn` / FLA 版本与训练路径；日志警告表修正（来源 FLA）；Triton/Python **不建议升级**；`TRITON_CACHE_DIR` 项目目录；`ds_config` 弃用项；终端 `getcwd` / `pip` 问题 |
| 2026-05-23 | **单卡 GPU6 bs 探测**（`probe_single_gpu_bs.sh`）：bs=4/2/1 均 OOM；双卡 ZeRO-2 bs=1 仍为推荐配置 |
| 2026-05-23 | **GR00T 对齐 `trash/starVLA-starVLA`**：yaml / `QwenGR00T` / `share_tools` / dataloader / `GR00T_ActionHeader`（见 §18） |
| 2026-05-23 | **复查修复**：`GR00T_ActionHeader` 改为读 `action_horizon`；`train_starvla_cotrain` 补 `apply_config_compat`（见 §19） |
| 2026-05-23 | **bs32 双卡训练**：NaN 防护、repeat=6、显存 57→47 GB、加速尝试结论（见 §20） |

---

## 1. 一句话目标

- **框架**：`QwenGR00T`（VLM 特征 → DiT action head）
- **VLM**：本地 `spikebrain/SpikeBrain-2.0-VL`（~5.58B），替代原 Qwen3-VL
- **数据**：LIBERO LeRobot，`data_mix=libero_all`
- **训练**：GPU 6/7，DeepSpeed ZeRO-2，`per_device_batch_size=1`

---

## 2. 快速索引（最常找的路径）

| 用途 | 路径 |
|------|------|
| **启动脚本** | `examples/LIBERO/train_files/run_libero_train.sh` |
| **训练配置 YAML** | `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |
| **DeepSpeed 配置** | `starVLA/config/deepseeds/ds_config.yaml` |
| **训练入口** | `starVLA/training/train_starvla.py` |
| **VLM 封装** | `starVLA/model/modules/vlm/spikebrain.py` |
| **Liger patch** | `starVLA/model/modules/vlm/spb2_liger.py` |
| **框架** | `starVLA/model/framework/QwenGR00T.py` |
| **SPB2 模型代码** | `spikebrain/SpikeBrain-2.0-VL/` |
| **FLA（editable）** | `spikebrain/envs/flash-linear-attention_dev` |
| **MoBA** | `spikebrain/envs/MoBA/` |
| **Triton 缓存** | `$STARVLA_ROOT/.cache/triton_autotune` |
| **训练日志** | `results/Checkpoints/train_logs/` |
| **Checkpoint** | `results/Checkpoints/<YYYYMMDD>_<HHMM>_libero_all_spb2_gr00t_freeze/`（`run_libero_train.sh` 按启动日期+时分命名） |
| **lmms 参考** | `trash/lmms-engine/` |
| **本文档** | `examples/LIBERO/docs/SPB2_LIBERO_TRAINING_NOTES.md` |

---

## 3. Conda 环境（务必用 `starvla`）

训练脚本**写死**（全小写 **`starvla`**，不是 `starVLA` / `libero`）：

```bash
cd /9950backfile/zhangjh/wh/starvla_wh/starVLA
export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
```

### 3.1 已验证版本（`pip list` / 训练日志）

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.10.20 | **不建议**为消警告升到 3.11（见 §10） |
| torch | 2.5.1+cu121 | 自带 **triton 3.1.0** |
| transformers | 5.8.0 | |
| **flash_attn** | **2.6.3** | site-packages；lmms 同主版本号，wheel 可能为 cu124/torch2.6/py311 |
| **flash-linear-attention** | **0.4.0** | **editable** → `spikebrain/envs/flash-linear-attention_dev` |
| liger-kernel | >=0.6.5,<0.7 | **0.6.4** 在 tf 5.x 下 `HybridCache` 导入失败 |

### 3.2 自检

```bash
cd /9950backfile/zhangjh/wh/starvla_wh/starVLA
export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
export PYTHONNOUSERSITE=1
which python   # 应为 .../envs/starvla/bin/python

python -c "import sys, fla, flash_attn; print(sys.executable); print('fla', fla.__version__); print('flash_attn', flash_attn.__version__)"
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; from flash_attn import flash_attn_func; print('ok')"
python -c "from starVLA.model.modules.vlm.spb2_liger import _LIGER_AVAILABLE; print('liger', _LIGER_AVAILABLE)"
pip show flash-linear-attention flash-attn liger-kernel | grep -E '^(Name|Version|Editable)'
```

### 3.3 终端注意

- 不要在已删除目录（如 `trash/`）里跑 `pip` / `python`，会出现 **`getcwd() failed`**、**`The folder you are executing pip from can no longer be found`** → 先 `cd` 到 `STARVLA_ROOT`。

### 3.4 安装 Liger

```bash
pip install 'liger-kernel>=0.6.5,<0.7'
```

---

## 4. 如何启动训练

```bash
cd /9950backfile/zhangjh/wh/starvla_wh/starVLA
bash examples/LIBERO/train_files/run_libero_train.sh
```

脚本会自动：

- `cd` 到 `STARVLA_ROOT`
- `PATH`、`PYTORCH_CUDA_ALLOC_CONF`、`TRITON_CACHE_DIR`（项目内 `.cache/triton_autotune`）
- 清除 SpikeBrain HF 模块缓存（§6）
- `CUDA_VISIBLE_DEVICES=6,7`，2 卡 DeepSpeed ZeRO-2
- 日志：`results/Checkpoints/train_logs/run_libero_train_<时间戳>.log`

**WandB**：`starVLA_Libero` / `wuhang23`（可改）。

---

## 5. 关键 YAML（`starvla_cotrain_libero.yaml`）

当前配置已按 **`trash/starVLA-starVLA`** 的 QwenGR00T 语义对齐，VLM 仍为 SpikeBrain（详见 §18）。

```yaml
version_id: "0.21"

framework:
  name: QwenGR00T
  qwenvl:
    base_vlm: .../spikebrain/SpikeBrain-2.0-VL
    attn_implementation: flash_attention_2
    ensure_gdn_fp32_params: true
    use_liger_kernel: true
  action_model:
    action_horizon: 8
    repeated_diffusion_steps: 8
    noise_beta_alpha: 1.5
    noise_beta_beta: 1.0
    noise_s: 0.999
    diffusion_model_cfg:
      num_layers: 16
      output_dim: 1024
      # cross_attention_dim 运行时由 VLM hidden_size 覆盖

datasets:
  vla_data:
    action_type: delta_qpos
    CoT_prompt: "Your task is {instruction}. ..."

trainer:
  freeze_modules: 'qwen_vl_interface'
  learning_rate:
    base: 2.5e-05
    action_model: 1.0e-04
  gradient_accumulation_steps: 64   # 单卡 bs=1 时等效 batch≈64
```

**SPB2 模型配置**（`config.json`）：`linear_attn_type: "gdn"`。

**显存**：双卡约 ~72GB/卡，`per_device_batch_size=1` 为常态。

### 5.1 单卡 batch size 探测（GPU 6，ZeRO-2，`num_processes=1`）

脚本：`examples/LIBERO/train_files/probe_single_gpu_bs.sh`  
汇总：`results/Checkpoints/train_logs/bs_probe_gpu6_summary.txt`

| per_device_bs | 结果 | OOM 位置 | 备注 |
|---------------|------|----------|------|
| 4 | **OOM** | 第 1 次 backward（梯度归约） | 再申请 ~10.65 GiB 失败 |
| 2 | **OOM** | Step0 跑完 64×grad_accum 后 **`optimizer.step`** | Adam `exp_avg`，再申请 ~**20.79 GiB** |
| 1 | **OOM** | 同上 **`optimizer.step`** | 同样 ~20.79 GiB |

**结论**：单卡 80GB **无法**稳定全量训 SPB2+VLA（连 bs=1 也在 optimizer 状态分配时 OOM）。**必须用双卡 ZeRO-2**（参数/优化器分摊），与已跑通的双卡 `per_device_batch_size=1` 一致。勿在单卡上加大 batch。

---

## 6. SpikeBrain HF 缓存

路径：`~/.cache/huggingface/modules/transformers_modules/SpikeBrain_hyphen_2_dot_0_hyphen_VL`

改本地 `modeling_spb2_vl.py` 后须与缓存一致；脚本**每次启动会删缓存**（稳定后可注释 `run_libero_train.sh` 对应段）。

---

## 7. 训练时注意力栈：flash_attn / FLA / MoBA 是否在用

**结论**：三个都在用；不是“装了没用”。

| 层类型 | 数量（约） | 实现 | 依赖 |
|--------|------------|------|------|
| **GDN + SSE-SWA** | ~26 层 | `SSEGDNH`（`sse_swa.py`） | **FLA** `chunk_gated_delta_rule` 等 + **flash_attn**（SWA 窗口） |
| **MoBA 稀疏** | 9 层 | `MoBAAttention` | `spikebrain/envs/MoBA/moba_efficient.py` |
| **全注意力** | 1 层（layer 35） | `attn.py` | **flash_attn** `flash_attn_varlen_func` |

另外：

- `attn_implementation=flash_attention_2`：传给 `from_pretrained`；`spikebrain.py` 仅在 **import flash_attn 失败** 时回退 `sdpa`。
- **Liger**：只 patch Norm / MLP / RoPE，**不替代** flash_attn 或 FLA。

**日志佐证**：

- `attn_implementation=flash_attention_2`
- `moba_efficient.py:226 index_reduce`（MoBA forward）
- 启动时 FLA 的 Triton/Python 推荐警告（说明 `import fla` 已执行）

**版本是否“对”**：`flash_attn 2.6.3` + `fla 0.4.0 editable` 与当前 SPB2-GDN 训练匹配；lmms 固定 FLA **git commit**，本地 dev 树一般与 SPB2 捆绑一致（可用 `git -C spikebrain/envs/flash-linear-attention_dev rev-parse HEAD` 核对）。

---

## 8. 代码改动清单（starVLA）

| 文件 | 要点 |
|------|------|
| `spikebrain.py` | `AutoModelForCausalLM`、backbone forward、GDN fp32、Liger（加载后、LoRA 前） |
| `spb2_liger.py` | SPB2 显式 Liger patch；`fused_linear_cross_entropy=False` |
| `modeling_spb2_vl.py` | `_tied_weights_keys` → dict（tf 5.8） |
| `QwenGR00T.py` | GR00T 对齐：`merge_framework_config`、`action_horizon`、`encoder_attention_mask`；冻 VLM 时 `no_grad` + `last_hidden_state` |
| `share_tools.py` | 新增 `merge_framework_config`、`apply_config_compat`（config 0.21） |
| `GR00T_ActionHeader.py` | `predict_action` 支持 `encoder_attention_mask` |
| `gr00t_lerobot/data_config.py` | `Libero4in1DataConfig.state_indices = range(-16, 0)` |
| `train_starvla.py` | `apply_config_compat`；显存日志；DeepSpeed prepare 后 `ensure_gdn_fp32_params` |
| `ds_config.yaml` | `cpu_offload` → `offload_optimizer.device: none` |

---

## 9. Liger Kernel

| 项 | 说明 |
|----|------|
| lmms 对 SPB2 | 配置里可写 `use_liger_kernel`，但 **未注册 spb2_vl**；starVLA **显式** `spb2_liger.py` |
| 效果 | 用户反馈：指标影响不大，**训练更快** |
| 确认 | 日志：`[SpikeBrain] Liger kernel ENABLED on SPB2-VL` |
| 生效 | 改 yaml/装包后 **必须重启** 训练进程 |

---

## 10. Triton 3.2 / Python 3.11：要不要升级？

**建议：现阶段不要升级。** 继续 **Py3.10 + triton 3.1.0（随 torch 2.5.1）** 完成 LIBERO 训练。

| | 不升级（当前） | 升级到“推荐” |
|--|----------------|--------------|
| **状态** | 已跑通加载、forward、loss | 需新 conda + 全套重装 |
| **Triton 3.2** | 与 torch 2.5.1 官方搭配一致 | 单独升可能和 torch 拧巴 |
| **Python 3.11** | FLA 官方下限 3.10 | 主要多 FLA 部分 `torch.compile`；GDN 主路径仍是 Triton |
| **警告** | 启动时 FLA `fla/utils.py` 打印 | 可减少 log，非功能必需 |

**警告来源**（易误判为 Liger）：`spikebrain/envs/flash-linear-attention_dev/fla/utils.py`（`check_environments()`）；`fla/ops/rwkv7/fused_addcmul.py`（Py3.10 无 compile，**GDN 主路径不依赖此文件**）。

**何时再考虑**：FLA/Triton **Traceback**、换 H100 等新卡、训练告一段落后愿做 100 step 回归 → **新建环境**，勿在跑着训的 `starvla` 里只 `pip install triton==3.2`。

---

## 11. FAQ（已踩坑）

| 现象 | 处理 |
|------|------|
| `_tied_weights_keys` list → AttributeError | dict + 清 HF 缓存 |
| 无 `last_hidden_state` | backbone + `BaseModelOutputWithPast` |
| 权重 0% / MISSING | `AutoModelForCausalLM` |
| `PYTORCH_ALLOC_CONF` 无效 | 用 **`PYTORCH_CUDA_ALLOC_CONF`** |
| Liger `HybridCache` | `liger-kernel>=0.6.5` |
| `pip` / `getcwd` 报错 | `cd STARVLA_ROOT`，勿在已删的 `trash/` |
| `action_model lr=0.0001` 与 yaml 1e-5 不符 | 重启后看 LR Group；仍不对再查 `build_param_lr_groups` |

---

## 12. lmms-engine 参考（`trash/lmms-engine`）

**已采纳**：grad checkpoint、flash_attention_2、GDN fp32、`PYTORCH_CUDA_ALLOC_CONF`、Liger 思路（自实现）、`flash_attn 2.6.3` / FLA 同源思路。

**不宜照搬**：FSDP2、sequence packing、全 LM loss、lmms 分组 lr 5e-5、lmms 内 liger 自动注册。

---

## 13. 重启训练验证清单

1. `[run_libero_train] python=.../starvla/bin/python`
2. `[run_libero_train] TRITON_CACHE_DIR=.../starVLA/.cache/triton_autotune`
3. `[SpikeBrain] loaded ... SPB2VLForConditionalGeneration | attn_implementation=flash_attention_2`
4. `[SpikeBrain] Liger kernel ENABLED ...`（若开 liger）
5. `[SpikeBrain] GDNFP32Params: ensured float32 ...`
6. `Loading weights: 100%`（1130/1130）
7. Step 0+：`action_dit_loss`，无 Traceback

---

## 14. 环境变量速查

| 变量 | 值 |
|------|-----|
| `PATH` | `.../envs/starvla/bin` 前置 |
| `PYTHONNOUSERSITE` | `1` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True`（不支持时仅多一行 warning） |
| `TRITON_CACHE_DIR` | `$STARVLA_ROOT/.cache/triton_autotune`（项目在 NFS，DeepSpeed 仍可能提示 NFS；卡住可改 `/tmp/$USER/triton_cache`） |
| `CUDA_VISIBLE_DEVICES` | `6,7` |

---

## 15. 训练日志常见警告

**原则**：loss 正常、无 Traceback → 多数可忽略。

| 警告 | 来源 | 建议 |
|------|------|------|
| Triton cache on NFS | DeepSpeed | 已设项目内缓存；NFS 提示可忽略；退出卡住 → `/tmp` |
| expandable_segments not supported | PyTorch | 忽略或删掉该 env |
| **Triton 3.1 < 3.2** | **FLA** `fla/utils.py` | **不升级**（§10） |
| **Python 3.10 < 3.11** | **FLA** | **不升级** |
| torch.compile not available in 3.10 | **FLA** rwkv7 路径 | GDN 训练可忽略 |
| docstring `[ERROR]` rope_deltas 等 | transformers 检查 | 非训练失败 |
| DeepSpeed `cpu_offload` deprecated | 旧 ds_config | 已改 `offload_optimizer`；重启消失 |
| MoBA `index_reduce` beta | MoBA forward | 忽略；可 `PYTHONWARNINGS=ignore:...index_reduce...` |
| torch.autocast outside DeepSpeed | 混精 | 常见，bf16 由 DS 管 |
| OMP_NUM_THREADS | torchrun | 信息 |

**必须停训**：Traceback、OOM、权重 0%、loss nan。

---

## 16. 可选后续

- 注释掉「每次清 SpikeBrain HF 缓存」
- `freeze_modules` 只训 DiT 省显存
- 代码/依赖变更后 **新启进程**
- 对照 lmms FLA commit 与本地 `flash-linear-attention_dev` HEAD

---

## 17. 检索关键词

`SpikeBrain` `SPB2` `starvla` `flash_attn` `flash-linear-attention` `fla` `MoBA` `GDN` `liger-kernel` `TRITON_CACHE_DIR` `run_libero_train` `QwenGR00T` `LIBERO` `DeepSpeed` `HybridCache` `ensure_gdn_fp32` `triton_autotune` `starVLA-starVLA` `apply_config_compat` `diffusion_model_cfg` `repeated_diffusion_steps` `action_forward_chunk_size` `dit_loss_max_retries` `bs32` `NaN`

---

## 18. GR00T 对齐 `trash/starVLA-starVLA`（2026-05-23）

### 18.1 目标

在 **仅将 VLM 换为 SpikeBrain-2.0-VL** 的前提下，使训练配置与代码路径尽可能贴近上游仓库  
`trash/starVLA-starVLA` 的 **QwenGR00T**（Flow-matching + DiT action head）。

**对齐基准**（勿与 `starVLA-starVLA_dev` 混用）：

| 用途 | 上游路径 |
|------|----------|
| GR00T 结构默认值 | `trash/starVLA-starVLA/starVLA/config/training/starvla_cotrain_libero.yaml` |
| LIBERO 跑法（冻 VLM、CoT、`libero_all`） | `trash/starVLA-starVLA/examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |

### 18.2 改动文件一览

| 文件 | 改动摘要 |
|------|----------|
| `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | 见 §18.3 |
| `starVLA/model/framework/share_tools.py` | `merge_framework_config`、`apply_config_compat`；checkpoint 加载时 compat |
| `starVLA/training/train_starvla.py` | `build_framework` 前调用 `apply_config_compat` |
| `starVLA/model/framework/QwenGR00T.py` | `QwenGR00TDefaultConfig`、merge 默认、horizon/mask/repeated 逻辑 |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | `predict_action(..., encoder_attention_mask=...)` |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | LIBERO `state_indices` 恢复为 `range(-16, 0)` |

### 18.3 YAML 变更（`starvla_cotrain_libero.yaml`）

| 项 | 改前（典型） | 改后 |
|----|-------------|------|
| `version_id` | 无 | `"0.21"` |
| `diffusion_model_cfg` | 缺失（DiT 默认 12 层） | **16 层**，`output_dim: 1024`，dropout 等与上游模板一致 |
| `noise_beta_*` / `noise_s` | 2.0 / 2.0 / 1.0 | **1.5 / 1.0 / 0.999** |
| `repeated_diffusion_steps` | 仅在 `trainer` | **`framework.action_model: 8`** |
| `action_type` | `abs_qpos` | **`delta_qpos`** |
| `CoT_prompt` | 无 | 与上游 examples 相同的 bbox CoT 串 |
| `learning_rate` | base/action 均 1e-5 | **base 2.5e-5，action 1e-4** |
| `scheduler min_lr` | 5e-7 | **1e-6** |
| FlowRAM 字段 | 有 | **保留**（注释说明：当前仍走 `GR00T_ActionHeader`，未接 FlowRAM 头） |

**刻意未改**（SPB / 显存约束）：

- `base_vlm` → SpikeBrain 本地路径；`ensure_gdn_fp32_params`、`use_liger_kernel`
- `per_device_batch_size: 1`（vla/vlm）
- `gradient_accumulation_steps: 64`（单卡 bs=1 时，每步有效样本数仍约 **1×64=64**，与上游 examples 的 **16×4=64** 同量级）
- `freeze_modules: qwen_vl_interface`

### 18.4 代码行为变更（`QwenGR00T.py`）

| 行为 | 上游 starVLA-starVLA | 当前 SPB（对齐后） |
|------|----------------------|-------------------|
| 配置合并 | `merge_framework_config(QwenGR00TDefaultConfig)` | **同** |
| action 切片 | `action_horizon` | **同**（不再用 `future_action_window_size+1`） |
| `repeated_diffusion_steps` | `framework.action_model` | **同**（`trainer` 仅作后备） |
| DiT cross-attn | 传 `encoder_attention_mask` | **同** |
| VLM 输出 | `hidden_states[-1]` | **`last_hidden_state`**（省显存；冻 VLM 时语义通常等价） |
| 冻 VLM | examples 冻参数，forward 无专门 `no_grad` | **保留** `eval()` + `no_grad()` |
| 数值稳定 | 无 | **保留** `nan_to_num`、predict 路径 FP32 |

`apply_config_compat` 会在训练启动时自动补齐 `future_action_window_size`、`past_action_window_size`、`diffusion_model_cfg.output_dim` 等（与上游 `version_id: 0.21` 一致）。

### 18.5 与上游仍不一致之处（已知、可接受）

| 项 | 说明 |
|----|------|
| VLM backbone | SpikeBrain vs Qwen2.5/3-VL |
| `per_device_batch_size` | 上游 examples 写 16；SPB 单卡/双卡仅能 1 |
| `gradient_accumulation_steps` | 上游 examples 为 4；SPB 为 64（配合 bs=1 维持有效 batch） |
| FlowRAM yaml 字段 | 保留但未接入 FlowRAM 头 |
| `action_type` 在 Python 中 | yaml 已写 `delta_qpos`；数据 pipeline 是否做 delta 变换需与数据集口径核对 |
| 目录结构 | 未迁移到上游 `VLM4A/QwenGR00T.py`，仍在 `model/framework/QwenGR00T.py` |

### 18.6 验证

```bash
cd /9950backfile/zhangjh/wh/starvla_wh/starVLA
export PATH="/9950backfile/zhangjh/.conda/envs/starvla/bin:${PATH}"
python -c "
from omegaconf import OmegaConf
from starVLA.model.framework.share_tools import apply_config_compat
cfg = OmegaConf.load('examples/LIBERO/train_files/starvla_cotrain_libero.yaml')
apply_config_compat(cfg)
assert cfg.framework.action_model.diffusion_model_cfg.num_layers == 16
assert cfg.framework.action_model.repeated_diffusion_steps == 8
print('GR00T yaml compat ok')
"
```

正常开训仍用：`bash examples/LIBERO/train_files/run_libero_train.sh`。

---

## 19. GR00T 对齐复查（2026-05-23）

### 19.1 已修复的问题

| 问题 | 说明 | 修复 |
|------|------|------|
| **DiT `action_horizon` 不一致** | `QwenGR00T` 用 `action_horizon` 切片，但 `GR00T_ActionHeader` 仍用 `future_action_window_size + 1`；若 compat 未补齐会错位 | `GR00T_ActionHeader` 优先读 `config.action_horizon` |
| **cotrain 未 compat** | `train_starvla_cotrain.py` 未调用 `apply_config_compat` | 与 `train_starvla.py` 一致，在 `wrap_config` 前调用 |
| **推理无 `trainer` 段** | `getattr(self.config.trainer, ...)` 在缺 `trainer` 时会崩 | `QwenGR00T._is_vlm_frozen()` 安全判断 |

### 19.2 无逻辑错误 / 已验证项

- `merge_framework_config` **不会**覆盖 SPB 的 `use_liger_kernel`、`ensure_gdn_fp32_params`、`base_vlm` 等额外 yaml 字段。
- `apply_config_compat` 在 `version_id: "0.21"` 时仍会做 horizon / `diffusion_model_cfg` 补全（幂等）。
- `encoder_attention_mask` 在 `GR00T_ActionHeader.forward` / `predict_action` 均已传递。
- LIBERO 默认 **`state=None`**（`datasets.py` 里 `include_state` 未开），`state_indices` 改动**不影响**当前训练；仅在未来 `include_state: true` 时生效。

### 19.3 已知限制（非本次引入 bug）

| 项 | 说明 |
|----|------|
| **`action_type: delta_qpos`** | 仅写在 yaml，**Python 数据管线未读取**；是否真为 delta 取决于 LeRobot 数据本身，与上游 starVLA 相同 |
| **`obs: [image_0, image_1]`** | yaml 字段基本未消费；双图由 `modality.json` + `Libero4in1DataConfig.video_keys` 决定 |
| **FlowRAM yaml 字段** | 仍不生效，动作头为 `GR00T_ActionHeader` |
| **VLM 特征** | 仍用 `last_hidden_state` 而非 `hidden_states[-1]`（省显存；冻 VLM 时一般等价） |

### 19.4 开训前建议

改代码后需**新起训练进程**；旧日志（如 `run_libero_train_20260522_233058.log`）为对齐**前**配置，不能代表本次改动后行为。

---

## 20. bs32 双卡：显存、NaN 与加速尝试（2026-05-23）

### 20.1 当前推荐生产配置（rep6）

双卡 GPU 6/7、DeepSpeed ZeRO-2、**全局 batch=128**：

| 项 | 值 |
|----|-----|
| `per_device_batch_size` | 32 |
| `gradient_accumulation_steps` | 2（由 `target_global_batch_size=128` 自动推算） |
| `repeated_diffusion_steps` | **6** |
| DiT 有效 batch | **192**（32×6） |
| `action_forward_chunk_size` | **64**（3 段） |
| `enable_gradient_checkpointing` | **true** |
| `eval_interval` | 200（`run_libero_train.sh` 覆盖） |
| 稳态速度 | ~**9.2 s/it**（`model_times` ~4.55 s/micro-batch） |
| 稳态显存 | `nvidia-smi` ~**47 GB** / 80 GB |

日志示例：`results/Checkpoints/train_logs/run_libero_train_bs32_rep6_20260523_050602.log`

### 20.2 显存：为什么从 ~57 GB 降到 ~47 GB？

| 运行 | `repeat` | DiT 有效 batch | `peak allocated` | `nvidia-smi` |
|------|----------|----------------|------------------|--------------|
| `044102` / `045410` | **8** | **256** | ~51.9 GB | **~57 GB** |
| `rep6` / `fast` | **6** | **192** | ~42.6 GB | **~47 GB** |

**主因是 `repeated_diffusion_steps: 8 → 6`**，不是 chunk / 关 GC。

训练路径里 VLM 只出 bs=32 的 hidden，随后 **repeat** 再进 DiT（`QwenGR00T.py`）：

```python
last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
```

| repeat | 进 DiT 的 batch | chunk=64 时 DiT forward 次数 |
|--------|-----------------|------------------------------|
| 8 | 256 | 4 |
| 6 | 192 | 3 |

显存随 repeat 上涨的原因：

1. **repeat 后的大张量**（hidden、mask、action 等）规模随 256/192 线性变；
2. **DiT 多 chunk 的计算图**：loss 对多段 chunk 取平均再 backward，有效样本越多，backward 中间结果越多；
3. **PyTorch 缓存池**：新进程会从 ~43–47 GB 重新涨 reserved，不会停在旧 run 的 57 GB。

`chunk=64` 限制的是**单次 DiT forward 的 batch**，但 repeat 仍决定整条 DiT 路径上的**总样本数**。

### 20.3 加速尝试（fast run）：基本无效

**2026-05-23** 在 rep6 基础上又改了：

| 改动 | yaml / 脚本 |
|------|-------------|
| `enable_gradient_checkpointing: false` | yaml |
| `action_forward_chunk_size: 128` | yaml |
| `eval_interval: 200` | `run_libero_train.sh` |

日志：`results/Checkpoints/train_logs/run_libero_train_bs32_fast_20260523_054154.log`

**稳态对比（同 bs32、grad_accum=2、repeat=6）：**

| 指标 | rep6（chunk=64，GC 开） | fast（chunk=128，GC 关） |
|------|-------------------------|-------------------------|
| `model_times` | ~4.54–4.56 s | ~4.52–4.56 s |
| `s/it` | ~9.2–9.4 s | ~9.1–9.2 s |

**结论：没有实质加速**（误差范围内，甚至略慢）。

### 20.4 为什么 fast 改动几乎不动显存/速度？

**关 VLM gradient checkpointing**

- VLM **已冻结**，forward 在 `torch.no_grad()` 里（`QwenGR00T.py`）。
- GC 主要省的是**反向要用的激活**；`no_grad` 下本来就不建计算图。
- 对冻结 VLM，关 GC **几乎不涨显存、也不怎么加速**（与「训练整棵 VLM」时 +8~12 GB 的直觉不同）。

**chunk 64 → 128**

- 192 条：3×64 → 2×（128+64），**少 1 次 DiT launch，但每次算力约翻倍**，DiT 部分大致打平。
- 峰值仍由 **整批 VLM forward（bs=32 × 5.58B）** 主导，chunk 微调被盖住。

**eval_interval 100 → 200**

- 只减少 step 0/200/400… 的 eval 毛刺（`predict_action` + DDIM），对平均 `s/it` 影响很小（约零点几秒量级）。

### 20.5 bs32 NaN 防护（与显存/速度独立）

针对 `repeat=8`、无 chunk 时 rank1 micro-batch 偶发 NaN，已在代码中落地：

| 防护 | 位置 | 说明 |
|------|------|------|
| VLM hidden `nan_to_num` | `QwenGR00T._sanitize_vlm_hidden` | 每步 forward；**仅 bad_count>0 打 WARNING**；Step 日志有 `vlm_hidden_bad_elems` |
| DiT chunk forward | `GR00T_ActionHeader.forward` | 每步；`dit_chunk_size` / `dit_num_chunks` 每 10/100 step 打 metrics |
| DiT 噪声重采样 | `_forward_loss` | `dit_loss_max_retries: 8`；触发时 WARNING |
| skip 非有限 loss | `train_starvla.py` | `skip_non_finite_loss: true` |
| `upcast_attention: true` | yaml → diffusers | **注意**：PyTorch 2.x 默认 `AttnProcessor2_0` **可能不读该 flag**；真正 fp32 matmul 需切回 `AttnProcessor` 或自定义 processor |

rep6 跑 100+ step：`vlm_hidden_bad_elems=0`、`dit_loss_retries=0`、无 NaN。

### 20.6 若还要再提速（值得做的）

| 方向 | 预期 | 代价 |
|------|------|------|
| **离线缓存 VLM hidden** | 最大（去掉每步 5.58B forward） | 改 dataloader + 磁盘 |
| **加 GPU**（4 卡、保持 global 128） |  wall time 可接近减半 | 资源 |
| **`torch.compile`（DiT）** | 10–25%（视环境） | 首步编译、兼容性 |
| **`repeat` 6→4** | 显存与 DiT 时间 ↓ | 训练动力学变化 |
| 关 GC / 调 chunk | **本条路径上收益很小** | 已实测 |

### 20.7 相关日志索引

| 日志 | 配置要点 | 显存 | 速度 |
|------|----------|------|------|
| `run_libero_train_bs32_20260523_044102.log` | repeat=8，早期 | ~57 GB | — |
| `run_libero_train_bs32_20260523_045410.log` | repeat=8，chunk=64 | ~57 GB | — |
| `run_libero_train_bs32_rep6_20260523_050602.log` | **repeat=6，chunk=64，GC 开** | **~47 GB** | **~9.2 s/it** |
| `run_libero_train_bs32_fast_20260523_054154.log` | repeat=6，chunk=128，GC 关 | ~46 GB | ~9.1 s/it（≈无加速） |

**建议**：长期训练用 **rep6 配置**（GC 开、chunk=64）；fast 改动可回滚 yaml，不必为加速保留。
