# 06 · P3 远端 POC 落地说明

> 日期：2026-08-08  
> 对应计划：`03_执行计划.md` → P3  
> 协议：`05_P2协议_Client_Server.md`

---

## 1. 已改动的远端文件

根目录：`/root/code/Starvla`

| 路径 | 改动 |
|------|------|
| `starVLA/model/modules/rtc/` | **新增** guidance 模块（EVO1 时间映射 + 维掩码） |
| `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | Euler 环内可选 RTC guidance |
| `starVLA/model/framework/QwenGR00T.py` | 透传 `rtc_*` / leftover 到 action head |
| `deployment/model_server/policy_norm_processor.py` | 新增 `apply_actions`（物理→normalized） |
| `deployment/model_server/policy_wrapper.py` | 解析 RTC 请求字段；默认关；失败回退 |

备份目录示例：`/root/code/Starvla/.rtc_p3_backup_YYYYMMDD_HHMMSS/`

本仓镜像与重放脚本：`RTC/starvla_patch/`

---

## 2. 默认行为（安全）

- 旧 client **不传** `rtc_*` → 与补丁前相同（不进 guidance）。  
- `rtc_enabled=false` / 缺省 → 忽略 leftover。  
- `rtc_enabled=true` 但 prefix 空/失败 → **降级无 RTC**，响应 `data.rtc.fell_back=true`。  
- 夹爪维强制不 guidance：`[1,1,1,1,1,1,0]`。

---

## 3. 启用方式（需重启 server 进程后才加载新代码）

当前内存里的 5694/5695/5794 **不会自动热更新**。要验证 P3，需重启目标端口的 `server_policy.py`（建议先只动 5694 测试口，或另开端口）。

请求示例：

```python
{
    "examples": [example],
    "unnorm_key": "franka",
    "rtc_enabled": True,
    "rtc_prefix": leftover_physical,  # (L,7) 或 (L,6)，unnorm 后、scale 前
    "rtc_prefix_space": "physical",
    "inference_delay": 2,
    "execution_horizon": 4,
}
```

响应额外字段（可选读）：

```python
data["rtc"] = {
    "applied": True/False,
    "fell_back": ...,
    "prefix_len": ...,
    ...
}
```

---

## 4. 已做烟雾验证

| 检查 | 结果 |
|------|------|
| 补丁文件 Python 语法 | OK |
| `guide_starvla_velocity` 夹爪维不变 | OK（diff=0） |
| `apply_actions` ↔ `unapply_actions` roundtrip（franka / 5694 ckpt） | OK（~1e-8） |
| 整模 GPU 开关对比（P5） | OK；并修复 `inference_mode` 阻断（见 `08`） |

---

## 5. 回滚

```bash
# 查看备份
ls -d /root/code/Starvla/.rtc_p3_backup_*

# 示例：从某次备份恢复四个文件 + 删除 rtc 包（按需）
BAK=/root/code/Starvla/.rtc_p3_backup_XXXXXXXX
cp "$BAK"/policy_norm_processor.py /root/code/Starvla/deployment/model_server/
cp "$BAK"/policy_wrapper.py /root/code/Starvla/deployment/model_server/
cp "$BAK"/GR00T_ActionHeader.py /root/code/Starvla/starVLA/model/modules/action_model/
cp "$BAK"/QwenGR00T.py /root/code/Starvla/starVLA/model/framework/
# 重启对应 server 进程
```

或始终不传 `rtc_enabled`，即使已打补丁也不走 RTC。
