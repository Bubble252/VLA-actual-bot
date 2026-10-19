# RTC（轨 B / ΔEEF）接入本仓

把 Physical Intelligence / LeRobot 风格的 **Real-Time Chunking** 接到现网 **StarVLA ΔEEF** 链路。  
默认关闭；需异步 chunk 执行（`--rtc-mode track_b`）才会生效。

## 文档

| 文件 | 内容 |
|------|------|
| `DOC/01_项目背景.md` | 为何做、两轨边界 |
| `DOC/02_技术栈.md` | 组件与约束 |
| `DOC/03_执行计划.md` | P0–P5 勾选进度 |
| `DOC/04_P1挂载点清单.md` | Euler 挂载点 / 维掩码 / 归一化 |
| `DOC/05_P2协议_Client_Server.md` | websocket 字段约定 |
| `DOC/06_P3远端POC.md` | 远端补丁与回滚 |
| `DOC/07_P4客户端.md` | 本机 `track_b` client |
| `DOC/08_P5联调记录.md` | 5694 重启与烟雾结果 |
| `DOC/09_频率与RTC配比.md` | 20/30 Hz 数据 vs 部署 5 Hz：方案 A–E；**§4.1 锁定 20 Hz 时 H 怎么定** |
| `DOC/03_执行计划.md` → P6 | 20 Hz + 加长 chunk 重训后开 RTC（计划 H=24 / 备选 30） |

## 怎么开 / 怎么关

**关（默认，现网兼容）**

```bash
python3 lewm_x5_grasp/scripts/eval_starvla_libero_x5.py --port 5694 --rtc-mode off ...
# 或不传 --rtc-mode
```

**开（异步 A + server guidance）**

1. 远端 5694 已加载 P3 补丁（改代码后需重启 `server_policy.py`）。  
2. 本机：

```bash
python3 lewm_x5_grasp/scripts/eval_starvla_libero_x5.py \
  --host 127.0.0.1 --port 5694 \
  --rtc-mode track_b --dry-run --max-steps 40
```

日志里关注：`rtc={..., srv: True}`（server `data.rtc.applied`）。

## 范围

- 本期：ΔEEF（5694/5695，`unnorm_key=franka`）  
- 夹爪：旁路，不进 guidance  
- 不做：5794 关节口、MuJoCo 硬验收  

## 代码位置

- 本仓 client：`RTC/rtc_arx/`  
- 本仓远端补丁镜像：`RTC/starvla_patch/`  
- 远端运行树：`/root/code/Starvla`（GPU 机）

## 回滚

- Client：`--rtc-mode off`  
- Server：不传 `rtc_enabled` 即不走 guidance；或从 `/root/code/Starvla/.rtc_p3_backup_*` 恢复文件后重启端口  
