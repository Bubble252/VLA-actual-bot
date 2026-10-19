"""Gripper mapping for arx_x5_ljn_pick_stage2_90k deployment.

与 0813 的核心区别：夹爪是**连续** openpi_closedness，不是 binary。
模型输出 action 第 7 维 ∈ [0,1]（OpenPI 约定：1=完全闭合，0=完全张开）。
部署时映射回 ARX 原生 qpos 再下发。

来源：迁移包 metadata/action_contract.json
  qpos_open   = 4.491874016821384   （示教时夹爪张开读数 ≈ 4.5）
  qpos_closed = -0.06599569314857945（示教时夹爪闭合读数 ≈ 0.0 / 负值）
"""
import numpy as np

# 来自 action_contract.json / dataset gripper_map.json
QPOS_OPEN = 4.491874016821384
QPOS_CLOSED = -0.06599569314857945


def closedness_to_qpos(closedness: float) -> float:
    """OpenPI closedness ∈ [0,1] (0=开, 1=闭) → ARX 原生 qpos.

    qpos = qpos_open + closedness * (qpos_closed - qpos_open)
      closedness=0 → qpos = qpos_open (4.49, 开) ✅
      closedness=1 → qpos = qpos_closed (-0.066, 闭) ✅

    OpenPI 约定 closedness=1 表示"闭合程度满"=夹爪闭合。
    ARX qpos: qpos_open(4.49) > qpos_closed(-0.066)，即 qpos 大=开。
    故 closedness=1(闭) → qpos 应= qpos_closed(小)。
    """
    c = float(np.clip(closedness, 0.0, 1.0))
    # closedness=0(开) → qpos_open；closedness=1(闭) → qpos_closed
    return QPOS_OPEN + c * (QPOS_CLOSED - QPOS_OPEN)


def qpos_to_closedness(qpos: float) -> float:
    """反向映射（state 侧用）：ARX qpos → OpenPI closedness。"""
    if QPOS_OPEN == QPOS_CLOSED:
        return 0.0
    c = (qpos - QPOS_OPEN) / (QPOS_CLOSED - QPOS_OPEN)
    return float(np.clip(c, 0.0, 1.0))


# state 侧：state.pad（原始 qpos）→ state.gripper（openpi closedness）
state_gripper_to_openpi = qpos_to_closedness


if __name__ == "__main__":
    # 自检
    print(f"qpos_open={QPOS_OPEN}, qpos_closed={QPOS_CLOSED}")
    for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q = closedness_to_qpos(c)
        print(f"  closedness={c:.2f} → qpos={q:.4f}  (back={qpos_to_closedness(q):.3f})")
    # 验证：closedness=0 → qpos_open(4.49 开)；closedness=1 → qpos_closed(-0.066 闭)
    assert abs(closedness_to_qpos(0.0) - QPOS_OPEN) < 1e-6
    assert abs(closedness_to_qpos(1.0) - QPOS_CLOSED) < 1e-6
    print("OK: closedness=0 → open, closedness=1 → closed")
