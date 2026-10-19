#!/usr/bin/env python3
"""执行层影子回放 — 不用真机验证 rot_delta 语义修复。

原理
----
bug 期录制（rollouts/*/traj.jsonl）包含每步:
  joint_pos[7]   机器人实际关节角
  eef_pos6[6]    SDK 上报法兰位姿
  world_delta    模型 raw Δxyz（frame_align=none, 未剪裁）
  rot_delta      模型 raw 旋转增量
  state_vec[8]   发给模型的 state（含 rotvec）

把同一串模型输出分别喂给两条执行路径:
  OLD(euler,link6) = 0813 遗产: rot_delta 逐元素加 extrinsic rpy + link6 Jacobian
  NEW(rotvec,eef_link)     = 修复: Rv(δ)@R_cur 转欧拉增量 + eef_link(tip) Jacobian

OLD 路径的仿真应能**复现**录制中的 rotvec 漂移（sim 与真机一致的验证）；
NEW 路径是反事实推演，rotvec 应保持在训练分布内、执行旋转 = 模型意图。

用法
----
python3 replay_execution.py [rollout_dir ...]     # 默认回放全部 rollouts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_EVAL_SRC = Path("/home/casia/project/ARX_X5_clean(7.12eval)/lewm_x5_grasp/src")
if str(_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_EVAL_SRC))

from lewm_x5_grasp.eef_to_joint_ik import (  # noqa: E402
    differential_ik,
    rotvec_delta_to_euler_increment,
)
from lewm_x5_grasp.transforms import rpy_to_matrix  # noqa: E402
from lewm_x5_grasp.x5_fk import eef_link_fk, full_fk  # noqa: E402

# ===== client 参数复刻（x5_play_params.yaml + client 默认，见疑点清单 §3）=====
POS_SCALE = 1.0
ROT_SCALE = 1.0
IK_DAMP = 0.02
IK_MAX_DQ = 0.12
MAX_JOINT_DELTA = 0.08
IK_FAIL_DELTA = 0.03
JOINT_EMA = 0.9
WS_MIN = np.array([-0.3, 0.05, -0.3])
WS_MAX = np.array([0.3, 0.55, 0.7])
CONTROL_HZ = 20.0
SOFT_ENTRY_STEPS = int(1.0 * CONTROL_HZ)  # eef_soft_entry_sec=1.0


# ---- 数学工具（纯 numpy，避开系统 scipy ABI 问题） ----
def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _smoothstep(alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _matrix_to_rotvec(R):
    """旋转矩阵 → 旋转向量（轴角），与 scipy R.as_rotvec() 一致。"""
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-10:
        return np.zeros(3)
    if abs(ang - np.pi) < 1e-4:
        # 接近 π：用对称部分推轴
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 0:
            axis = A[:, k] / axis[k]
        axis = axis / np.linalg.norm(axis)
        return axis * ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (ang / (2.0 * np.sin(ang)))


def _rate_limit_joints(cur, tgt, max_delta_rad):
    out = np.asarray(cur, dtype=np.float64).copy()
    for j in range(6):
        d = float(np.arctan2(np.sin(tgt[j] - cur[j]), np.cos(tgt[j] - cur[j])))
        d = float(np.clip(d, -max_delta_rad, max_delta_rad))
        out[j] = cur[j] + d
    return out


def _ema_joints(prev, new, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(prev) * (1.0 - a) + np.asarray(new) * a


def _clip_jump_dq(q_seed, cart_delta6, fk_fn, damp, max_dq):
    """走 client 同一条管辖链: differential_ik 一次 + 限幅。"""
    q_new, info = differential_ik(q_seed, cart_delta6, damp=damp, max_dq=max_dq, fk_fn=fk_fn)
    return q_new, info


def replay_one(rows, *, semantics, ik_fk, strict_recorded):
    """回放一条轨迹。

    semantics: 'euler' | 'rotvec';  ik_fk: 'link6' | 'eef_link'
    strict_recorded=True: 每步用录制的真机关节/SDK 做当前状态（OLD 路径仿真验证用）
    strict_recorded=False: 理想跟踪自演化（NEW 路径反事实用）
    返回 per-step 记录 dict list。
    """
    fk_fn = full_fk if ik_fk == "link6" else eef_link_fk
    q_rec = [np.asarray(r["joint_pos"][:6], dtype=np.float64) for r in rows]
    sdk_rec = [np.asarray(r["eef_pos6"][:6], dtype=np.float64) for r in rows]
    deltas = [np.asarray(r["world_delta"][:3], dtype=np.float64) for r in rows]
    rots = [np.asarray(r["rot_delta"][:3], dtype=np.float64) for r in rows]

    last_cmd_q = q_rec[0].copy()
    entry_q0 = q_rec[0].copy()
    q_sim = q_rec[0].copy()          # 理想机器人状态
    out = []
    n_clip = 0
    for t in range(len(rows)):
        # 当前状态源
        q_act = q_rec[t] if strict_recorded else q_sim
        cur_eef = eef_link_fk(q_act)  # tip xyzrpy（state 源, train-aligned）
        # SDK 法兰帧：OLD 严格用录制；NEW 用 full_fk 法兰近似（clip 基本不触发，见输出）
        cur_eef_sdk = sdk_rec[t] if strict_recorded else np.concatenate([full_fk(q_act)[:3], cur_eef[3:6]])

        # ---- Δxyz 剪裁（client 逐帧复刻）----
        tentative = cur_eef.copy()
        tentative[:3] = cur_eef[:3] + deltas[t] * POS_SCALE
        flange = tentative[:3] - (cur_eef[:3] - cur_eef_sdk[:3])
        flange_c = np.clip(flange, WS_MIN, WS_MAX)
        if np.any(flange_c != flange):
            n_clip += 1
        tentative[:3] = flange_c + (cur_eef[:3] - cur_eef_sdk[:3])
        clipped_delta = tentative[:3] - cur_eef[:3]

        # ---- 旋转语义 ----
        if semantics == "rotvec":
            rot_ik = rotvec_delta_to_euler_increment(rots[t], last_cmd_q, rot_scale=ROT_SCALE)
        else:
            rot_ik = rots[t] * ROT_SCALE
        cart = np.concatenate([clipped_delta, rot_ik])
        q_ik, info = _clip_jump_dq(last_cmd_q, cart, fk_fn, IK_DAMP, IK_MAX_DQ)
        ik_ok = bool(info.get("ok", False))
        lim = MAX_JOINT_DELTA if ik_ok else min(MAX_JOINT_DELTA, IK_FAIL_DELTA)
        q_lim = _rate_limit_joints(last_cmd_q, q_ik, lim)
        q6 = _ema_joints(last_cmd_q, q_lim, JOINT_EMA)
        if t < SOFT_ENTRY_STEPS:
            q6 = _ema_joints(entry_q0, q6, _smoothstep((t + 1) / SOFT_ENTRY_STEPS))
        last_cmd_q = q6.copy()
        q_sim = q6.copy()  # 理想跟踪假设

        # ---- 指标 ----
        tip = eef_link_fk(q_sim)
        R_sim = rpy_to_matrix(*tip[3:6])
        R_prev = rpy_to_matrix(*(eef_link_fk(q_act)[3:6]))
        R_exec = R_sim @ R_prev.T
        R_intend = rpy_to_matrix(1.0, 0.0, 0.0)  # 占位，下面直接构造
        ang = float(np.linalg.norm(rots[t]))
        if ang > 1e-12:  # Rv(rots[t])
            k = rots[t] / ang
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R_intend = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        err_rot = _matrix_to_rotvec(R_exec @ R_intend.T)
        e_sim = eef_link_fk(q_sim)
        out.append({
            "t": t,
            "q": q_sim.copy(),
            "tip": tip.copy(),
            "state_rotvec": _matrix_to_rotvec(R_sim),
            "rot_intent_err": float(np.linalg.norm(err_rot)),
            "ik_ok": ik_ok,
            "J5": float(q_sim[4]),
            "z": float(tip[2]),
        })
    return out, n_clip


def summarize(rows, sim, label, recorded_state=None):
    n = len(sim)
    rv = np.array([s["state_rotvec"] for s in sim])
    z = np.array([s["z"] for s in sim])
    J5 = np.array([s["J5"] for s in sim])
    err = np.array([s["rot_intent_err"] for s in sim])
    t = np.arange(n)
    slopes = []
    for c in range(3):
        A = np.vstack([t, np.ones(n)]).T
        slopes.append(float(np.linalg.lstsq(A, rv[:, c], rcond=None)[0][0]))

    fidelity = ""
    if recorded_state is not None:
        rs = np.array([r["state_vec"][3:6] for r in rows], dtype=np.float64)
        sim_vs_rec = float(np.abs(rv - rs).max())
        fidelity = (
            f"\n  [sim-真机一致性] rotvec 最大偏差 vs 录制: {sim_vs_rec:.4f} rad "
            f"(录制漂移斜率 rotvec[1]: 见下表)"
        )
    print(f"\n===== {label} ({n} 步) =====")
    print(fidelity)
    print(f"  rotvec 起点: [{rv[0,0]:+.3f},{rv[0,1]:+.3f},{rv[0,2]:+.3f}]  终点: [{rv[-1,0]:+.3f},{rv[-1,1]:+.3f},{rv[-1,2]:+.3f}]")
    print(f"  rotvec 斜率 (mrad/步): [{slopes[0]*1000:+.2f}, {slopes[1]*1000:+.2f}, {slopes[2]*1000:+.2f}]")
    print(f"  |rotvec[1]| 最大: {np.abs(rv[:,1]).max():.3f} rad  超出训练分布(±0.3)比例: {np.mean(np.abs(rv[:,1])>0.3)*100:.1f}%")
    print(f"  J5 极值: [{J5.min():+.2f}, {J5.max():+.2f}] rad  (录制建议 |J5|<~1.5 保持分布)")
    print(f"  z 起→终: {z[0]:.4f} → {z[-1]:.4f} m")
    print(f"  执行旋转 vs 模型意图 每步误差: mean={err.mean()*1000:.3f} max={err.max()*1000:.3f} mrad")
    return {"rv": rv, "z": z, "J5": J5, "err": err, "slopes": slopes}


def main():
    base = Path(__file__).resolve().parent / "rollouts"
    dirs = sorted(base.iterdir()) if len(sys.argv) < 2 else [Path(a) for a in sys.argv[1:]]

    for d in dirs:
        traj = d / "traj.jsonl"
        if not traj.is_file():
            print(f"[skip] {d}")
            continue
        rows = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("world_delta") and r.get("joint_pos")]
        hz_real = 1000.0 / np.median(np.diff([r["wall_time"] for r in rows])) if len(rows) > 2 else CONTROL_HZ
        print(f"\n########## {d.name}: {len(rows)} 步, 步进中位耗时 ≈{hz_real:.1f} Hz ##########")

        # ---- 训练分布基线（demo 统计, 来自此前 parquet 分析）----
        print("  [基线] 训练 demo: rotvec[1] ∈ 约±0.3 rad; 抓取 z≈0.106; grip_raw 抓取时 ~0.37")

        sim_old, clip_old = replay_one(rows, semantics="euler", ik_fk="link6", strict_recorded=True)
        res_old = summarize(rows, sim_old, "OLD 旧路径 (euler+link6) — 仿真,录制状态驱动")
        np.savez(d / "replay_old.npz", **{k: np.asarray(v) for k, v in res_old.items() if k != "slopes"})

        sim_new, clip_new = replay_one(rows, semantics="rotvec", ik_fk="eef_link", strict_recorded=False)
        res_new = summarize(rows, sim_new, "NEW 新路径 (rotvec+eef_link) — 反事实,理想跟踪")
        np.savez(d / "replay_new.npz", **{k: np.asarray(v) for k, v in res_new.items() if k != "slopes"})
        print(f"  [clip] OLD 触发工作区剪裁 {clip_old} 步; NEW ~{clip_new} 步（NEW 的 SDK 帧是近似）")

        # --- 关键对照 ---
        rec_rv1 = np.array([r["state_vec"][5] for r in rows])  # rotvec[1] = state[5]... 见下
        # state_vec = [xyz, rotvec(3), pad, grip] → rotvec[1] = index 4
        rec_rv1 = np.array([r["state_vec"][4] for r in rows])
        print(f"\n  ► 录制 rotvec[1]: 起点 {rec_rv1[0]:+.3f} → 终点 {rec_rv1[-1]:+.3f} rad")
        print(f"  ► OLD 仿真 rotvec[1]: {res_old['rv'][0,1]:+.3f} → {res_old['rv'][-1,1]:+.3f} rad  (应≈录制, 仿真有效性)")
        print(f"  ► NEW 仿真 rotvec[1]: {res_new['rv'][0,1]:+.3f} → {res_new['rv'][-1,1]:+.3f} rad  (应保持 ±0.3 内)")


if __name__ == "__main__":
    main()