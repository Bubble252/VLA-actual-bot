#!/usr/bin/env python3
"""终极离线验证：demo 动作流 → 我们的执行链 → 复现 demo 状态轨迹。

为什么这是终极验证
------------------
• 固定指令流回放（replay_execution.py）的局限：录制里的模型指令是 OOD 状态下
  生成的"毒数据"，反事实轨迹不能说明模型行为。
• 而训练 demo 的动作流是干净的、且有标准答案（demo 自己的 state 轨迹）。
  若我们的执行链（rotvec→欧拉转换 + diff-IK + 限幅 + EMA）把 demo 动作
  转出来的关节轨迹 FK 后能贴合 demo state 轨迹，就证明了：
  **部署执行链 + 训练动作契约 全链路一致**（模型除外）。
• 同一 demo 动作流喂旧路径（euler+link6）作为对照——应看到旋转发散。

评分
----
xyz 误差（米）、旋转角误差（rad）、每步"执行旋转 vs 动作意图"误差、
末端 z / rotvec 是否与 demo 一致（能否到达抓取姿态）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EVAL_SRC = Path("/home/casia/project/ARX_X5_clean(7.12eval)/lewm_x5_grasp/src")
if str(_EVAL_SRC) not in sys.path:
    sys.path.insert(0, str(_EVAL_SRC))

from lewm_x5_grasp.eef_to_joint_ik import (  # noqa: E402
    dls_ik,
    differential_ik,
    rotvec_delta_to_euler_increment,
)
from lewm_x5_grasp.transforms import rpy_to_matrix  # noqa: E402
from lewm_x5_grasp.x5_fk import eef_link_fk, full_fk  # noqa: E402

IK_DAMP = 0.02
IK_MAX_DQ = 0.12
MAX_JOINT_DELTA = 0.08
IK_FAIL_DELTA = 0.03
JOINT_EMA = 0.9
HOME_Q = np.array([np.pi / 2, 0.0, 0.0, 0.0, 0.0, 0.0])


def _matrix_to_rpy(R):
    """旋转矩阵 → extrinsic-XYZ rpy（与 x5_fk._matrix_to_rpy 同约定）。"""
    r20 = float(R[2, 0])
    pitch = float(-np.arcsin(np.clip(r20, -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        return 0.0, pitch, float(np.arctan2(-R[0, 1], R[1, 1]))
    return (float(np.arctan2(R[2, 1], R[2, 2])),
            pitch,
            float(np.arctan2(R[1, 0], R[0, 0])))


def _tip_to_flange_target(tip_xyzrpy):
    """tip xyzrpy → link6 法兰 xyzrpy（沿 tool +X 后退 0.145m），供 dls_ik 使用。"""
    out = np.asarray(tip_xyzrpy, dtype=np.float64).copy()
    R = rpy_to_matrix(out[3], out[4], out[5])
    out[:3] = out[:3] - (R @ np.array([0.145, 0.0, 0.0]))
    return out


def _matrix_to_rotvec(R):
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    ang = float(np.arccos(c))
    if ang < 1e-10:
        return np.zeros(3)
    if abs(ang - np.pi) < 1e-4:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        axis = A[:, k] / axis[k] if axis[k] > 0 else axis
        axis = axis / np.linalg.norm(axis)
        return axis * ang
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (ang / (2.0 * np.sin(ang)))


def _rv_to_mat(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _rate_limit(cur, tgt, lim):
    out = np.asarray(cur, dtype=np.float64).copy()
    for j in range(6):
        d = float(np.arctan2(np.sin(tgt[j] - cur[j]), np.cos(tgt[j] - cur[j])))
        out[j] = cur[j] + float(np.clip(d, -lim, lim))
    return out


def replay_demo(state, action, *, semantics, ik_fk, q_init=None):
    """初始化到 demo 起点位形（绝对 IK），把 demo 动作流过一条执行路径。"""
    fk_fn = full_fk if ik_fk == "link6" else eef_link_fk
    if q_init is None:
        flange_target = _tip_to_flange_target(state[0].copy())  # xyz+rotvec → xyzrpy
        # rotvec → rpy（dls_ik 吃 rpy）
        R0 = _rv_to_mat(flange_target[3:6])
        rpy0 = _matrix_to_rpy(R0)
        tgt = np.concatenate([flange_target[:3], rpy0])
        q_init, ikinfo = dls_ik(tgt, HOME_Q, damp=0.05, max_iters=120)
    q = q_init.copy()
    last_cmd_q = q.copy()
    n = len(action)
    tip_hist = np.zeros((n, 6))
    for t in range(n):
        d_xyz = action[t, :3]
        d_rot = action[t, 3:6]
        cur_eef = eef_link_fk(last_cmd_q)  # tip 一体化基（state 源一致）
        if semantics == "rotvec":
            rot_ik = rotvec_delta_to_euler_increment(d_rot, last_cmd_q, rot_scale=1.0)
        else:
            rot_ik = d_rot
        cart = np.concatenate([d_xyz, rot_ik])
        q_ik, info = differential_ik(
            last_cmd_q, cart, damp=IK_DAMP, max_dq=IK_MAX_DQ, fk_fn=fk_fn
        )
        lim = MAX_JOINT_DELTA if info.get("ok", True) else min(MAX_JOINT_DELTA, IK_FAIL_DELTA)
        q_lim = _rate_limit(last_cmd_q, q_ik, lim)
        q_new = last_cmd_q * (1.0 - JOINT_EMA) + q_lim * JOINT_EMA
        last_cmd_q = q_new.copy()
        tip_hist[t] = eef_link_fk(last_cmd_q)
    return tip_hist


def score(state, tip_hist, label):
    """对比 sim tip xyzrpy 与 demo state（前 6 维=xyz+rotvec? → xyz+rotvec 转 rpy 差）。"""
    n = len(tip_hist)
    dem_xyz = np.asarray(state[:, :3], dtype=np.float64)
    dem_rv = np.asarray(state[:, 3:6], dtype=np.float64)
    sim_xyz = tip_hist[:, :3]
    # sim 是 rpy → 转 rotvec 比
    sim_rv = np.zeros((n, 3))
    for i in range(n):
        sim_rv[i] = _matrix_to_rotvec(rpy_to_matrix(*tip_hist[i, 3:6]))
    xyz_err = np.linalg.norm(sim_xyz - dem_xyz, axis=1)
    rot_err = np.zeros(n)
    for i in range(n):
        rot_err[i] = float(np.linalg.norm(_matrix_to_rotvec(
            rpy_to_matrix(*tip_hist[i, 3:6]) @ _rv_to_mat(dem_rv[i]).T)))
    warm = 100  # 跳过 warmup（init IK 残差+EMA 渐入）
    w = slice(warm, n) if n > warm else slice(0, n)
    print(f"  [{label}]")
    print(f"    xyz 误差 (m):  mean={xyz_err.mean():.4f}  p95={np.percentile(xyz_err,95):.4f}  max={xyz_err.max():.4f}   "
          f"[washout后 mean={xyz_err[w].mean():.4f} max={xyz_err[w].max():.4f}]")
    print(f"    旋转误差 (rad): mean={rot_err.mean():.4f}  p95={np.percentile(rot_err,95):.4f}  max={rot_err.max():.4f}   "
          f"[washout后 mean={rot_err[w].mean():.4f} max={rot_err[w].max():.4f}]")
    print(f"    终点: sim z={tip_hist[-1,2]:.4f} (demo {dem_xyz[-1,2]:.4f})  "
          f"sim rotvec[1]={sim_rv[-1,1]:+.3f} (demo {dem_rv[-1,1]:+.3f})  "
          f"sim rotvec[0]={sim_rv[-1,0]:+.3f} (demo {dem_rv[-1,0]:+.3f})")
    return xyz_err, rot_err


def main():
    demo_dir = Path(__file__).resolve().parent / "demo_parquet"
    files = sorted(demo_dir.glob("episode_*.parquet"))
    for f in files:
        df = pd.read_parquet(f)
        state = np.asarray(df["observation.state"].tolist(), dtype=np.float64)
        action = np.asarray(df["action"].tolist(), dtype=np.float64)
        print(f"\n########## {f.name}: {len(state)} 步 ##########")
        print(f"  demo 起点: xyz={np.round(state[0,:3],4)}  rotvec={np.round(state[0,3:6],4)}")
        print(f"  FK(home) tip: {np.round(eef_link_fk(HOME_Q)[:3],4)}  rpy={np.round(eef_link_fk(HOME_Q)[3:6],4)}")
        print(f"  demo 动作幅值均值: dxyz={np.linalg.norm(action[:,:3],axis=1).mean()*1000:.2f}mm/步  "
              f"|drot|={np.linalg.norm(action[:,3:6],axis=1).mean()*1000:.2f}mrad/步")

        # 绝对 IK 求起点关节（两个路径共用同一初值，公平对照）
        fl_t = _tip_to_flange_target(state[0].copy())
        R0 = _rv_to_mat(fl_t[3:6])
        tgt = np.concatenate([fl_t[:3], _matrix_to_rpy(R0)])
        q_init, _ = dls_ik(tgt, HOME_Q, damp=0.05, max_iters=120)
        tip0 = eef_link_fk(q_init)
        rv0 = _matrix_to_rotvec(rpy_to_matrix(*tip0[3:6]))
        print(f"  [init] IK 起点: tip xyz 差 {np.linalg.norm(tip0[:3]-state[0,:3])*1000:.1f}mm  "
              f"rotvec 差 {np.linalg.norm(rv0-state[0,3:6])*1000:.1f}mrad")

        tip_new = replay_demo(state, action, semantics="rotvec", ik_fk="eef_link", q_init=q_init)
        score(state, tip_new, "NEW (rotvec+eef_link)")
        tip_old = replay_demo(state, action, semantics="euler", ik_fk="link6", q_init=q_init)
        score(state, tip_old, "OLD (euler+link6) 对照——应发散")


if __name__ == "__main__":
    main()