"""Convert EEF xyzrpy targets to joint angles via pure-Python FK + damped least-squares IK.

Avoids ARX SDK mode=4 (known singularity / branch-jump issues on real X5).
Use with RobotCmd.mode=5 (POSITION_CONTROL).

IMPORTANT: SDK /arm_status end_pos often does NOT match our URDF FK absolute pose.
Prefer differential IK (default): map Δxyzrpy → Δq at current joints, without
solving absolute pose. Absolute IK is kept only for offline experiments.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .transforms import rpy_to_matrix
from .x5_fk import full_fk


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rotvec_delta_to_euler_increment(
    rot_delta: Sequence[float],
    cur_q: Sequence[float],
    *,
    rot_scale: float = 1.0,
) -> np.ndarray:
    """Convert a world-frame rotation-vector increment into the extrinsic-XYZ
    Euler rpy increment understood by this module's Jacobian.

    For rotvec-diff datasets (e.g. arx_x5_real_ljn), training actions are
    element-wise diffs of the state rotation vector, i.e. WORLD-frame rotation
    increments. The differential-IK Jacobian rows are d(extrinsic-XYZ rpy)/dq;
    adding a rotvec delta element-wise to rpy is WRONG at yaw≈±90° (an input
    roll increment executes as a world-pitch rotation → end-effector pose drift
    loop). Correct semantics: R_new = Rv(delta) @ R_cur, then the wrapped
    euler difference euler_xyz(R_new) - euler_xyz(R_cur).
    """
    d = np.asarray(rot_delta, dtype=np.float64).reshape(3) * float(rot_scale)
    q6 = np.asarray(cur_q, dtype=np.float64).reshape(-1)[:6]
    rpy_cur = full_fk(q6)[3:6]
    ang = float(np.linalg.norm(d))
    if ang < 1e-12:
        return np.zeros(3, dtype=np.float64)
    k = d / ang
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    Rv = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)
    R_cur = rpy_to_matrix(float(rpy_cur[0]), float(rpy_cur[1]), float(rpy_cur[2]))
    R_new = Rv @ R_cur
    # extrinsic-XYZ extraction (same convention as x5_fk._matrix_to_rpy)
    r20 = float(R_new[2, 0])
    pitch = float(-np.arcsin(np.clip(r20, -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-R_new[0, 1], R_new[1, 1]))
    else:
        roll = float(np.arctan2(R_new[2, 1], R_new[2, 2]))
        yaw = float(np.arctan2(R_new[1, 0], R_new[0, 0]))
    inc = np.array([roll, pitch, yaw], dtype=np.float64) - rpy_cur
    return _wrap_pi(inc)


def _pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    err[3:6] = _wrap_pi(err[3:6])
    return err


def numerical_jacobian(q: Sequence[float], step: float = 1e-4, fk_fn=None) -> np.ndarray:
    """6x6 geometric Jacobian of ``fk_fn`` (default full_fk=link6) at q (finite difference).

    Pass ``fk_fn=eef_link_fk`` for the tip (link6+0.145m along +X) Jacobian.
    """
    q0 = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    fk = full_fk if fk_fn is None else fk_fn
    cur = fk(q0)
    jac = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        qp = q0.copy()
        qp[i] += step
        jac[:, i] = _pose_error(cur, fk(qp)) / step
        jac[3:6, i] = _wrap_pi(jac[3:6, i] * step) / step
    return jac


def dls_ik(
    target_xyzrpy: Sequence[float],
    seed_q: Sequence[float],
    *,
    max_iters: int = 40,
    tol_pos: float = 1e-3,
    tol_rot: float = 2e-2,
    damp: float = 2e-2,
    step: float = 1e-4,
    max_dq: float = 0.12,
) -> Tuple[np.ndarray, dict]:
    """Solve 6-DoF absolute IK. Returns (q6, info). Prefer differential_ik on real robot."""
    q0 = np.asarray(seed_q, dtype=np.float64).reshape(-1)[:6].copy()
    q = q0.copy()
    target = np.asarray(target_xyzrpy, dtype=np.float64).reshape(6)
    last_err = np.ones(6)
    max_cond = 1.0
    for it in range(max_iters):
        cur = full_fk(q)
        err = _pose_error(cur, target)
        last_err = err
        if float(np.linalg.norm(err[:3])) < tol_pos and float(np.linalg.norm(err[3:6])) < tol_rot:
            return q, {
                "iters": it,
                "err": err,
                "ok": True,
                "cond": max_cond,
                "jump": float(np.linalg.norm(_wrap_pi(q - q0))),
            }

        jac = numerical_jacobian(q, step=step)
        try:
            singular = np.linalg.svd(jac, compute_uv=False)
            cond = float(singular[0] / max(singular[-1], 1e-12))
        except np.linalg.LinAlgError:
            cond = 1e6
        max_cond = max(max_cond, cond)
        damp_eff = damp * (1.0 + min(cond / 200.0, 20.0))

        jtj = jac.T @ jac
        dq = np.linalg.solve(jtj + (damp_eff**2) * np.eye(6), jac.T @ err)
        n = float(np.linalg.norm(dq))
        if n > max_dq:
            dq *= max_dq / n
        q = q + dq

    ok = bool(
        float(np.linalg.norm(last_err[:3])) < 5e-3
        and float(np.linalg.norm(last_err[3:6])) < 5e-2
    )
    return q, {
        "iters": max_iters,
        "err": last_err,
        "ok": ok,
        "cond": max_cond,
        "jump": float(np.linalg.norm(_wrap_pi(q - q0))),
    }


def differential_ik(
    cur_q: Sequence[float],
    cart_delta6: Sequence[float],
    *,
    damp: float = 2e-2,
    max_dq: float = 0.15,
    step: float = 1e-4,
    fk_fn=None,
) -> Tuple[np.ndarray, dict]:
    """Map a small cartesian Δ (xyzrpy) to Δq at cur_q. No absolute pose needed."""
    q = np.asarray(cur_q, dtype=np.float64).reshape(-1)[:6].copy()
    dx = np.asarray(cart_delta6, dtype=np.float64).reshape(6)
    jac = numerical_jacobian(q, step=step, fk_fn=fk_fn)
    try:
        singular = np.linalg.svd(jac, compute_uv=False)
        cond = float(singular[0] / max(singular[-1], 1e-12))
    except np.linalg.LinAlgError:
        cond = 1e6
    damp_eff = damp * (1.0 + min(cond / 200.0, 20.0))
    jtj = jac.T @ jac
    dq = np.linalg.solve(jtj + (damp_eff**2) * np.eye(6), jac.T @ dx)
    n = float(np.linalg.norm(dq))
    clipped = False
    if n > max_dq:
        dq *= max_dq / n
        clipped = True
    q_new = q + dq
    # Local residual in cartesian after one step (FK-relative, not SDK)
    pred = jac @ dq
    resid = dx - pred
    ok = bool(float(np.linalg.norm(resid[:3])) < 0.02 and float(np.linalg.norm(resid[3:6])) < 0.15)
    return q_new, {
        "ok": ok,
        "cond": cond,
        "jump": float(np.linalg.norm(dq)),
        "clipped": clipped,
        "err": resid,
        "mode": "diff",
    }


def eef_delta_to_joint_cmd(
    *,
    cur_eef: Sequence[float],
    cur_q: Sequence[float],
    world_delta: Sequence[float],
    rot_delta: Sequence[float],
    pos_scale: float,
    rot_scale: float,
    seed_q: Optional[Sequence[float]] = None,
    use_differential: bool = True,
    damp: Optional[float] = None,
    max_dq: Optional[float] = None,
    fk_fn=None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Integrate OSC-style deltas → joint command.

    Default ``use_differential=True``: Δxyzrpy → Δq via Jacobian at cur_q.
    Absolute IK (False) needs FK to match SDK end_pos; on real X5 it usually does not.

    ``damp`` / ``max_dq`` default to ``differential_ik`` built-ins; callers should
    pass values from ``x5_play_params.yaml`` for replay-v / eval parity.

    Returns (q6, target_eef6, ik_info).
    """
    cur_eef_a = np.asarray(cur_eef, dtype=np.float64).reshape(6)
    d_xyz = np.asarray(world_delta, dtype=np.float64).reshape(3) * float(pos_scale)
    d_rpy = np.asarray(rot_delta, dtype=np.float64).reshape(3) * float(rot_scale)
    target = cur_eef_a.copy()
    target[:3] = cur_eef_a[:3] + d_xyz
    target[3:6] = cur_eef_a[3:6] + d_rpy
    seed = cur_q if seed_q is None else seed_q

    if use_differential:
        cart = np.concatenate([d_xyz, d_rpy])
        ik_kwargs = {}
        if damp is not None:
            ik_kwargs["damp"] = float(damp)
        if max_dq is not None:
            ik_kwargs["max_dq"] = float(max_dq)
        if fk_fn is not None:
            ik_kwargs["fk_fn"] = fk_fn
        q, info = differential_ik(seed, cart, **ik_kwargs)
        return q, target, info

    q, info = dls_ik(target, seed)
    info["mode"] = "abs"
    return q, target, info
