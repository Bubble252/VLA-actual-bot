"""Pose helpers and MuJoCo user_scn axis drawing."""

from __future__ import annotations

import numpy as np
import mujoco


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic XYZ / fixed R = Rz @ Ry @ Rx."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_matrix(float(v[3]), float(v[4]), float(v[5]))
    T[:3, 3] = v[:3]
    return T


def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    rot = T[:3, :3]
    r20 = float(rot[2, 0])
    pitch = float(-np.arcsin(np.clip(r20, -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))
    else:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    return np.array([T[0, 3], T[1, 3], T[2, 3], roll, pitch, yaw], dtype=np.float64)


def body_pose_T(data: mujoco.MjData, body_id: int) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = data.xpos[body_id]
    T[:3, :3] = data.xmat[body_id].reshape(3, 3)
    return T


def _add_arrow(
    scn: mujoco.MjvScene,
    p0: np.ndarray,
    p1: np.ndarray,
    rgba: tuple[float, float, float, float],
    width: float = 0.012,
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
    scn.ngeom += 1


def draw_axes(
    scn: mujoco.MjvScene,
    T: np.ndarray,
    *,
    length: float = 0.10,
    width: float = 0.012,
    alpha: float = 1.0,
) -> None:
    """Draw RGB→XYZ axes at pose T (4x4)."""
    origin = np.asarray(T[:3, 3], dtype=np.float64)
    colors = (
        (1.0, 0.2, 0.15, alpha),
        (0.2, 0.95, 0.25, alpha),
        (0.25, 0.55, 1.0, alpha),
    )
    for i, rgba in enumerate(colors):
        tip = origin + length * T[:3, i]
        _add_arrow(scn, origin, tip, rgba, width=width)


def draw_base_axes(scn: mujoco.MjvScene, *, length: float = 0.14) -> None:
    draw_axes(scn, np.eye(4), length=length, width=0.014)


def _add_capsule(
    scn: mujoco.MjvScene,
    p0: np.ndarray,
    p1: np.ndarray,
    rgba: tuple[float, float, float, float],
    radius: float = 0.018,
) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, p0, p1)
    scn.ngeom += 1


def draw_ghost_arm(
    scn: mujoco.MjvScene,
    data: mujoco.MjData,
    body_ids: list[int],
    *,
    rgba: tuple[float, float, float, float] = (0.25, 0.75, 1.0, 0.35),
    radius: float = 0.016,
) -> None:
    """Draw semi-transparent capsules along consecutive body origins (P4 ghost)."""
    pts: list[np.ndarray] = []
    for bid in body_ids:
        if bid < 0:
            continue
        pts.append(np.asarray(data.xpos[bid], dtype=np.float64).copy())
    for i in range(len(pts) - 1):
        if np.linalg.norm(pts[i + 1] - pts[i]) < 1e-6:
            continue
        _add_capsule(scn, pts[i], pts[i + 1], rgba, radius=radius)
