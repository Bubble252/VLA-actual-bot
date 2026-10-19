"""ARX ↔ LIBERO-like frame alignment (training tag ``neg_x_z1``).

Training convert (ARX → LIBERO-like):
  x' = -x
  y' =  y
  z' =  z + 1.0
  R' = S R S,  S = diag(-1, 1, 1)

Deployment must invert model outputs (LIBERO-like → ARX) before applying
to the real arm. Absolute z-offset cancels for *deltas*; only the X reflection
and rotation conjugacy remain.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .transforms import rpy_to_matrix

S_NEG_X = np.diag([-1.0, 1.0, 1.0])
Z_OFFSET = 1.0


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Extrinsic fixed XYZ (R = Rz @ Ry @ Rx), matching ``rpy_to_matrix``."""
    r20 = float(rot[2, 0])
    pitch = float(-np.arcsin(np.clip(r20, -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))
    else:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _needs_neg_x(mode: str) -> bool:
    return str(mode).startswith("neg_x_z1")


def arx_xyz_to_libero(xyz: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    """Forward: ARX absolute xyz → LIBERO-like (for state / dataset)."""
    p = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
    if not _needs_neg_x(mode):
        return p
    p[0] = -p[0]
    p[2] = p[2] + Z_OFFSET
    return p


def libero_xyz_to_arx(xyz: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    """Inverse: LIBERO-like absolute xyz → ARX."""
    p = np.asarray(xyz, dtype=np.float64).reshape(3).copy()
    if not _needs_neg_x(mode):
        return p
    p[0] = -p[0]
    p[2] = p[2] - Z_OFFSET
    return p


def arx_rpy_to_libero(rpy: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    if not _needs_neg_x(mode):
        return np.asarray(rpy, dtype=np.float64).reshape(3).copy()
    Rm = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return matrix_to_rpy(S_NEG_X @ Rm @ S_NEG_X)


def libero_rpy_to_arx(rpy: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    # S = S^{-1}, so conjugacy is involutive.
    return arx_rpy_to_libero(rpy, mode=mode)


def arx_eef_to_libero(xyzrpy: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    e = np.asarray(xyzrpy, dtype=np.float64).reshape(6).copy()
    if not _needs_neg_x(mode):
        return e
    e[:3] = arx_xyz_to_libero(e[:3], mode=mode)
    e[3:6] = arx_rpy_to_libero(e[3:6], mode=mode)
    return e


def libero_eef_to_arx(xyzrpy: Sequence[float], mode: str = "neg_x_z1") -> np.ndarray:
    e = np.asarray(xyzrpy, dtype=np.float64).reshape(6).copy()
    if not _needs_neg_x(mode):
        return e
    e[:3] = libero_xyz_to_arx(e[:3], mode=mode)
    e[3:6] = libero_rpy_to_arx(e[3:6], mode=mode)
    return e


def action_deltas_libero_to_arx(
    world_delta: Sequence[float],
    rot_delta: Sequence[float],
    mode: str = "neg_x_z1",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse on policy OSC deltas (LIBERO-like → ARX).

    Position: Δp_arx = S Δp_libero  (z-offset cancels for deltas).
    Rotation: R_arx = S R_libero S.
    """
    wd = np.asarray(world_delta, dtype=np.float64).reshape(3).copy()
    rd = np.asarray(rot_delta, dtype=np.float64).reshape(3).copy()
    if not _needs_neg_x(mode):
        return wd.astype(np.float32), rd.astype(np.float32)
    wd = S_NEG_X @ wd
    Rm = rpy_to_matrix(float(rd[0]), float(rd[1]), float(rd[2]))
    rd = matrix_to_rpy(S_NEG_X @ Rm @ S_NEG_X)
    return wd.astype(np.float32), rd.astype(np.float32)
