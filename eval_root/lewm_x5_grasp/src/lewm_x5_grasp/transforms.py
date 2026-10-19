from __future__ import annotations

from typing import Iterable, List

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def xyzrpy_to_matrix(xyzrpy: Iterable[float]) -> np.ndarray:
    values = list(float(v) for v in xyzrpy)
    if len(values) != 6:
        raise ValueError("xyzrpy must have length 6.")

    transform = np.eye(4)
    transform[:3, :3] = rpy_to_matrix(values[3], values[4], values[5])
    transform[:3, 3] = values[:3]
    return transform


def transform_point(transform: np.ndarray, point_xyz: Iterable[float]) -> np.ndarray:
    point = np.asarray(list(point_xyz), dtype=float)
    if point.shape != (3,):
        raise ValueError("point_xyz must have length 3.")
    hom = np.ones(4)
    hom[:3] = point
    return (transform @ hom)[:3]


def replace_xyz(xyzrpy: Iterable[float], xyz: Iterable[float]) -> List[float]:
    result = list(float(v) for v in xyzrpy)
    result[:3] = [float(v) for v in xyz]
    return result
