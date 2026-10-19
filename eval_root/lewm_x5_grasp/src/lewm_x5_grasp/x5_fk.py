"""Pure-Python forward kinematics for ARX X5 (from x5.urdf)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .transforms import rpy_to_matrix

# Joint fixed origins from ROS2/X5_ws/.../x5.urdf
_X5_JOINT_SPECS: tuple[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.0603), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02, 0.0, 0.0402), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((-0.264, 0.0, 0.0), (-np.pi, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.06575, -0.001, -0.0825), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.02845, 0.0, 0.0825), (-np.pi, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


def _axis_angle_transform(axis: Sequence[float], theta: float) -> np.ndarray:
    axis_arr = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-12:
        raise ValueError("joint axis must be non-zero")
    x, y, z = axis_arr / norm
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    t = 1.0 - c
    rot = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    return out


def _fixed_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def _matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Extrinsic fixed XYZ (R = Rz @ Ry @ Rx), matching transforms.rpy_to_matrix."""
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


def chain_transforms(joints: Iterable[float]) -> list[np.ndarray]:
    """Return world transforms after each joint (6 entries)."""
    q = np.asarray(list(joints), dtype=np.float64).reshape(-1)
    if q.shape[0] < 6:
        raise ValueError("expected at least 6 joint angles")
    world = np.eye(4, dtype=np.float64)
    out: list[np.ndarray] = []
    for i, (xyz, rpy, axis) in enumerate(_X5_JOINT_SPECS):
        world = world @ _fixed_transform(xyz, rpy) @ _axis_angle_transform(axis, float(q[i]))
        out.append(world.copy())
    return out


def wrist_center(joints: Iterable[float]) -> np.ndarray:
    """Wrist center = origin of joint5 frame (after joints 1..4)."""
    transforms = chain_transforms(joints)
    return transforms[3][:3, 3].copy()


def j1_axis_distance(joints: Iterable[float]) -> float:
    """Horizontal distance from wrist center to joint1 (base Z) axis."""
    wc = wrist_center(joints)
    return float(np.hypot(wc[0], wc[1]))


def full_fk(joints: Iterable[float]) -> np.ndarray:
    """Return link6 / joint6 frame xyzrpy (6,) in base — NOT eef_link tip.

    Official ARX ``X5.urdf`` places ``eef_link`` a further fixed
    ``+0.145 m`` along link6 +X (``gripper_fixed_joint``). Use
    :func:`eef_link_fk` to match bake / arx_example tip.
    """
    transform = chain_transforms(joints)[-1]
    xyz = transform[:3, 3].copy()
    rpy = _matrix_to_rpy(transform[:3, :3])
    return np.concatenate([xyz, rpy])


# X5.urdf: <joint name="gripper_fixed_joint" type="fixed">
#   <origin xyz="0.145 0 0" rpy="0 0 0"/> parent=link6 child=eef_link
EEF_LINK_OFFSET_IN_LINK6_M = np.array([0.145, 0.0, 0.0], dtype=np.float64)


def eef_link_T(joints: Iterable[float]) -> np.ndarray:
    """4x4 pose of ``eef_link`` (bake / arx_example tip frame)."""
    T6 = chain_transforms(joints)[-1]
    tip = T6.copy()
    tip[:3, 3] = T6[:3, 3] + T6[:3, :3] @ EEF_LINK_OFFSET_IN_LINK6_M
    return tip


def eef_link_fk(joints: Iterable[float]) -> np.ndarray:
    """Return eef_link xyzrpy (6,) — same as ``bake_lerobot_eef_from_fk``.

    Orientation equals link6; position is link6 + 0.145 m along link6 +X.
    """
    tip = eef_link_T(joints)
    xyz = tip[:3, 3].copy()
    rpy = _matrix_to_rpy(tip[:3, :3])
    return np.concatenate([xyz, rpy])


@dataclass(frozen=True)
class SingularityMetrics:
    elbow_proximity: float
    forearm_proximity: float
    shoulder_proximity: float
    wrist_proximity: float

    @property
    def max_proximity(self) -> float:
        return float(
            max(
                self.elbow_proximity,
                self.forearm_proximity,
                self.shoulder_proximity,
                self.wrist_proximity,
            )
        )

    @property
    def is_dangerous(self) -> bool:
        return self.max_proximity > 0.0
