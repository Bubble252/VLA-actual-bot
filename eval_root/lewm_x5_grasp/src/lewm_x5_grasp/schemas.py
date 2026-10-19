from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ArmState:
    end_pos: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_cur: np.ndarray
    stamp_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "end_pos": self.end_pos.tolist(),
            "joint_pos": self.joint_pos.tolist(),
            "joint_vel": self.joint_vel.tolist(),
            "joint_cur": self.joint_cur.tolist(),
            "stamp_sec": self.stamp_sec,
        }


@dataclass
class Observation:
    arm_state: ArmState
    image: Optional[np.ndarray]
    phase: str
    step_index: int
    depth_image: Optional[np.ndarray] = None
    camera_info: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        image_shape = None if self.image is None else list(self.image.shape)
        depth_shape = None if self.depth_image is None else list(self.depth_image.shape)
        return {
            "arm_state": self.arm_state.to_dict(),
            "image": self.image,
            "image_shape": image_shape,
            "depth_image": self.depth_image,
            "depth_shape": depth_shape,
            "camera_info": self.camera_info,
            "phase": self.phase,
            "step_index": self.step_index,
            "history": self.history,
        }


@dataclass
class PolicyOutput:
    target_xyzrpy: List[float]
    gripper: float
    phase: Optional[str] = None


@dataclass
class ArmCommand:
    xyzrpy: List[float]
    gripper: float
    mode: int = 4
    joint_pos: Optional[List[float]] = None
