from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class VizFrame:
    """One visualization tick shared between eval client and viz process."""

    # Prefer commanded joints for FK; fall back to feedback joints.
    q_cmd: Optional[np.ndarray] = None  # (6,) or (6+gripper,)
    joint_pos: Optional[np.ndarray] = None
    end_pos: Optional[np.ndarray] = None  # raw SDK xyzrpy (6,); viz may +EEF_XYZ_OFFSET when drawing
    cam_high: Optional[np.ndarray] = None  # HxWx3 uint8
    cam_wrist: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def joints_for_fk(self) -> Optional[np.ndarray]:
        if self.q_cmd is not None:
            return np.asarray(self.q_cmd, dtype=np.float64).reshape(-1)
        if self.joint_pos is not None:
            return np.asarray(self.joint_pos, dtype=np.float64).reshape(-1)
        return None
