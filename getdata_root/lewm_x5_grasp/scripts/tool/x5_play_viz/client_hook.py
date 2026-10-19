"""Optional helpers for eval client to publish VizFrame (P5)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .frame_bus import publish
from .types import VizFrame


def try_publish(
    *,
    q_cmd: Optional[np.ndarray] = None,
    joint_pos: Optional[np.ndarray] = None,
    end_pos: Optional[np.ndarray] = None,
    cam_high: Optional[np.ndarray] = None,
    cam_wrist: Optional[np.ndarray] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Safe publish for control loop: never raises."""
    try:
        publish(
            VizFrame(
                q_cmd=None if q_cmd is None else np.asarray(q_cmd, dtype=np.float64),
                joint_pos=None if joint_pos is None else np.asarray(joint_pos, dtype=np.float64),
                end_pos=None if end_pos is None else np.asarray(end_pos, dtype=np.float64),
                cam_high=cam_high,
                cam_wrist=cam_wrist,
                meta=dict(meta or {}),
            )
        )
    except Exception:
        pass
