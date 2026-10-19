"""P2: read /arm_status into VizFrame (joint feedback → FK)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .types import VizFrame


class RosStateSource:
    """Subscribe ROS2 /arm_status via lewm X5ArmBridge (or minimal Node)."""

    def __init__(self, *, timeout_sec: float = 5.0):
        self._timeout_sec = timeout_sec
        self._node = None
        self._step = 0
        self._init_ros()

    def _init_ros(self) -> None:
        import rclpy
        from lewm_x5_grasp.arm_ros2 import X5ArmBridge

        if not rclpy.ok():
            rclpy.init(args=None)
        self._rclpy = rclpy
        self._node = X5ArmBridge()
        t0 = time.time()
        while self._node.latest_state is None and (time.time() - t0) < self._timeout_sec:
            rclpy.spin_once(self._node, timeout_sec=0.05)
        if self._node.latest_state is None:
            raise RuntimeError(
                "Timed out waiting for /arm_status. "
                "Start open_single_arm.launch.py first."
            )
        print("[ros] /arm_status connected", flush=True)

    def next(self) -> Optional[VizFrame]:
        assert self._node is not None
        self._rclpy.spin_once(self._node, timeout_sec=0.0)
        state = self._node.latest_state
        if state is None:
            return None
        self._step += 1
        jp = np.asarray(state.joint_pos, dtype=np.float64).reshape(-1)
        ep = np.asarray(state.end_pos, dtype=np.float64).reshape(-1)
        return VizFrame(
            # No q_cmd in pure ROS mode → FK uses feedback joints.
            joint_pos=jp[:6],
            end_pos=ep[:6] if ep.size >= 6 else None,
            meta={"step": self._step, "source": "ros"},
        )

    def close(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
