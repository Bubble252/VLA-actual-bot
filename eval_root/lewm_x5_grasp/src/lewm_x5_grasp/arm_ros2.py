from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from arx5_arm_msg.msg import RobotCmd, RobotStatus

from .mode4_guard import Mode4Guard, Mode4GuardConfig
from .schemas import ArmCommand, ArmState


class X5ArmBridge(Node):
    def __init__(self, mode4_guard: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("lewm_x5_grasp")
        self._latest_state: Optional[ArmState] = None
        self._mode4_guard = Mode4Guard(Mode4GuardConfig.from_dict(mode4_guard))
        self._cmd_pub = self.create_publisher(RobotCmd, "/arm_cmd", 10)
        self.create_subscription(RobotStatus, "/arm_status", self._on_status, 10)

    def set_mode4_guard(self, config: Optional[Dict[str, Any]]) -> None:
        self._mode4_guard.update_config(config)

    def _on_status(self, msg: RobotStatus) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self._latest_state = ArmState(
            end_pos=np.asarray(msg.end_pos, dtype=float),
            joint_pos=np.asarray(msg.joint_pos, dtype=float),
            joint_vel=np.asarray(msg.joint_vel, dtype=float),
            joint_cur=np.asarray(msg.joint_cur, dtype=float),
            stamp_sec=stamp,
        )

    @property
    def latest_state(self) -> Optional[ArmState]:
        return self._latest_state

    def wait_for_state(self, timeout_sec: float = 5.0) -> ArmState:
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_sec
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._latest_state is not None:
                return self._latest_state
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec >= deadline:
                break
        raise RuntimeError("Timed out waiting for /arm_status. Confirm X5 controller is running.")

    def publish_command(self, command: ArmCommand) -> None:
        guarded = self._mode4_guard.apply(command, self._latest_state)
        command = guarded.command
        self._mode4_guard.maybe_log(self, guarded)

        if command.joint_pos is not None:
            joint_pos = [float(v) for v in command.joint_pos[:6]]
        elif self._latest_state is not None:
            joint_pos = self._latest_state.joint_pos[:6].tolist()
        else:
            joint_pos = [0.0] * 6

        msg = RobotCmd()
        msg.end_pos = [float(v) for v in command.xyzrpy]
        msg.joint_pos = joint_pos
        msg.gripper = float(command.gripper)
        msg.mode = int(command.mode)
        self._cmd_pub.publish(msg)

    def publish_joint_command(
        self,
        joint_pos_6: list[float],
        gripper: float,
        mode: int = 5,
    ) -> None:
        """Publish joint-space command (mode=5 POSITION_CONTROL)."""
        end_pos = [0.0] * 6
        if self._latest_state is not None:
            end_pos = self._latest_state.end_pos.tolist()

        msg = RobotCmd()
        msg.end_pos = [float(v) for v in end_pos]
        msg.joint_pos = [float(v) for v in joint_pos_6]
        msg.gripper = float(gripper)
        msg.mode = int(mode)
        self._cmd_pub.publish(msg)
