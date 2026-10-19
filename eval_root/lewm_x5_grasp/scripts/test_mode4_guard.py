#!/usr/bin/env python3
"""Quick checks for mode4_guard (no ROS required)."""

from __future__ import annotations

import numpy as np

from lewm_x5_grasp.mode4_guard import Mode4Guard, Mode4GuardConfig
from lewm_x5_grasp.schemas import ArmCommand, ArmState


def _state(
    end_pos: list[float],
    joint_vel: list[float] | None = None,
) -> ArmState:
    return ArmState(
        end_pos=np.asarray(end_pos, dtype=float),
        joint_pos=np.zeros(7),
        joint_vel=np.asarray(joint_vel or [0.0] * 7, dtype=float),
        joint_cur=np.zeros(7),
        stamp_sec=0.0,
    )


def test_pitch_limit() -> None:
    guard = Mode4Guard(Mode4GuardConfig(max_pitch_rad=1.2, max_rpy_step_rad=0.5))
    cmd = ArmCommand(xyzrpy=[0.3, 0.0, 0.2, 0.0, 1.6, 0.0], gripper=0.0, mode=4)
    result = guard.apply(cmd, _state([0.3, 0.0, 0.2, 0.0, 1.1, 0.0]))
    assert abs(result.command.xyzrpy[4] - 1.2) < 1e-6
    assert result.modified


def test_joint_vel_hold() -> None:
    guard = Mode4Guard(Mode4GuardConfig(max_joint_vel_rad_s=2.0, hold_sec=0.0))
    cmd = ArmCommand(xyzrpy=[0.5, 0.0, 0.3, 0.0, 0.0, 0.0], gripper=1.0, mode=4)
    current = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
    result = guard.apply(cmd, _state(current, joint_vel=[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0]))
    assert result.held
    assert result.command.xyzrpy[:3] == current[:3]


def test_joint_vel_hold_cooldown() -> None:
    guard = Mode4Guard(Mode4GuardConfig(max_joint_vel_rad_s=2.0, hold_sec=0.3))
    cmd = ArmCommand(xyzrpy=[0.5, 0.0, 0.3, 0.0, 0.0, 0.0], gripper=1.0, mode=4)
    current = [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
    spike = _state(current, joint_vel=[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    calm = _state(current, joint_vel=[0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    assert guard.apply(cmd, spike).held
    # Still holding even after velocity drops (cooldown).
    held_again = guard.apply(cmd, calm)
    assert held_again.held
    assert "hold_cooldown" in ",".join(held_again.reasons)


def test_mode5_passthrough() -> None:
    guard = Mode4Guard()
    cmd = ArmCommand(xyzrpy=[0.3, 0.0, 0.2, 0.0, 1.6, 0.0], gripper=0.0, mode=5)
    result = guard.apply(cmd, _state([0.3, 0.0, 0.2, 0.0, 0.0, 0.0]))
    assert not result.modified
    assert result.command.xyzrpy[4] == 1.6


if __name__ == "__main__":
    test_pitch_limit()
    test_joint_vel_hold()
    test_joint_vel_hold_cooldown()
    test_mode5_passthrough()
    print("mode4_guard tests passed")
