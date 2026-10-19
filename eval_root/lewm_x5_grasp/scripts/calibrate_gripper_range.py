#!/usr/bin/env python3
"""Sweep /arm_cmd gripper values and print /arm_status joint_pos[6].

Use this once to find the open/close command range for your X5 firmware.
Keyboard node uses `o`/`c` (+/- 0.15); larger value = more open on this setup.

Example:
  source ROS2/X5_ws/install/setup.bash
  python3 scripts/calibrate_gripper_range.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from lewm_x5_grasp.arm_ros2 import X5ArmBridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find ARX X5 gripper open/close cmd range.")
    p.add_argument("--min", type=float, default=0.0, help="Lowest gripper cmd to try.")
    p.add_argument("--max", type=float, default=3.0, help="Highest gripper cmd to try.")
    p.add_argument("--step", type=float, default=0.3, help="Cmd step size.")
    p.add_argument("--hold", type=float, default=0.8, help="Seconds per step.")
    p.add_argument("--mode", type=int, default=5, help="RobotCmd mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    bridge = X5ArmBridge()
    try:
        state = bridge.wait_for_state(timeout_sec=30.0)
        joints = state.joint_pos[:6].tolist()
        print("Sweep gripper cmd -> arm_status joint_pos[6]")
        print(f"  joint hold: {joints[:3]} ...")
        print("-" * 48)
        best_open_cmd = args.min
        best_open_status = float("-inf")
        for cmd in _frange(args.min, args.max + 1e-9, args.step):
            deadline = time.monotonic() + args.hold
            while time.monotonic() < deadline:
                bridge.publish_joint_command(joint_pos_6=joints, gripper=cmd, mode=args.mode)
                rclpy.spin_once(bridge, timeout_sec=0.02)
            status = bridge.latest_state
            g = float(status.joint_pos[6]) if status is not None else float("nan")
            print(f"  cmd={cmd:+.2f}  status[6]={g:+.3f}")
            if g > best_open_status:
                best_open_status = g
                best_open_cmd = cmd
        print("-" * 48)
        print(f"Suggested gripper_open_cmd: {best_open_cmd:.2f}  (status={best_open_status:.3f})")
        print("Put that value in config/teleop_uarm.example.json -> mapping.gripper_open_cmd")
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _frange(start: float, stop: float, step: float):
    v = start
    while v <= stop + 1e-9:
        yield round(v, 4)
        v += step


if __name__ == "__main__":
    main()
