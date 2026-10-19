#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import rclpy

from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.image_source import build_image_source
from lewm_x5_grasp.policy import build_policy
from lewm_x5_grasp.schemas import Observation
from lewm_x5_grasp.state_machine import GraspStateMachine


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LeWM-style grasp loop on ARX X5.")
    parser.add_argument("--config", required=True, help="Path to the task json config.")
    parser.add_argument("--max-steps", type=int, default=300, help="Safety step cap. Default: 300")
    args = parser.parse_args()

    config = load_config(args.config)
    loop_hz = float(config.get("loop_hz", 10.0))
    sleep_sec = 1.0 / max(loop_hz, 1.0)
    image_source = build_image_source(config)
    policy = build_policy(config)
    machine = GraspStateMachine(config)

    history: List[Dict[str, Any]] = []
    last_logged_phase = None
    last_logged_target = None
    last_wait_log = 0.0

    rclpy.init()
    guard_cfg = None
    if "mode4_guard" in config or "workspace" in config:
        from lewm_x5_grasp.mode4_guard_config import mode4_guard_from_task_config

        guard_cfg = mode4_guard_from_task_config(config)
    node = X5ArmBridge(mode4_guard=guard_cfg)

    try:
        node.wait_for_state()
        node.get_logger().info(f"LeWM grasp loop started with policy={config['policy']['type']}")

        for step_index in range(args.max_steps):
            rclpy.spin_once(node, timeout_sec=0.0)
            arm_state = node.latest_state
            if arm_state is None:
                time.sleep(sleep_sec)
                continue

            image = image_source.read()
            observation = Observation(
                arm_state=arm_state,
                image=image,
                phase=machine.phase,
                step_index=step_index,
                depth_image=getattr(image_source, "last_depth", None),
                camera_info=getattr(image_source, "camera_info", None),
                history=history[-8:],
            )
            proposal = policy.predict(observation)
            command = machine.update(arm_state, proposal)
            node.publish_command(command)

            now = time.monotonic()
            target_key = tuple(round(float(v), 4) for v in command.xyzrpy[:3])
            should_log = machine.phase != last_logged_phase or target_key != last_logged_target
            if machine.phase == "wait" and now - last_wait_log >= 5.0:
                should_log = True
                last_wait_log = now
            if should_log:
                current_xyz = [float(v) for v in arm_state.end_pos[:3]]
                target_xyz = [float(v) for v in command.xyzrpy[:3]]
                dist = sum((a - b) ** 2 for a, b in zip(current_xyz, target_xyz)) ** 0.5
                node.get_logger().info(
                    "phase=%s current_xyz=[%.3f, %.3f, %.3f] target_xyz=[%.3f, %.3f, %.3f] "
                    "dist=%.3f gripper=%.3f"
                    % (
                        machine.phase,
                        current_xyz[0],
                        current_xyz[1],
                        current_xyz[2],
                        target_xyz[0],
                        target_xyz[1],
                        target_xyz[2],
                        dist,
                        command.gripper,
                    )
                )
                last_logged_phase = machine.phase
                last_logged_target = target_key

            history.append(
                {
                    "step": step_index,
                    "phase": machine.phase,
                    "target_xyzrpy": command.xyzrpy,
                    "gripper": command.gripper,
                }
            )

            if machine.is_done():
                node.get_logger().info("Grasp state machine reached done.")
                break

            time.sleep(sleep_sec)
        else:
            node.get_logger().warning("Stopped because max_steps was reached.")
    finally:
        close = getattr(image_source, "close", None)
        if close is not None:
            close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
