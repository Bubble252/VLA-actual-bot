#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import rclpy

from lewm_x5_grasp.arm_ros2 import X5ArmBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect X5 episode data for future LeWM training.")
    parser.add_argument("--output-dir", required=True, help="Episode directory to create.")
    parser.add_argument("--image-path", required=True, help="Path to the latest frame file.")
    parser.add_argument("--hz", type=float, default=5.0, help="Sampling rate. Default: 5")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duration. Default: 30")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "samples.jsonl"
    image_path = Path(args.image_path).expanduser().resolve()

    rclpy.init()
    node = X5ArmBridge()
    sleep_sec = 1.0 / max(args.hz, 1.0)

    try:
        node.wait_for_state()
        deadline = time.monotonic() + args.seconds
        step = 0
        with open(metadata_path, "w", encoding="utf-8") as f:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.0)
                state = node.latest_state
                if state is None:
                    time.sleep(sleep_sec)
                    continue

                frame_name = None
                if image_path.exists():
                    frame_name = f"frame_{step:06d}{image_path.suffix}"
                    shutil.copy2(image_path, frames_dir / frame_name)

                record = {
                    "step": step,
                    "time": time.time(),
                    "frame": frame_name,
                    "arm_state": state.to_dict(),
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                f.flush()
                step += 1
                time.sleep(sleep_sec)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
