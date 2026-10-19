#!/usr/bin/env python3
"""CLI wrapper: bake observations/eef_fk (logic lives in episode_replay.py)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from episode_replay import bake_eef_fk, write_eef_fk_to_hdf5  # noqa: E402
from lewm_x5_grasp.arm_ros2 import X5ArmBridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bake observations/eef_fk (standalone; enable via config auto_bake_before_eef_replay)."
    )
    p.add_argument("hdf5", type=Path, help="Input HDF5 episode path.")
    p.add_argument("--output", type=Path, default=None, help="Output HDF5 (default: in-place).")
    p.add_argument("--replay-hz", type=float, default=100.0)
    p.add_argument("--smooth-window", type=int, default=15)
    p.add_argument("--quat-smooth-window", type=int, default=7)
    p.add_argument("--blend-sec", type=float, default=0.5)
    p.add_argument("--max-blend-sec", type=float, default=2.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = args.hdf5.resolve()
    if not hdf5_path.is_file():
        raise SystemExit(f"Not found: {hdf5_path}")
    output = (args.output or hdf5_path).resolve()

    if args.dry_run:
        print(f"Would bake {hdf5_path} -> {output}")
        return

    rclpy.init()
    bridge = X5ArmBridge(
        mode4_guard={
            "enabled": True,
            "hold_on_joint_vel_spike": False,
            "max_joint_vel_rad_s": 3.5,
        }
    )
    try:
        eef_fk = bake_eef_fk(
            bridge,
            hdf5_path,
            replay_hz=args.replay_hz,
            smooth_window=args.smooth_window,
            quat_smooth_window=args.quat_smooth_window,
            blend_sec=args.blend_sec,
            max_blend_sec=args.max_blend_sec,
            write_back=False,
        )
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    target = write_eef_fk_to_hdf5(
        hdf5_path,
        eef_fk,
        output=None if output == hdf5_path else output,
        quat_smooth_window=args.quat_smooth_window,
    )
    print(f"Wrote observations/eef_fk -> {target}", flush=True)


if __name__ == "__main__":
    main()
