#!/usr/bin/env python3
"""CLI: replay HDF5 via episode_replay control modes (default eef_via_joint).

Uses the same get-data replay stack (blend / smooth / return-to-start) with
optional ``eef_via_joint`` (client differential IK → mode=5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_SRC = _PROJECT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from episode_replay import (  # noqa: E402
    ARM_JOINT_DIMS,
    EefViaJointStreamer,
    load_episode_arrays,
    prepare_replay_trajectory,
    replay_hdf5_file,
)

DEFAULT_HDF5 = Path(
    "/home/casia/project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/"
    "data/pour_foam_from_cup/episode_2026-07-14_200904.hdf5"
)


def _dry_run(
    hdf5_path: Path,
    *,
    control_mode: str,
    replay_hz: float,
    eef_source: str,
    max_steps: int,
) -> None:
    eef, qpos, stamps = load_episode_arrays(
        hdf5_path,
        joint_source="action",
        control_mode=control_mode,  # type: ignore[arg-type]
        eef_source=eef_source,  # type: ignore[arg-type]
    )
    eef_i, qpos_i, period, note = prepare_replay_trajectory(
        eef,
        qpos,
        stamps,
        out_hz=replay_hz,
        control_mode=control_mode,  # type: ignore[arg-type]
    )
    n = len(qpos_i) if max_steps <= 0 else min(len(qpos_i), max_steps)
    print(f"[dry-run] {hdf5_path.name}: {n}/{len(qpos_i)} steps, period={period:.4f}s", flush=True)
    if note:
        print(f"  preprocess: {note}", flush=True)
    if control_mode != "eef_via_joint":
        print(f"  mode={control_mode}: skip IK convert", flush=True)
        return
    streamer = EefViaJointStreamer()
    streamer.reset(qpos_i[0, :ARM_JOINT_DIMS])
    jumps: list[float] = []
    for i in range(n):
        q = streamer.step(eef_i[i], qpos_i[i])
        jumps.append(float(streamer.last_info.get("jump", 0.0)))
        if i in (0, n - 1) or (i + 1) % max(1, int(replay_hz)) == 0:
            dq = float(np.max(np.abs(q - qpos_i[i, :ARM_JOINT_DIMS])))
            print(
                f"  [{i:4d}] ik_ok={streamer.last_info.get('ok')} "
                f"jump={jumps[-1]:.4f} max|q−rec|={dq:.4f}",
                flush=True,
            )
    print(
        f"  done: max_jump={max(jumps) if jumps else 0:.4f} "
        f"mean_jump={float(np.mean(jumps)) if jumps else 0:.4f}",
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Replay HDF5 (default: eef→diff IK→mode=5).")
    p.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5, help="Episode HDF5 path.")
    p.add_argument(
        "--mode",
        choices=("eef_via_joint", "joint", "eef"),
        default="eef_via_joint",
        help="Replay control mode. Default: eef_via_joint",
    )
    p.add_argument("--hz", type=float, default=50.0, help="Replay rate (Hz).")
    p.add_argument(
        "--eef-source",
        choices=("auto", "action", "observations", "eef_fk"),
        default="auto",
    )
    p.add_argument("--dry-run", action="store_true", help="Offline convert only (no ROS).")
    p.add_argument("--max-steps", type=int, default=0, help="Dry-run step cap (0=all).")
    p.add_argument("--no-return-to-start", action="store_true")
    args = p.parse_args()

    hdf5_path = args.hdf5.expanduser().resolve()
    if not hdf5_path.is_file():
        raise SystemExit(f"HDF5 not found: {hdf5_path}")

    if args.dry_run:
        _dry_run(
            hdf5_path,
            control_mode=args.mode,
            replay_hz=args.hz,
            eef_source=args.eef_source,
            max_steps=args.max_steps,
        )
        return 0

    import rclpy
    from lewm_x5_grasp.arm_ros2 import X5ArmBridge

    rclpy.init()
    bridge = X5ArmBridge()
    try:
        replay_hdf5_file(
            bridge,
            hdf5_path,
            replay_hz=args.hz,
            joint_source="action",
            control_mode=args.mode,
            eef_source=args.eef_source,
            return_to_start=not args.no_return_to_start,
        )
    except KeyboardInterrupt:
        print("Aborted.", flush=True)
        return 130
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
