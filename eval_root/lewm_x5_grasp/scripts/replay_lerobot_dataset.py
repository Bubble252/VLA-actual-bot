#!/usr/bin/env python3
"""Replay a saved LeRobot episode on the real ARX X5 arm."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lerobot_episode_io import ACTION_KEY, EEF_KEY, load_parquet_episode

from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.schemas import ArmCommand


MODE_GRAVITY_COMPENSATION = 3
MODE_POSITION_CONTROL = 5


def _keyframe_indices(t_out: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_rel, t_out, side="right") - 1
    return np.clip(idx, 0, len(t_rel) - 1)


def interpolate_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    out_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(stamps) < 2:
        return eef, qpos, 1.0 / max(out_hz, 1.0)

    t_rel = stamps - stamps[0]
    duration = max(float(t_rel[-1]), 1.0 / max(out_hz, 1.0))
    n_out = max(2, int(duration * out_hz) + 1)
    t_out = np.linspace(0.0, duration, n_out)
    idx = _keyframe_indices(t_out, t_rel)

    eef_out = np.zeros((n_out, eef.shape[1]), dtype=np.float64)
    qpos_out = np.zeros((n_out, qpos.shape[1]), dtype=np.float64)
    for d in range(min(6, qpos.shape[1])):
        qpos_out[:, d] = np.interp(t_out, t_rel, qpos[:, d])
    if qpos.shape[1] > 6:
        qpos_out[:, 6] = np.interp(t_out, t_rel, qpos[:, 6])
    for d in range(min(3, eef.shape[1])):
        eef_out[:, d] = np.interp(t_out, t_rel, eef[:, d])
    if eef.shape[1] >= 6:
        eef_out[:, 3:6] = eef[idx, 3:6]
    if eef.shape[1] > 6:
        eef_out[:, 6] = qpos_out[:, 6]
    return eef_out, qpos_out, 1.0 / max(out_hz, 1.0)


def _publish_frame(bridge: X5ArmBridge, eef_row: np.ndarray, qpos_row: np.ndarray) -> None:
    # Use joint position mode; mode=4 is endpoint IK control in the ARX driver.
    bridge.publish_command(
        ArmCommand(
            xyzrpy=eef_row[:6].tolist(),
            gripper=float(qpos_row[6]),
            joint_pos=qpos_row[:6].tolist(),
            mode=MODE_POSITION_CONTROL,
        )
    )


def _enter_position_control(bridge: X5ArmBridge) -> None:
    state = bridge.latest_state
    if state is None:
        return
    bridge.publish_command(
        ArmCommand(
            xyzrpy=state.end_pos.tolist(),
            gripper=float(state.joint_pos[6]),
            mode=MODE_POSITION_CONTROL,
        )
    )


def _enter_mode3(bridge: X5ArmBridge) -> None:
    state = bridge.latest_state
    if state is None:
        return
    bridge.publish_command(
        ArmCommand(
            xyzrpy=state.end_pos.tolist(),
            gripper=float(state.joint_pos[6]),
            mode=MODE_GRAVITY_COMPENSATION,
        )
    )


def replay_episode(
    bridge: X5ArmBridge,
    parquet_path: Path,
    replay_hz: float,
    use_action: bool,
) -> None:
    data = load_parquet_episode(parquet_path)
    qpos = data["action"] if use_action else data["state"]
    stamps = data["timestamp"]
    n = len(qpos)
    eef = data.get("eef")

    if eef is None:
        print(
            "WARNING: parquet has no observation.eef 鈥?arm motion may be wrong.\n"
            "  Re-record with the latest collector (eef is saved now), or verify via mp4 only.",
            flush=True,
        )
        eef = np.zeros((n, 7), dtype=np.float64)

    eef_i, qpos_i, period = interpolate_trajectory(eef, qpos, stamps, out_hz=replay_hz)
    n_play = len(qpos_i)
    src = "action" if use_action else "state"
    print(
        f"Replaying {parquet_path.name}: {n} frames -> {n_play} steps @ {replay_hz:.0f} Hz "
        f"(joint source={src}). Ctrl+C to abort.",
        flush=True,
    )

    _enter_position_control(bridge)
    time.sleep(0.3)

    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is not None:
        cur_eef = np.asarray(state.end_pos[:6], dtype=np.float64)
        cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
        blend_steps = max(5, int(replay_hz * 0.4))
        for k in range(1, blend_steps + 1):
            alpha = k / blend_steps
            eef_b = cur_eef * (1 - alpha) + eef_i[0, :6] * alpha
            qpos_b = cur_qpos * (1 - alpha) + qpos_i[0] * alpha
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_frame(bridge, eef_b, qpos_b)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_frame(bridge, eef_i[i], qpos_i[i])
            if i % max(1, int(replay_hz)) == 0:
                print(f"  replay {i}/{n_play}", flush=True)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Replay aborted.", flush=True)
    finally:
        _enter_mode3(bridge)

    print("Replay finished.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay LeRobot parquet episode on ARX X5.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/arx_x5_real/runs/latest"),
        help="LeRobot dataset root (dated run folder or latest symlink).",
    )
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    p.add_argument("--replay-hz", type=float, default=50.0, help="Playback control rate.")
    p.add_argument(
        "--joint-source",
        choices=("action", "state"),
        default="action",
        help="Use action (commanded) or state (measured) joints. Default: action",
    )
    args = p.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if dataset.is_symlink():
        dataset = dataset.resolve()

    parquet = dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    if not parquet.exists():
        raise SystemExit(f"Episode not found: {parquet}")

    info_path = dataset / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        tasks_path = dataset / "meta" / "episodes.jsonl"
        task = "?"
        if tasks_path.exists():
            for line in tasks_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if int(row["episode_index"]) == args.episode:
                    task = row["tasks"][0]
                    break
        print("=" * 60)
        print(f"Dataset : {dataset}")
        print(f"Episode : {args.episode}")
        print(f"Task    : {task}")
        print(f"Parquet : {parquet}")
        print(f"FPS meta: {info.get('fps')}")
        print("=" * 60)
        print("Ensure teach launch is running. Arm will move in 2 s...")
        time.sleep(2)

    rclpy.init()
    bridge = X5ArmBridge()
    try:
        bridge.wait_for_state(timeout_sec=30.0)
        replay_episode(
            bridge,
            parquet,
            replay_hz=args.replay_hz,
            use_action=(args.joint_source == "action"),
        )
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
