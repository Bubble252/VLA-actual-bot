#!/usr/bin/env python3
"""Record kinesthetic-teaching demonstrations for single-arm ARX X5.

Adapted from AC_One act/collect.py for ARX_X5_clean (single arm + D405, keyboard).

Prerequisites:
  1. CAN up:  bash ARX_CAN/arx_can/arx_can1.sh
  2. Teach mode: ros2 launch arx_x5_controller open_single_arm_teach.launch.py
  3. This script: ./run_record_teach.sh

Controls (terminal must be focused):
  0 / Space  start episode (records init pose, begin logging)
  s / Enter  stop and save current episode
  d          delete last saved episode
  q          quit
  Auto-stop: return within 0.05 rad of init pose for 2 s (after first 2 s)
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import rclpy

from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.image_source import RealSenseD405ImageSource
from lewm_x5_grasp.schemas import ArmCommand, ArmState


STATE_DIM = 7
EEF_DIM = 7
CAMERA_NAME = "wrist"


class KeyPoller:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        self._lock = threading.Lock()
        self._pending: List[str] = []

    def __enter__(self) -> KeyPoller:
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *args: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> List[str]:
        keys: List[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch:
                keys.append(ch)
        with self._lock:
            self._pending.extend(keys)
            out = self._pending[:]
            self._pending.clear()
        return out


def state_to_obs(state: ArmState) -> Dict[str, np.ndarray]:
    qpos = np.asarray(state.joint_pos, dtype=np.float64)
    eef = np.concatenate([np.asarray(state.end_pos, dtype=np.float64), qpos[6:7]])
    return {
        "qpos": qpos,
        "eef": eef,
        "qvel": np.asarray(state.joint_vel, dtype=np.float64),
        "effort": np.asarray(state.joint_cur, dtype=np.float64),
    }


def enter_gravity_compensation(bridge: X5ArmBridge) -> None:
    state = bridge.latest_state
    if state is None:
        return
    bridge.publish_command(
        ArmCommand(
            xyzrpy=state.end_pos.tolist(),
            gripper=float(state.joint_pos[6]),
            mode=3,
        )
    )


def wait_for_start_key(
    bridge: X5ArmBridge,
    keys: KeyPoller,
    episode_idx: int,
    enter_gravity: bool,
) -> Union[str, np.ndarray]:
    print(f"\nEpisode {episode_idx}: press 0/Space to start, d=delete last, q=quit")

    while rclpy.ok():
        rclpy.spin_once(bridge, timeout_sec=0.0)
        state = bridge.latest_state
        if state is None:
            time.sleep(0.01)
            continue

        for ch in keys.poll():
            if ch in {"0", " "}:
                init_pos = np.asarray(state.joint_pos, dtype=np.float64).copy()
                if enter_gravity:
                    enter_gravity_compensation(bridge)
                print(f"Recording started. init qpos[:7]={init_pos.round(3).tolist()}")
                return init_pos
            if ch in {"d", "2"}:
                return "delete"
            if ch in {"q", "\x03"}:
                raise KeyboardInterrupt

        time.sleep(0.01)

    return "quit"


def collect_episode(
    bridge: X5ArmBridge,
    camera: RealSenseD405ImageSource,
    keys: KeyPoller,
    init_pos: np.ndarray,
    hz: float,
    max_steps: int,
    auto_stop: bool,
    home_tol: float,
    home_hold_sec: float,
) -> Optional[Dict[str, List[Any]]]:
    period = 1.0 / max(hz, 1.0)
    buffers: Dict[str, List[Any]] = {
        "qpos": [],
        "eef": [],
        "qvel": [],
        "effort": [],
        "action": [],
        "images": [],
        "depth": [],
        "time": [],
        "arm_stamp": [],
    }

    return_home_start: Optional[float] = None
    step = 0

    while rclpy.ok() and step < max_steps:
        loop_start = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)

        for ch in keys.poll():
            if ch in {"s", "\r", "\n"}:
                print("Manual stop.")
                return buffers if buffers["qpos"] else None
            if ch in {"q", "\x03"}:
                raise KeyboardInterrupt

        state = bridge.latest_state
        if state is None:
            time.sleep(0.001)
            continue

        rgb = camera.read()
        if rgb is None:
            time.sleep(0.001)
            continue

        obs = state_to_obs(state)
        action = obs["qpos"].copy()

        buffers["qpos"].append(obs["qpos"])
        buffers["eef"].append(obs["eef"])
        buffers["qvel"].append(obs["qvel"])
        buffers["effort"].append(obs["effort"])
        buffers["action"].append(action)
        buffers["images"].append(rgb)
        depth = camera.last_depth
        buffers["depth"].append(None if depth is None else depth.copy())
        buffers["time"].append(time.time())
        buffers["arm_stamp"].append(state.stamp_sec)

        if auto_stop and step > int(hz * 2):
            if all(abs(v - i) <= home_tol for v, i in zip(action, init_pos)):
                if return_home_start is None:
                    return_home_start = time.time()
                    print("Returned near init pose, holding 2 s to auto-stop...")
                elif time.time() - return_home_start >= home_hold_sec:
                    print("Auto-stop after holding init pose.")
                    return buffers
            elif return_home_start is not None:
                return_home_start = None

        step += 1
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, period - elapsed))

    print(f"Reached max_steps={max_steps}.")
    return buffers if buffers["qpos"] else None


def write_hdf5(path: Path, task: str, buffers: Dict[str, List[Any]], use_depth: bool) -> None:
    import h5py

    n = len(buffers["qpos"])
    images = np.stack(buffers["images"], axis=0).astype(np.uint8)

    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["task"] = str(task)
        root.attrs["state_dim"] = STATE_DIM
        root.attrs["camera_names"] = json.dumps([CAMERA_NAME])

        obs = root.create_group("observations")
        img_grp = obs.create_group("images")
        img_grp.create_dataset(CAMERA_NAME, data=images, compression="gzip", compression_opts=4)

        if use_depth and any(d is not None for d in buffers["depth"]):
            depth_grp = obs.create_group("images_depth")
            depth_stack = []
            for d in buffers["depth"]:
                depth_stack.append(np.zeros_like(buffers["depth"][0]) if d is None else d)
            depth_grp.create_dataset(
                CAMERA_NAME,
                data=np.stack(depth_stack, axis=0).astype(np.uint16),
                compression="gzip",
                compression_opts=4,
            )

        for name in ("qpos", "eef", "qvel", "effort"):
            obs.create_dataset(name, data=np.stack(buffers[name], axis=0))

        root.create_dataset("action", data=np.stack(buffers["action"], axis=0))
        root.create_dataset("action_eef", data=np.stack(buffers["eef"], axis=0))
        root.create_dataset("time", data=np.asarray(buffers["time"], dtype=np.float64))
        root.create_dataset("arm_stamp", data=np.asarray(buffers["arm_stamp"], dtype=np.float64))

    print(f"Saved {n} steps -> {path}")


def write_jsonl_episode(ep_dir: Path, task: str, buffers: Dict[str, List[Any]]) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ep_dir / "images"
    img_dir.mkdir(exist_ok=True)

    meta = {"task": task, "steps": len(buffers["qpos"]), "state_dim": STATE_DIM}
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with open(ep_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(buffers["qpos"])):
            frame_name = f"frame_{i:06d}.png"
            bgr = buffers["images"][i][:, :, ::-1]
            cv2.imwrite(str(img_dir / frame_name), bgr)

            record = {
                "step": i,
                "time": buffers["time"][i],
                "arm_stamp": buffers["arm_stamp"][i],
                "frame": f"images/{frame_name}",
                "qpos": buffers["qpos"][i].tolist(),
                "eef": buffers["eef"][i].tolist(),
                "qvel": buffers["qvel"][i].tolist(),
                "effort": buffers["effort"][i].tolist(),
                "action": buffers["action"][i].tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Saved {len(buffers['qpos'])} steps -> {ep_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record kinesthetic teaching demos (single X5 + D405).")
    p.add_argument("--output-dir", required=True, help="Directory for episode files.")
    p.add_argument("--task", default="grab the bottle", help="Task description stored in metadata.")
    p.add_argument("--hz", type=float, default=20.0, help="Sampling rate. Default: 20")
    p.add_argument("--num-episodes", type=int, default=-1, help="Stop after N episodes. -1 = unlimited.")
    p.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
    p.add_argument("--format", choices=("hdf5", "dir"), default="hdf5", help="Output format.")
    p.add_argument("--use-depth", action="store_true", help="Store depth in HDF5 (if D405 provides it).")
    p.add_argument("--no-auto-stop", action="store_true", help="Disable return-to-init auto stop.")
    p.add_argument("--enter-gravity", action="store_true", help="Publish mode=3 on start (normal launch).")
    p.add_argument("--camera-width", type=int, default=480)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    bridge = X5ArmBridge()
    camera = RealSenseD405ImageSource(
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        display=True,
    )

    episode_idx = 0
    delete_requested = False

    print("=" * 60)
    print("X5 kinesthetic teaching recorder")
    print(f"  output : {output_dir}")
    print(f"  task   : {args.task}")
    print(f"  hz     : {args.hz}")
    print(f"  format : {args.format}")
    print("=" * 60)
    print("Launch teach mode first:")
    print("  ros2 launch arx_x5_controller open_single_arm_teach.launch.py")
    print("=" * 60)

    try:
        bridge.wait_for_state(timeout_sec=30.0)
        print("Receiving /arm_status.")

        with KeyPoller() as keys:
            while rclpy.ok():
                if args.num_episodes >= 0 and episode_idx >= args.num_episodes:
                    break

                try:
                    start_signal = wait_for_start_key(
                        bridge, keys, episode_idx, enter_gravity=args.enter_gravity
                    )
                except KeyboardInterrupt:
                    break

                if start_signal == "quit":
                    break
                if start_signal == "delete":
                    # delete last episode
                    prev = episode_idx - 1
                    if prev < 0:
                        print("No episode to delete.")
                        continue
                    if args.format == "hdf5":
                        target = output_dir / f"episode_{prev}.hdf5"
                    else:
                        target = output_dir / f"episode_{prev:03d}"
                    if target.exists():
                        if target.is_dir():
                            import shutil

                            shutil.rmtree(target)
                        else:
                            target.unlink()
                        episode_idx = prev
                        print(f"Deleted {target}")
                    else:
                        print(f"Not found: {target}")
                    continue

                init_pos = start_signal
                buffers = collect_episode(
                    bridge=bridge,
                    camera=camera,
                    keys=keys,
                    init_pos=init_pos,
                    hz=args.hz,
                    max_steps=args.max_steps,
                    auto_stop=not args.no_auto_stop,
                    home_tol=0.05,
                    home_hold_sec=2.0,
                )

                if not buffers or not buffers["qpos"]:
                    print("Empty episode, not saved.")
                    continue

                if args.format == "hdf5":
                    path = output_dir / f"episode_{episode_idx}.hdf5"
                    write_hdf5(path, args.task, buffers, use_depth=args.use_depth)
                else:
                    path = output_dir / f"episode_{episode_idx:03d}"
                    write_jsonl_episode(path, args.task, buffers)

                episode_idx += 1

    finally:
        camera.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
