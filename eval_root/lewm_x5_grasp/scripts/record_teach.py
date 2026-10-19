#!/usr/bin/env python3
"""Record demonstrations for single-arm ARX X5 + D405.

Default workflow (--format lerobot):
  1. Press 0, drag arm manually (trajectory only, no camera save)
  2. Press s -> arm auto-replays trajectory (mode=5, joint position)
  3. During replay: record D405 + joint state -> LeRobot v2.1 (dated run dir)

Legacy (--format hdf5): save manual drag directly to HDF5.
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
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lerobot_episode_io import (
    compare_hdf5_parquet,
    hdf5_to_parquet,
    load_hdf5_episode,
    load_parquet_episode,
)
from lerobot_run_writer import LeRobotRunWriter

from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.image_source import MultiRealSenseSource, build_record_cameras
from lewm_x5_grasp.schemas import ArmCommand, ArmState


STATE_DIM = 7
EEF_DIM = 7
ARM_JOINT_DIMS = 6
GRIPPER_JOINT_IDX = 6
MODE_GRAVITY_COMPENSATION = 3
MODE_POSITION_CONTROL = 5
GRIPPER_EVENT_HOLD_SEC = 0.0
GRIPPER_EVENT_DELTA = 0.08
REPLAY_TIME_SCALE = 1.5


def _empty_image_buffers(camera_names: List[str]) -> Dict[str, List[Any]]:
    return {name: [] for name in camera_names}


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
    """Switch to drag/teach mode."""
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


def enter_position_control(bridge: X5ArmBridge, state: Optional[ArmState] = None) -> None:
    """Switch to joint position tracking before replay."""
    state = state or bridge.latest_state
    if state is None:
        return
    bridge.publish_command(
        ArmCommand(
            xyzrpy=state.end_pos.tolist(),
            gripper=float(state.joint_pos[6]),
            mode=MODE_POSITION_CONTROL,
        )
    )


def wait_for_start_key(
    bridge: X5ArmBridge,
    keys: KeyPoller,
    episode_idx: int,
    enter_gravity: bool,
) -> Union[str, np.ndarray]:
    print(
        f"\nEpisode {episode_idx}: press 0/Space to START, drag arm, then s/Enter",
        flush=True,
    )
    print("  (s saves trajectory -> auto replay -> record LeRobot data)", flush=True)

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
                print(f"Recording trajectory (no camera). init qpos={init_pos.round(3).tolist()}", flush=True)
                print("  Drag the arm, then press s or Enter to replay+record.", flush=True)
                print(
                    "  If you stop at the GRASP pose, arm will first return to start "
                    "(not recorded), then replay once forward.",
                    flush=True,
                )
                return init_pos
            if ch in {"s", "\r", "\n"}:
                print(
                    "Not recording yet 鈥?press 0 or Space first, then drag, then s to save.",
                    flush=True,
                )
            if ch in {"d", "2"}:
                return "delete"
            if ch in {"q", "\x03"}:
                raise KeyboardInterrupt

        time.sleep(0.01)

    return "quit"


def collect_episode(
    bridge: X5ArmBridge,
    camera: MultiRealSenseSource,
    camera_names: List[str],
    keys: KeyPoller,
    init_pos: np.ndarray,
    hz: float,
    max_steps: int,
    auto_stop: bool,
    home_tol: float,
    home_hold_sec: float,
    min_steps: int,
    require_move: bool,
    record_observations: bool = True,
) -> Optional[Dict[str, List[Any]]]:
    period = 1.0 / max(hz, 1.0)
    buffers: Dict[str, List[Any]] = {
        "qpos": [],
        "eef": [],
        "qvel": [],
        "effort": [],
        "action": [],
        "images": _empty_image_buffers(camera_names),
        "depth": [],
        "time": [],
        "arm_stamp": [],
    }

    return_home_start: Optional[float] = None
    has_left_init = False
    step = 0

    while rclpy.ok() and step < max_steps:
        loop_start = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)

        for ch in keys.poll():
            if ch in {"s", "\r", "\n"}:
                n = len(buffers["qpos"])
                label = "keyframes" if not record_observations else "frames"
                print(f"Manual stop ({n} {label}).", flush=True)
                return buffers if buffers["qpos"] else None
            if ch in {"q", "\x03"}:
                raise KeyboardInterrupt

        state = bridge.latest_state
        if state is None:
            time.sleep(0.001)
            continue

        rgb_frames = None
        if record_observations:
            rgb_frames = camera.read_all_ready()
            if rgb_frames is None:
                time.sleep(0.001)
                continue

        obs = state_to_obs(state)
        action = obs["qpos"].copy()

        buffers["qpos"].append(obs["qpos"])
        buffers["eef"].append(obs["eef"])
        buffers["qvel"].append(obs["qvel"])
        buffers["effort"].append(obs["effort"])
        buffers["action"].append(action)
        if record_observations and rgb_frames is not None:
            for name in camera_names:
                buffers["images"][name].append(rgb_frames[name])
        buffers["time"].append(time.time())
        buffers["arm_stamp"].append(state.stamp_sec)

        if not all(abs(v - i) <= home_tol for v, i in zip(action, init_pos)):
            has_left_init = True

        if auto_stop and step >= min_steps and (not require_move or has_left_init):
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
        if record_observations and (step == 1 or step % max(1, int(hz)) == 0):
            print(f"  recording... {step} frames", flush=True)
        elif not record_observations and (step == 1 or step % max(1, int(hz)) == 0):
            print(f"  trajectory... {step} keyframes (max {max_steps})", flush=True)
        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, period - elapsed))

    print(f"Reached max_steps={max_steps}.")
    return buffers if buffers["qpos"] else None


def write_hdf5(
    path: Path,
    task: str,
    buffers: Dict[str, List[Any]],
    camera_names: List[str],
    use_depth: bool,
    record_fps: float = 20.0,
) -> None:
    import h5py

    n = len(buffers["qpos"])
    images_dict = buffers["images"]
    if isinstance(images_dict, list):
        images_dict = {camera_names[0]: images_dict}

    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["task"] = str(task)
        root.attrs["state_dim"] = STATE_DIM
        root.attrs["camera_names"] = json.dumps(camera_names)
        root.attrs["fps"] = float(record_fps)

        obs = root.create_group("observations")
        img_grp = obs.create_group("images")
        for cam_name in camera_names:
            frames = np.stack(images_dict[cam_name], axis=0).astype(np.uint8)
            img_grp.create_dataset(cam_name, data=frames, compression="gzip", compression_opts=4)

        if use_depth and any(d is not None for d in buffers["depth"]):
            depth_grp = obs.create_group("images_depth")
            depth_stack = []
            for d in buffers["depth"]:
                depth_stack.append(np.zeros_like(buffers["depth"][0]) if d is None else d)
            depth_grp.create_dataset(
                camera_names[0],
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
            try:
                from PIL import Image

                Image.fromarray(buffers["images"][i]).save(img_dir / frame_name)
            except ImportError:
                import cv2

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


def episode_has_movement(buffers: Dict[str, List[Any]], home_tol: float = 0.05) -> bool:
    if not buffers["qpos"]:
        return False
    qpos = np.stack(buffers["qpos"], axis=0)
    init = qpos[0]
    return bool(np.any(np.max(np.abs(qpos - init), axis=0) > home_tol))


def _keyframe_indices(t_out: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_rel, t_out, side="right") - 1
    return np.clip(idx, 0, len(t_rel) - 1)


def interpolate_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    out_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Upsample teach keyframes: interpolate arm joints and gripper; step wrist rpy."""
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

    for d in range(min(ARM_JOINT_DIMS, qpos.shape[1])):
        qpos_out[:, d] = np.interp(t_out, t_rel, qpos[:, d])
    if qpos.shape[1] > GRIPPER_JOINT_IDX:
        qpos_out[:, GRIPPER_JOINT_IDX] = np.interp(t_out, t_rel, qpos[:, GRIPPER_JOINT_IDX])

    for d in range(min(3, eef.shape[1])):
        eef_out[:, d] = np.interp(t_out, t_rel, eef[:, d])
    if eef.shape[1] >= 6:
        eef_out[:, 3:6] = eef[idx, 3:6]
    if eef.shape[1] > GRIPPER_JOINT_IDX:
        eef_out[:, GRIPPER_JOINT_IDX] = qpos_out[:, GRIPPER_JOINT_IDX]

    return eef_out, qpos_out, 1.0 / max(out_hz, 1.0)


def upsample_trajectory_keyframes(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    out_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Hold each teach keyframe until the next sample time (no linear blending)."""
    stamps = np.asarray(stamps, dtype=np.float64)
    period = 1.0 / max(out_hz, 1.0)
    if len(stamps) < 2:
        return eef, qpos, period

    t_rel = stamps - stamps[0]
    duration = max(float(t_rel[-1]), period)
    n_out = max(2, int(duration * out_hz) + 1)
    t_out = np.linspace(0.0, duration, n_out)

    idx = np.searchsorted(t_rel, t_out, side="right") - 1
    idx = np.clip(idx, 0, len(qpos) - 1)
    return eef[idx], qpos[idx], period


def _pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _stack_teach_arrays(
    teach_buffers: Dict[str, List[Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eef = np.stack(teach_buffers["eef"], axis=0)
    qpos = np.stack(teach_buffers["qpos"], axis=0)
    stamps = np.asarray(teach_buffers.get("arm_stamp") or teach_buffers["time"], dtype=np.float64)
    return eef, qpos, stamps


def _reverse_teach_stamps(stamps: np.ndarray) -> np.ndarray:
    dts = np.diff(stamps)
    if len(dts) == 0:
        return np.zeros(1, dtype=np.float64)
    rev_dts = dts[::-1]
    out = np.zeros(len(stamps), dtype=np.float64)
    for i in range(1, len(out)):
        out[i] = out[i - 1] + rev_dts[i - 1]
    return out


def _upsample_teach_segment(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    replay_hz: float,
    replay_interp: str,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if replay_interp == "keyframe":
        eef_i, qpos_i, period = upsample_trajectory_keyframes(eef, qpos, stamps, out_hz=replay_hz)
    else:
        eef_i, qpos_i, period = interpolate_trajectory(eef, qpos, stamps, out_hz=replay_hz)
    if smooth_window > 1:
        eef_i, qpos_i = smooth_trajectory(eef_i, qpos_i, window=smooth_window)
    return eef_i, qpos_i, period


def _run_replay_steps(
    bridge: X5ArmBridge,
    eef_i: np.ndarray,
    qpos_i: np.ndarray,
    period: float,
    *,
    progress_label: str = "replay",
    progress_hz: float = 10.0,
) -> None:
    n_play = len(qpos_i)
    for i in range(n_play):
        t0 = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)
        _publish_replay_frame(bridge, eef_i[i], qpos_i[i])
        if i == 0 or i % max(1, int(progress_hz)) == 0:
            print(f"  {progress_label} step {i}/{n_play}", flush=True)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


def go_to_start_unrecorded(
    bridge: X5ArmBridge,
    teach_buffers: Dict[str, List[Any]],
    replay_hz: float,
    replay_interp: str,
    home_tol: float = 0.08,
    settle_sec: float = 2.0,
) -> None:
    """Move directly back to the first teach pose without recording."""
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return

    eef, qpos, _stamps = _stack_teach_arrays(teach_buffers)
    target_qpos = np.asarray(qpos[0], dtype=np.float64)
    target_eef = np.asarray(eef[0], dtype=np.float64)
    current_qpos = np.asarray(state.joint_pos[:STATE_DIM], dtype=np.float64)
    current_eef = np.asarray(state.end_pos[:6], dtype=np.float64)

    dist = _pose_distance(current_qpos, target_qpos)
    period = 1.0 / max(replay_hz, 1.0)
    if dist < 0.03:
        if settle_sec > 0.0:
            print(f"Already at start pose; holding {settle_sec:.1f} s before recording...", flush=True)
            hold_until = time.monotonic() + settle_sec
            while time.monotonic() < hold_until:
                rclpy.spin_once(bridge, timeout_sec=0.0)
                _publish_replay_frame(bridge, target_eef, target_qpos)
                time.sleep(period)
        print("At taught start pose; recording forward replay now.", flush=True)
        return

    return_speed = 0.7  # rad/s, conservative direct joint-space return
    duration = float(np.clip(dist / return_speed, 0.8, 4.0))
    steps = max(2, int(duration * replay_hz))
    print(
        f"Returning directly to taught start pose (NOT recorded): "
        f"max joint delta={dist:.3f} rad, {steps} steps.",
        flush=True,
    )

    for i in range(1, steps + 1):
        alpha = i / steps
        qpos_cmd = current_qpos * (1.0 - alpha) + target_qpos * alpha
        eef_cmd = current_eef * (1.0 - alpha) + target_eef[:6] * alpha
        t0 = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)
        _publish_replay_frame(bridge, eef_cmd, qpos_cmd)
        if i == 1 or i % max(1, int(replay_hz)) == 0 or i == steps:
            print(f"  return-to-start direct {i}/{steps}", flush=True)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    settle_deadline = time.monotonic() + 1.0
    while time.monotonic() < settle_deadline:
        rclpy.spin_once(bridge, timeout_sec=0.0)
        state = bridge.latest_state
        if state is not None:
            measured = np.asarray(state.joint_pos[:STATE_DIM], dtype=np.float64)
            if _pose_distance(measured, target_qpos) <= home_tol:
                break
        _publish_replay_frame(bridge, target_eef, target_qpos)
        time.sleep(period)

    if settle_sec > 0.0:
        print(f"Holding taught start pose for {settle_sec:.1f} s before recording...", flush=True)
        hold_until = time.monotonic() + settle_sec
        while time.monotonic() < hold_until:
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_frame(bridge, target_eef, target_qpos)
            time.sleep(period)

    print("At taught start pose; recording forward replay now.", flush=True)

def add_gripper_event_holds(
    eef: np.ndarray,
    qpos: np.ndarray,
    period: float,
    hold_sec: float = GRIPPER_EVENT_HOLD_SEC,
    gripper_delta: float = GRIPPER_EVENT_DELTA,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Hold arm pose around gripper changes so objects are not released while moving."""
    if hold_sec <= 0.0 or len(qpos) < 2 or qpos.shape[1] <= GRIPPER_JOINT_IDX:
        return eef, qpos, 0

    hold_steps = max(1, int(round(hold_sec / max(period, 1e-6))))
    eef_out: List[np.ndarray] = [eef[0].copy()]
    qpos_out: List[np.ndarray] = [qpos[0].copy()]
    events = 0

    for i in range(1, len(qpos)):
        prev = qpos[i - 1]
        cur = qpos[i]
        gripper_jump = abs(float(cur[GRIPPER_JOINT_IDX] - prev[GRIPPER_JOINT_IDX]))
        if gripper_jump >= gripper_delta:
            events += 1
            hold_prev = prev.copy()
            hold_cur = cur.copy()
            hold_cur[:ARM_JOINT_DIMS] = hold_prev[:ARM_JOINT_DIMS]
            for _ in range(hold_steps):
                eef_out.append(eef[i - 1].copy())
                qpos_out.append(hold_prev.copy())
            for _ in range(hold_steps):
                eef_out.append(eef[i - 1].copy())
                qpos_out.append(hold_cur.copy())
            for _ in range(hold_steps):
                eef_out.append(eef[i - 1].copy())
                qpos_out.append(hold_cur.copy())

        eef_out.append(eef[i].copy())
        qpos_out.append(cur.copy())

    return np.stack(eef_out, axis=0), np.stack(qpos_out, axis=0), events

def prepare_replay_trajectory(
    teach_buffers: Dict[str, List[Any]],
    replay_hz: float,
    replay_interp: str,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Build forward-only replay from taught keyframes."""
    eef, qpos, stamps = _stack_teach_arrays(teach_buffers)

    notes: List[str] = ["forward teach path"]
    if REPLAY_TIME_SCALE > 1.0 and len(stamps) > 1:
        stamps = stamps[0] + (stamps - stamps[0]) * REPLAY_TIME_SCALE
        notes.append(f"time scale={REPLAY_TIME_SCALE:.1f}x")
    eef_i, qpos_i, period = _upsample_teach_segment(
        eef, qpos, stamps, replay_hz, replay_interp, smooth_window
    )
    if replay_interp == "keyframe":
        notes.append("keyframe hold")
    else:
        notes.append("linear interpolation")
    if smooth_window > 1:
        notes.append(f"smooth window={smooth_window}")

    eef_i, qpos_i, gripper_events = add_gripper_event_holds(eef_i, qpos_i, period)
    if gripper_events:
        notes.append(f"gripper holds={gripper_events}x{GRIPPER_EVENT_HOLD_SEC:.1f}s")

    return eef_i, qpos_i, period, "; ".join(notes)


def smooth_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth arm joint paths only; leave gripper and wrist rpy unchanged."""
    if window <= 1 or len(qpos) < 3:
        return eef, qpos

    win = min(window, len(qpos))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return eef, qpos

    pad = win // 2
    kernel = np.ones(win, dtype=np.float64) / win

    def _smooth_cols(data: np.ndarray, cols: slice) -> np.ndarray:
        out = data.copy()
        for d in range(cols.start or 0, cols.stop or data.shape[1]):
            padded = np.pad(data[:, d], (pad, pad), mode="edge")
            out[:, d] = np.convolve(padded, kernel, mode="valid")
        return out

    qpos_out = _smooth_cols(qpos, slice(0, ARM_JOINT_DIMS))
    eef_out = _smooth_cols(eef, slice(0, 3))
    return eef_out, qpos_out


def _replay_blend_to_start(
    bridge: X5ArmBridge,
    eef_i: np.ndarray,
    qpos_i: np.ndarray,
    period: float,
    blend_sec: float,
) -> None:
    if blend_sec <= 0.0:
        return

    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return

    cur_eef = np.asarray(state.end_pos[:6], dtype=np.float64)
    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    dist = _pose_distance(cur_qpos, qpos_i[0])
    if dist < 0.03:
        return

    blend_steps = max(3, int(blend_sec / period))
    for k in range(1, blend_steps + 1):
        alpha = k / blend_steps
        eef_b = cur_eef * (1 - alpha) + eef_i[0, :6] * alpha
        qpos_b = cur_qpos * (1 - alpha) + qpos_i[0] * alpha
        t0 = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)
        _publish_replay_frame(bridge, eef_b, qpos_b)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


def _publish_replay_joint(
    bridge: X5ArmBridge,
    qpos_row: np.ndarray,
) -> None:
    """Replay using absolute joint positions."""
    state = bridge.latest_state
    xyzrpy = state.end_pos.tolist() if state is not None else [0.0] * 6
    bridge.publish_command(
        ArmCommand(
            xyzrpy=xyzrpy,
            gripper=float(qpos_row[6]),
            joint_pos=qpos_row[:6].tolist(),
            mode=MODE_POSITION_CONTROL,
        )
    )


def _publish_replay_frame(
    bridge: X5ArmBridge,
    eef_row: np.ndarray,
    qpos_row: np.ndarray,
) -> None:
    """Replay a taught frame in joint position mode.

    The controller mode=4 is END_CONTROL. Kinesthetic replay needs
    mode=5 (POSITION_CONTROL), otherwise the arm follows IK/end_pos
    and can drift away from the hand-taught joint trajectory.
    """
    bridge.publish_command(
        ArmCommand(
            xyzrpy=eef_row[:6].tolist(),
            gripper=float(qpos_row[6]),
            joint_pos=qpos_row[:6].tolist(),
            mode=MODE_POSITION_CONTROL,
        )
    )


def replay_and_record(
    bridge: X5ArmBridge,
    camera: MultiRealSenseSource,
    camera_names: List[str],
    teach_buffers: Dict[str, List[Any]],
    replay_hz: float,
    record_hz: float,
    replay_interp: str = "linear",
    replay_blend_sec: float = 0.0,
    home_tol: float = 0.08,
    replay_smooth: int = 1,
) -> Optional[Dict[str, List[Any]]]:
    """Replay taught trajectory and record camera + state during autonomous motion."""
    n_teach = len(teach_buffers["qpos"])
    if n_teach == 0:
        return None

    eef_i, qpos_i, replay_period, replay_note = prepare_replay_trajectory(
        teach_buffers,
        replay_hz=replay_hz,
        replay_interp=replay_interp,
        smooth_window=replay_smooth,
    )
    n_play = len(eef_i)

    record_buffers: Dict[str, List[Any]] = {
        "qpos": [],
        "eef": [],
        "action": [],
        "images": _empty_image_buffers(camera_names),
        "time": [],
        "arm_stamp": [],
    }
    record_interval = 1.0 / max(record_hz, 1.0)
    next_record_at = time.monotonic()

    print(
        f"Auto replay+record: {n_teach} teach keyframes -> {n_play} forward replay steps "
        f"(~{n_play / max(replay_hz, 1):.1f}s at {replay_hz:.0f} Hz), record {record_hz:.0f} Hz. "
        f"{replay_note}. Ctrl+C to abort.",
        flush=True,
    )
    g = np.stack(teach_buffers["qpos"], axis=0)[:, GRIPPER_JOINT_IDX]
    print(
        f"  gripper teach range: [{g.min():.3f}, {g.max():.3f}] "
        f"(joint index {GRIPPER_JOINT_IDX}, not a wrong mapping)",
        flush=True,
    )

    enter_position_control(bridge)
    time.sleep(0.3)

    go_to_start_unrecorded(
        bridge,
        teach_buffers,
        replay_hz=replay_hz,
        replay_interp=replay_interp,
        home_tol=home_tol,
        settle_sec=2.0,
    )

    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_frame(bridge, eef_i[i], qpos_i[i])
            if i == 0 or i % max(1, int(replay_hz)) == 0:
                print(f"  forward replay step {i}/{n_play}", flush=True)

            now = time.monotonic()
            if now >= next_record_at:
                arm_state = bridge.latest_state
                rgb_frames = camera.read_all_ready()
                if arm_state is not None and rgb_frames is not None:
                    obs = state_to_obs(arm_state)
                    cmd = np.asarray(qpos_i[i], dtype=np.float64)
                    # Some X5 firmware reports gripper state as binary. Store the
                    # continuous commanded gripper value so demos are not just 0/1.
                    obs["qpos"][GRIPPER_JOINT_IDX] = cmd[GRIPPER_JOINT_IDX]
                    obs["eef"][GRIPPER_JOINT_IDX] = cmd[GRIPPER_JOINT_IDX]
                    record_buffers["qpos"].append(obs["qpos"])
                    record_buffers["eef"].append(obs["eef"])
                    record_buffers["action"].append(cmd)
                    for name in camera_names:
                        record_buffers["images"][name].append(rgb_frames[name])
                    record_buffers["time"].append(time.time())
                    record_buffers["arm_stamp"].append(arm_state.stamp_sec)
                    n_rec = len(record_buffers["qpos"])
                    if n_rec == 1 or n_rec % max(1, int(record_hz)) == 0:
                        print(
                            f"  replay-record {n_rec} frames (replay step {i}/{n_play})",
                            flush=True,
                        )
                next_record_at += record_interval

            elapsed = time.monotonic() - t0
            if elapsed < replay_period:
                time.sleep(replay_period - elapsed)
    except KeyboardInterrupt:
        print("Replay-record aborted.", flush=True)
    else:
        print(f"Forward replay finished ({n_play} steps).", flush=True)
    finally:
        enter_gravity_compensation(bridge)
        print("Back to drag mode (mode=3).", flush=True)

    if record_buffers["qpos"]:
        measured = np.stack(record_buffers["qpos"], axis=0)
        commanded = np.stack(record_buffers["action"], axis=0)
        tracking_err = np.max(np.abs(measured[:, :STATE_DIM] - commanded[:, :STATE_DIM]), axis=0)
        print(
            "Replay joint tracking max abs error: "
            f"{tracking_err.round(4).tolist()} rad",
            flush=True,
        )
        grip = commanded[:, GRIPPER_JOINT_IDX]
        unique_grip = np.unique(np.round(grip, 3))
        print(
            "Replay gripper command range: "
            f"[{grip.min():.3f}, {grip.max():.3f}], "
            f"unique values (rounded 0.001)={len(unique_grip)}",
            flush=True,
        )

    return record_buffers if record_buffers["qpos"] else None


def replay_trajectory(
    bridge: X5ArmBridge,
    buffers: Dict[str, List[Any]],
    hz: float,
    record_hz: float,
    replay_interp: str = "linear",
    replay_blend_sec: float = 0.0,
    home_tol: float = 0.08,
    replay_smooth: int = 1,
) -> None:
    """Replay with timestamp upsampling + recorded joint/EEF targets."""
    n = len(buffers["qpos"])
    if n == 0:
        print("Nothing to replay.")
        return

    eef_i, qpos_i, period, replay_note = prepare_replay_trajectory(
        buffers,
        replay_hz=hz,
        replay_interp=replay_interp,
        smooth_window=replay_smooth,
    )
    n_play = len(eef_i)
    print(
        f"Replaying {n} keyframes -> {n_play} forward steps at {hz:.0f} Hz "
        f"(recorded ~{record_hz:.0f} Hz; {replay_note}). Ctrl+C to abort."
    )

    enter_position_control(bridge)
    time.sleep(0.3)

    go_to_start_unrecorded(
        bridge,
        buffers,
        replay_hz=hz,
        replay_interp=replay_interp,
        home_tol=home_tol,
        settle_sec=2.0,
    )
    _replay_blend_to_start(bridge, eef_i, qpos_i, period, replay_blend_sec)

    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_frame(bridge, eef_i[i], qpos_i[i])
            if i % max(1, int(hz)) == 0:
                print(f"  hdf5 {i}/{n_play}")
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Replay aborted.")
    finally:
        enter_gravity_compensation(bridge)
        print("Back to drag mode (mode=3).")

    print("Replay finished.")


def interpolate_joint_trajectory(
    qpos: np.ndarray,
    stamps: np.ndarray,
    out_hz: float,
) -> tuple[np.ndarray, float]:
    placeholder_eef = np.zeros((len(qpos), 6), dtype=np.float64)
    _, qpos_i, period = interpolate_trajectory(placeholder_eef, qpos, stamps, out_hz=out_hz)
    return qpos_i, period


def _run_joint_replay_loop(
    bridge: X5ArmBridge,
    qpos_i: np.ndarray,
    period: float,
    hz: float,
    label: str,
) -> None:
    n_play = len(qpos_i)
    enter_position_control(bridge)
    time.sleep(0.3)

    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is not None:
        cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
        blend_steps = max(5, int(hz * 0.4))
        for k in range(1, blend_steps + 1):
            alpha = k / blend_steps
            qpos_b = cur_qpos * (1 - alpha) + qpos_i[0] * alpha
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_joint(bridge, qpos_b)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_joint(bridge, qpos_i[i])
            if i % max(1, int(hz)) == 0:
                print(f"  {label} {i}/{n_play}")
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Replay aborted.")
    finally:
        enter_gravity_compensation(bridge)
        print("Back to drag mode (mode=3).")

    print("Replay finished.")


def replay_lerobot_trajectory(
    bridge: X5ArmBridge,
    parquet_path: Path,
    hz: float,
    hdf5_path: Optional[Path] = None,
) -> None:
    """Replay LeRobot parquet action; EEF pose from HDF5 (controller needs both)."""
    data = load_parquet_episode(parquet_path)
    action = data["action"]
    stamps = data["timestamp"]
    n = len(action)
    if n == 0:
        print("Nothing to replay (empty parquet).")
        return

    if hdf5_path is not None and hdf5_path.exists():
        diffs = compare_hdf5_parquet(hdf5_path, parquet_path)
        print(
            "LeRobot conversion check: "
            f"max|hdf5.action - parquet.action| = {diffs['action_max_abs_diff']:.2e}, "
            f"max|hdf5.qpos - parquet.state| = {diffs['state_max_abs_diff']:.2e}"
        )
        ep = load_hdf5_episode(hdf5_path)
        eef = ep["eef"].astype(np.float64)
        stamps = ep["stamps"]
    else:
        print(
            "Warning: no HDF5 for EEF pose; controller may move incorrectly. "
            "LeRobot replay after record always has HDF5."
        )
        eef = np.zeros((n, 7), dtype=np.float64)

    eef_i, qpos_i, period = interpolate_trajectory(eef, action, stamps, out_hz=hz)
    n_play = len(qpos_i)
    print(
        f"LeRobot replay: {n} frames -> {n_play} smooth steps at {hz:.0f} Hz "
        f"(parquet action + HDF5 EEF). Ctrl+C to abort."
    )

    enter_position_control(bridge)
    time.sleep(0.3)

    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is not None:
        cur_eef = np.asarray(state.end_pos[:6], dtype=np.float64)
        cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
        blend_steps = max(5, int(hz * 0.4))
        for k in range(1, blend_steps + 1):
            alpha = k / blend_steps
            eef_b = cur_eef * (1 - alpha) + eef_i[0, :6] * alpha
            qpos_b = cur_qpos * (1 - alpha) + qpos_i[0] * alpha
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_frame(bridge, eef_b, qpos_b)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            _publish_replay_frame(bridge, eef_i[i], qpos_i[i])
            if i % max(1, int(hz)) == 0:
                print(f"  lerobot {i}/{n_play}")
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Replay aborted.")
    finally:
        enter_gravity_compensation(bridge)
        print("Back to drag mode (mode=3).")

    print("Replay finished.")


def _lerobot_test_parquet_path(output_dir: Path, episode_idx: int) -> Path:
    return output_dir / ".lerobot_replay_test" / f"episode_{episode_idx:06d}.parquet"


ReplayChoice = Literal["skip", "hdf5", "lerobot"]


def wait_for_replay_prompt(
    keys: KeyPoller,
    timeout_sec: float = 30.0,
    default: ReplayChoice = "skip",
) -> ReplayChoice:
    print(
        "\n--- Episode saved ---\n"
        "  r = 鍘熺増 HDF5 鍥炴斁锛堝惈 EEF + 鍏宠妭锛塡n"
        "  l = LeRobot parquet 鍥炴斁锛坧arquet 鍏宠妭 + HDF5 EEF锛岄獙璇佽浆鎹級\n"
        "  Enter / 鍏朵粬閿?= 璺宠繃锛屽綍涓嬩竴鏉n"
        f"  ({int(timeout_sec)}s 鍐呮棤鎿嶄綔鍒欒烦杩?"
    )
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        for ch in keys.poll():
            if ch in {"r", "R"}:
                return "hdf5"
            if ch in {"l", "L"}:
                return "lerobot"
            if ch in {"\r", "\n", " ", "0", "q"}:
                return "skip"
        time.sleep(0.05)
    return default if default in {"hdf5", "lerobot"} else "skip"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record kinesthetic teaching demos (single X5 + D405).")
    p.add_argument(
        "--output-dir",
        default=None,
        help="hdf5: episode dir. lerobot: runs root (default data/arx_x5_real/runs).",
    )
    p.add_argument("--task", default="grab the bottle", help="Task description stored in metadata.")
    p.add_argument("--hz", type=float, default=20.0, help="Recording sample rate during replay. Default: 20")
    p.add_argument("--num-episodes", type=int, default=-1, help="Stop after N episodes. -1 = unlimited.")
    p.add_argument("--max-steps", type=int, default=2000, help="Max teach keyframes per episode (default 2000).")
    p.add_argument(
        "--format",
        choices=("lerobot", "hdf5", "dir"),
        default="lerobot",
        help="lerobot: replay-record to dated LeRobot run (default). hdf5: legacy manual save.",
    )
    p.add_argument("--use-depth", action="store_true", help="Store depth in HDF5 (if D405 provides it).")
    p.add_argument("--no-auto-stop", action="store_true", help="Disable return-to-init auto stop.")
    p.add_argument(
        "--min-steps",
        type=int,
        default=30,
        help="Min frames before auto-stop; episodes without joint movement are discarded.",
    )
    p.add_argument("--no-require-move", action="store_true", help="Allow auto-stop/save without leaving init pose.")
    p.add_argument("--enter-gravity", action="store_true", help="Publish mode=3 on start (normal launch).")
    p.add_argument("--no-replay-prompt", action="store_true", help="Do not ask to replay after each episode.")
    p.add_argument(
        "--replay-source",
        choices=("ask", "hdf5", "lerobot"),
        default="ask",
        help="After save: ask (r/l), or always replay hdf5 / lerobot parquet.",
    )
    p.add_argument("--replay-hz", type=float, default=50.0, help="Playback control rate (interpolated). Default: 50")
    p.add_argument(
        "--replay-interp",
        choices=("keyframe", "linear"),
        default="linear",
        help="linear: smooth replay (default). keyframe: step-wise, may jitter.",
    )
    p.add_argument(
        "--replay-blend-sec",
        type=float,
        default=0.0,
        help="Extra blend after returning to start (0 to disable). Default: 0",
    )
    p.add_argument(
        "--replay-smooth",
        type=int,
        default=1,
        help="Moving-average window on replay commands (1=off). Default: 1",
    )
    p.add_argument("--no-camera-display", action="store_true", help="Disable camera preview.")
    p.add_argument("--camera-width", type=int, default=480)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=20)
    p.add_argument("--wrist-serial", default=None, help="D405 serial (auto-detect if omitted).")
    p.add_argument("--desktop-serial", default=None, help="D435i desktop serial (auto-detect).")
    p.add_argument(
        "--no-desktop-camera",
        action="store_true",
        help="Record wrist cam only (default: use D435i if connected).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_parent = Path(__file__).resolve().parent.parent
    default_runs_root = script_parent / "data" / "arx_x5_real" / "runs"
    default_teach_dir = script_parent / "data" / "teach_demos"

    if args.format == "lerobot":
        runs_root = Path(args.output_dir or default_runs_root).expanduser().resolve()
        enable_desktop = not args.no_desktop_camera
        cameras = build_record_cameras(
            wrist_serial=args.wrist_serial,
            desktop_serial=args.desktop_serial,
            enable_desktop=enable_desktop,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            display=not args.no_camera_display,
        )
        camera_names = cameras.camera_names
        lerobot_writer = LeRobotRunWriter.create_dated_run(
            runs_root, fps=args.hz, camera_names=camera_names
        )
        output_dir = lerobot_writer.run_dir
    else:
        cameras = build_record_cameras(
            wrist_serial=args.wrist_serial,
            desktop_serial=args.desktop_serial,
            enable_desktop=not args.no_desktop_camera,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            display=not args.no_camera_display,
        )
        camera_names = cameras.camera_names
        output_dir = Path(args.output_dir or default_teach_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        lerobot_writer = None

    rclpy.init()
    bridge = X5ArmBridge()

    episode_idx = 0
    delete_requested = False

    print("=" * 60)
    print("X5 teach + replay recorder")
    print(f"  format : {args.format}")
    print(f"  output : {output_dir}")
    print(f"  task   : {args.task}")
    print(f"  record hz (replay): {args.hz}")
    print(f"  replay hz: {args.replay_hz}")
    if args.format == "lerobot":
        print("  flow   : drag (no save) -> s -> auto replay -> LeRobot save")
    print(f"  cameras: {', '.join(camera_names)}")
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

                if isinstance(start_signal, str):
                    if start_signal == "quit":
                        break
                    if start_signal == "delete":
                        prev = episode_idx - 1
                        if prev < 0:
                            print("No episode to delete.")
                            continue
                        if args.format == "lerobot":
                            if lerobot_writer and lerobot_writer.delete_last_episode():
                                episode_idx = prev
                                print(f"Deleted last LeRobot episode from {output_dir}", flush=True)
                            else:
                                print("No LeRobot episode to delete.")
                        elif args.format == "hdf5":
                            target = output_dir / f"episode_{prev}.hdf5"
                            if target.exists():
                                target.unlink()
                                episode_idx = prev
                                print(f"Deleted {target}")
                            else:
                                print(f"Not found: {target}")
                        else:
                            target = output_dir / f"episode_{prev:03d}"
                            if target.exists():
                                import shutil

                                shutil.rmtree(target)
                                episode_idx = prev
                                print(f"Deleted {target}")
                            else:
                                print(f"Not found: {target}")
                        continue
                    raise ValueError(f"Unknown start signal: {start_signal!r}")

                init_pos = start_signal
                teach_buffers = collect_episode(
                    bridge=bridge,
                    camera=cameras,
                    camera_names=camera_names,
                    keys=keys,
                    init_pos=init_pos,
                    hz=args.hz,
                    max_steps=args.max_steps,
                    auto_stop=not args.no_auto_stop,
                    home_tol=0.05,
                    home_hold_sec=2.0,
                    min_steps=args.min_steps,
                    require_move=not args.no_require_move,
                    record_observations=(args.format != "lerobot"),
                )

                if not teach_buffers or not teach_buffers["qpos"]:
                    print("Empty trajectory, not saved.")
                    continue

                if not args.no_require_move and not episode_has_movement(teach_buffers):
                    print(
                        "No joint movement detected (you did not drag the arm). "
                        "Discarded. Press 0 and move the arm."
                    )
                    continue

                if len(teach_buffers["qpos"]) < args.min_steps:
                    print(
                        f"Trajectory too short ({len(teach_buffers['qpos'])} < {args.min_steps}). "
                        "Discarded."
                    )
                    continue

                if args.format == "lerobot":
                    assert lerobot_writer is not None
                    print("Trajectory OK. Starting auto replay + record...", flush=True)
                    record_buffers = replay_and_record(
                        bridge=bridge,
                        camera=cameras,
                        camera_names=camera_names,
                        teach_buffers=teach_buffers,
                        replay_hz=args.replay_hz,
                        record_hz=args.hz,
                        replay_interp=args.replay_interp,
                        replay_blend_sec=args.replay_blend_sec,
                        replay_smooth=args.replay_smooth,
                    )
                    if not record_buffers or len(record_buffers["qpos"]) < args.min_steps:
                        n = 0 if not record_buffers else len(record_buffers["qpos"])
                        print(
                            f"Replay-record too short ({n} < {args.min_steps}). Episode discarded.",
                            flush=True,
                        )
                        continue

                    print("Saving LeRobot episode...", flush=True)
                    lerobot_writer.append_episode(args.task, record_buffers)
                    episode_idx += 1
                    print("\n" + "=" * 60, flush=True)
                    print(f"LeRobot saved: {output_dir}", flush=True)
                    print(f"  episode {episode_idx - 1}, {len(record_buffers['qpos'])} frames", flush=True)
                    print("=" * 60 + "\n", flush=True)
                    continue

                saved_idx = episode_idx
                if args.format == "hdf5":
                    path = output_dir / f"episode_{episode_idx}.hdf5"
                    write_hdf5(path, args.task, teach_buffers, camera_names, use_depth=args.use_depth, record_fps=args.hz)
                else:
                    path = output_dir / f"episode_{episode_idx:03d}"
                    write_jsonl_episode(path, args.task, teach_buffers)

                episode_idx += 1

                if args.no_replay_prompt:
                    continue

                if args.replay_source == "ask":
                    choice = wait_for_replay_prompt(keys)
                else:
                    choice = args.replay_source  # type: ignore[assignment]

                if choice == "skip":
                    continue
                if choice == "hdf5":
                    replay_trajectory(
                        bridge,
                        teach_buffers,
                        hz=args.replay_hz,
                        record_hz=args.hz,
                        replay_interp=args.replay_interp,
                        replay_blend_sec=args.replay_blend_sec,
                        replay_smooth=args.replay_smooth,
                    )
                elif choice == "lerobot":
                    parquet_path = _lerobot_test_parquet_path(output_dir, saved_idx)
                    try:
                        hdf5_to_parquet(path, parquet_path, fps=args.hz)
                        print(f"LeRobot test parquet: {parquet_path}")
                    except SystemExit as e:
                        print(f"LeRobot convert failed: {e}")
                        continue
                    replay_lerobot_trajectory(
                        bridge,
                        parquet_path,
                        hz=args.replay_hz,
                        hdf5_path=path,
                    )

    finally:
        cameras.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
