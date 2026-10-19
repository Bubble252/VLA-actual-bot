#!/usr/bin/env python3
"""U-Arm master -> ARX X5 teleoperation with integrated record / replay session.

Prerequisites:
  1. CAN: bash ARX_CAN/arx_can/arx_can1.sh
  2. Normal control: ros2 launch arx_x5_controller open_single_arm.launch.py
  3. U-Arm master on serial (default /dev/ttyUSB0)
  4. This script

Session controls (terminal focused):
  0 / Space  start recording
  s / Enter  stop and save episode -> post-save menu
  x / Esc    abort current recording (discard, back to ready for Space)
  d          delete last saved episode (when not recording)

Recording: joint teleop (mode=5); HDF5 stores action (joints) + observations/eef + qpos.

After each save:
  r          replay saved HDF5 (joint action, mode=5)
  p          replay HDF5 (observations/eef, mode=4 damped IK)
  v          replay EEF via differential IK → joints (mode=5; avoids mode=4)
  l          LeRobot-style replay (parquet action + HDF5 EEF)
  n          record a new episode
  q          quit

Quick-record mode (--quick-record / record.quick_record):
  After save, reset to the same go_home as startup and hold (no follow).
  Press n / 0 / Space / Enter once to go_home+sync again and start the next
  episode. m = full menu. Prevents pose drift across continuous collection.

HDF5 writes run on a background pipeline (record.async_save, default on) so you
can start the next episode while the previous one is still compressing to disk.

Use --teleop-only to disable recording and post-save menu.
Use --replay-file to open the post-save menu on an existing HDF5 (compare r / p / v).
"""

from __future__ import annotations

import argparse
import json
import queue
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from episode_replay import (
    ensure_eef_fk_baked,
    finish_replay_handoff,
    prepare_before_recording,
    replay_hdf5_file,
    replay_lerobot_from_hdf5,
    startup_blend_gripper_open,
)
from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.image_source import CameraPoller, MultiRealSenseSource, build_teleop_cameras_from_config, warmup_cameras
from lewm_x5_grasp.schemas import ArmState
from lewm_x5_grasp.uarm_mapper import TeleopCommand, UArmMappingConfig, UArmToX5Mapper
from lewm_x5_grasp.uarm_servo import UArmServoReader


STATE_DIM = 7
WRIST_CAM = "cam_wrist"
DEFAULT_LEROBOT_RUNS_REL = Path("data/arx_x5_real/runs")


def _path_display(path: Path, project_dir: Path) -> str:
    try:
        return str(path.relative_to(project_dir))
    except ValueError:
        return str(path)


def _lerobot_runs_root(project_dir: Path) -> Path:
    return (project_dir / DEFAULT_LEROBOT_RUNS_REL).resolve()


def _lerobot_convert_command(output_dir: Path, project_dir: Path, record_fps: float) -> str:
    return (
        f"bash scripts/run_convert_to_lerobot.sh "
        f"{_path_display(output_dir, project_dir)} --fps {int(record_fps)}"
    )


def _next_episode_index(output_dir: Path) -> int:
    return len(list(output_dir.glob("episode_*.hdf5")))


def _make_episode_save_path(output_dir: Path, *, use_timestamp: bool, episode_idx: int) -> Path:
    if use_timestamp:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base = output_dir / f"episode_{ts}.hdf5"
        if not base.exists():
            return base
        for i in range(1, 1000):
            candidate = output_dir / f"episode_{ts}_{i:03d}.hdf5"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Too many episodes saved in the same second under {output_dir}")
    return output_dir / f"episode_{episode_idx}.hdf5"


def _resolve_hdf5_path(path_str: str, *, output_dir: Path, project_dir: Path) -> Path:
    raw = Path(path_str).expanduser()
    candidates = [
        raw if raw.is_absolute() else None,
        raw if raw.exists() else None,
        output_dir / raw.name if not raw.is_absolute() else None,
        output_dir / raw if not raw.is_absolute() else None,
        project_dir / raw if not raw.is_absolute() else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    if raw.is_absolute():
        return raw.resolve()
    return (output_dir / raw).resolve()


def print_data_storage_info(
    *,
    output_dir: Path,
    project_dir: Path,
    camera_names: List[str],
    record_fps: float,
) -> None:
    lerobot_root = _lerobot_runs_root(project_dir)
    hdf5_glob = f"{_path_display(output_dir, project_dir)}/episode_<时间戳>.hdf5"
    if camera_names:
        cam_lines = [f"      observations/images/{name}" for name in camera_names]
        cam_text = "\n".join(cam_lines)
    else:
        cam_text = "      （未启用相机，--no-camera）"

    print("\n" + "=" * 60, flush=True)
    print("【数据保存路径】", flush=True)
    print("  1) HDF5 原始采集（本脚本 s/Enter 保存，含关节 action + observations/eef + 相机 RGB）", flush=True)
    print(f"     目录: {output_dir}", flush=True)
    print(f"     文件: {hdf5_glob}", flush=True)
    print("     相机数据在 HDF5 内:", flush=True)
    print(cam_text, flush=True)
    print("  2) LeRobot 训练集（退出后单独转换，mp4 + parquet + meta）", flush=True)
    print(f"     目录: {lerobot_root}/YYYY-MM-DD_HHMMSS/", flush=True)
    print(f"     快捷链: {lerobot_root}/latest", flush=True)
    print("     视频: videos/chunk-000/observation.images.<cam>/episode_*.mp4", flush=True)
    print("     关节: data/chunk-000/episode_*.parquet", flush=True)
    print("  3) 转换命令（采集完成后在新终端执行）:", flush=True)
    print(f"     {_lerobot_convert_command(output_dir, project_dir, record_fps)}", flush=True)
    print("=" * 60 + "\n", flush=True)


def print_episode_saved(
    path: Path,
    n: int,
    *,
    camera_names: List[str],
    project_dir: Path,
    output_dir: Path,
    record_fps: float,
) -> None:
    print(f"\nSaved {n} steps -> {path}", flush=True)
    if camera_names:
        cams = ", ".join(f"observations/images/{name}" for name in camera_names)
        print(f"  相机 RGB ({len(camera_names)} 路): {cams}", flush=True)
    print(f"  HDF5 目录: {output_dir}", flush=True)
    print(
        f"  转 LeRobot: {_lerobot_convert_command(output_dir, project_dir, record_fps)}",
        flush=True,
    )


def _empty_image_buffers(camera_names: List[str]) -> Dict[str, List[Any]]:
    return {name: [] for name in camera_names}


STOP_KEYS = frozenset({"s", "S", "\r", "\n", "e", "E"})
START_KEYS = frozenset({"0", " "})
ABORT_KEYS = frozenset({"x", "X", "\x1b"})  # discard in-progress episode
MENU_KEYS = frozenset({"r", "R", "p", "P", "v", "V", "l", "L", "n", "N", "q", "Q", "\x03"})
# After save in quick-record: one of these starts the next episode immediately.
QUICK_NEXT_KEYS = frozenset({"n", "N", "0", " ", "\r", "\n"})


class KeyPoller:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
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
        self._pending.extend(keys)
        out = self._pending[:]
        self._pending.clear()
        return out

    def drain_crlf(self) -> None:
        """Drop leftover Enter key bytes so one press does not fire twice."""
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch not in "\r\n":
                self._pending.append(ch)
                break

    def drain_all(self) -> List[str]:
        """Read and discard all pending stdin (including kernel buffer)."""
        drained: List[str] = list(self._pending)
        self._pending.clear()
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch:
                drained.append(ch)
        return drained


def _cv2_key_to_char(code: int) -> Optional[str]:
    if code < 0 or code == 255:
        return None
    if code in (13, 10):
        return "\r"
    if code == 27:
        return "\x1b"  # Esc -> abort while recording
    if 0 < code < 256:
        return chr(code)
    return None


def _collect_keys(poller: KeyPoller, cv_key: int) -> List[str]:
    keys = poller.poll()
    ch = _cv2_key_to_char(cv_key)
    if ch is not None:
        keys.append(ch)
    return keys


def _wants_stop(keys: List[str]) -> bool:
    return any(k in STOP_KEYS for k in keys)


def _wants_abort(keys: List[str]) -> bool:
    return any(k in ABORT_KEYS for k in keys)


def _key_label(ch: str) -> str:
    if ch in "\r\n":
        return "Enter"
    if ch == " ":
        return "Space"
    if ch == "\x03":
        return "Ctrl+C"
    if ch == "\x1b":
        return "Esc"
    return ch


def _count_meaningful_keys(keys: List[str]) -> int:
    return sum(1 for ch in keys if ch not in "\r\n")


def _pick_first_menu_key(keys: List[str]) -> Optional[str]:
    for ch in keys:
        if ch in MENU_KEYS:
            return ch
    return None


def _warn_ignored_keys(keys: KeyPoller, context: str) -> None:
    drained = keys.drain_all()
    n = _count_meaningful_keys(drained)
    if n:
        labels = [_key_label(c) for c in drained if c not in "\r\n"][:6]
        suffix = "…" if n > 6 else ""
        print(
            f">> [{context}] 已忽略积压按键 ({n}): {', '.join(labels)}{suffix}",
            flush=True,
        )


def publish_teleop_command(
    bridge: X5ArmBridge,
    command: "TeleopCommand",
    *,
    joint_control_mode: int,
) -> None:
    """Joint-space teleop only (mode=5). EEF is recorded from /arm_status, not used for control."""
    bridge.publish_joint_command(
        joint_pos_6=command.joint_pos_rad.tolist(),
        gripper=command.gripper,
        mode=joint_control_mode,
    )


def _snapshot_arm_qpos(bridge: X5ArmBridge) -> Optional[np.ndarray]:
    state = bridge.latest_state
    if state is None:
        return None
    return np.asarray(state.joint_pos[:7], dtype=np.float64).copy()


def _publish_hold_qpos(
    bridge: X5ArmBridge,
    qpos7: np.ndarray,
    *,
    joint_control_mode: int,
) -> None:
    bridge.publish_joint_command(
        joint_pos_6=qpos7[:6].tolist(),
        gripper=float(qpos7[6]),
        mode=joint_control_mode,
    )


def _should_follow_uarm(
    *,
    enable_session: bool,
    menu_pending_path: Optional[Path],
    follow_during_menu: bool,
    replaying: bool = False,
    quick_awaiting_next: bool = False,
) -> bool:
    if replaying:
        return False
    if not enable_session:
        return True
    # Between quick-record episodes: hold go_home (same as after startup), do not follow.
    if quick_awaiting_next:
        return False
    if menu_pending_path is not None and not follow_during_menu:
        return False
    return True


def _run_replay_action(
    keys: KeyPoller,
    *,
    bridge: X5ArmBridge,
    mapper: UArmToX5Mapper,
    servo: UArmServoReader,
    hdf5_path: Path,
    action: str,
    replay_hz: float,
    record_hz: float,
    camera_names: List[str],
    project_dir: Path,
    output_dir: Path,
    replay_smooth_window: int = 7,
    replay_blend_sec: float = 0.5,
    replay_smooth_gripper: bool = False,
    replay_max_blend_sec: float = 2.0,
    replay_blend_skip_tol: float = 0.05,
    replay_blend_speed: float = 0.8,
    replay_post_hold_sec: float = 1.0,
    replay_sync_uarm: bool = True,
    replay_eef_smooth_window: int = 15,
    replay_eef_max_rpy_step_rad: float = 0.04,
    replay_eef_ik_seed: str = "action",
    replay_orientation_interp: str = "slerp",
    replay_eef_source: str = "auto",
    replay_eef_ik_seed_smooth_window: int = 15,
    replay_eef_quat_smooth_window: int = 7,
    replay_auto_bake_eef_fk: bool = False,
    replay_bake_hz: float = 100.0,
    replay_bake_smooth_window: int = 15,
    replay_bake_quat_smooth_window: int = 7,
    replay_return_to_start: bool = True,
    replay_return_blend_sec: float = 1.5,
    replay_return_max_blend_sec: float = 6.0,
    replay_return_settle_sec: float = 1.5,
    replay_return_hold_sec: float = 1.0,
    replay_return_tol_rad: float = 0.04,
    replay_start_max_blend_sec: float = 6.0,
    replay_start_settle_sec: float = 0.8,
    replay_start_snap_sec: float = 1.0,
    replay_eef_joint_lead_sec: float = 0.0,
    replay_force_rebake_eef_fk: bool = False,
    replay_eef_joint_fallback_enabled: bool = False,
    replay_eef_joint_fallback_step_rad: float = 0.015,
    replay_eef_joint_fallback_hold_sec: float = 0.8,
    replay_eef_joint_fallback_lookahead_sec: float = 0.2,
    replay_singularity_adjust: bool = False,
    replay_singularity_joint_fallback: bool = False,
    replay_eef_ik_seed_closed_loop: bool = True,
    replay_eef_ik_seed_hint_blend: float = 0.0,
    replay_singularity_config: Optional[object] = None,
) -> Optional[np.ndarray]:
    label = _key_label(action)
    keys.drain_all()
    print(
        "\n>> 回放/Bake 进行中：从臂执行轨迹，不再跟随 U-Arm，请保持主臂不动",
        flush=True,
    )
    replay_common_kwargs = {
        "replay_hz": replay_hz,
        "smooth_window": replay_smooth_window,
        "blend_sec": replay_blend_sec,
        "smooth_gripper": replay_smooth_gripper,
        "max_blend_sec": replay_max_blend_sec,
        "blend_skip_tol": replay_blend_skip_tol,
        "blend_speed_rad_s": replay_blend_speed,
        "post_hold_sec": replay_post_hold_sec,
        "return_to_start": replay_return_to_start,
        "return_blend_sec": replay_return_blend_sec,
        "return_max_blend_sec": replay_return_max_blend_sec,
        "return_settle_sec": replay_return_settle_sec,
        "return_hold_sec": replay_return_hold_sec,
        "return_tol_rad": replay_return_tol_rad,
        "start_max_blend_sec": replay_start_max_blend_sec,
        "start_settle_sec": replay_start_settle_sec,
        "start_snap_sec": replay_start_snap_sec,
    }
    replay_eef_kwargs = {
        **replay_common_kwargs,
        "eef_smooth_window": replay_eef_smooth_window,
        "eef_max_rpy_step_rad": replay_eef_max_rpy_step_rad,
        "eef_ik_seed": replay_eef_ik_seed,
        "orientation_interp": replay_orientation_interp,
        "eef_source": replay_eef_source,
        "eef_ik_seed_smooth_window": replay_eef_ik_seed_smooth_window,
        "eef_quat_smooth_window": replay_eef_quat_smooth_window,
        "eef_joint_lead_sec": replay_eef_joint_lead_sec,
        "eef_joint_fallback_enabled": replay_eef_joint_fallback_enabled,
        "eef_joint_fallback_step_rad": replay_eef_joint_fallback_step_rad,
        "eef_joint_fallback_hold_sec": replay_eef_joint_fallback_hold_sec,
        "eef_joint_fallback_lookahead_sec": replay_eef_joint_fallback_lookahead_sec,
        "singularity_adjust": replay_singularity_adjust,
        "singularity_joint_fallback": replay_singularity_joint_fallback,
        "eef_ik_seed_closed_loop": replay_eef_ik_seed_closed_loop,
        "eef_ik_seed_hint_blend": replay_eef_ik_seed_hint_blend,
        "singularity_config": replay_singularity_config,
    }
    replay_parquet = hdf5_path.parent / f".replay_{hdf5_path.stem}.parquet"
    last_eef: Optional[np.ndarray] = None
    last_qpos: Optional[np.ndarray] = None
    replay_control_mode = "joint"
    replay_period = 1.0 / max(replay_hz, 1.0)
    try:
        if action in {"r", "R"}:
            print(f"\n>> [{label}] HDF5 关节回放（mode=5，smooth={replay_smooth_window}）…", flush=True)
            print(f"    数据: {hdf5_path}", flush=True)
            print("    提示：回放前可将 U-Arm 摆到接近从臂当前姿态", flush=True)
            last_eef, last_qpos = replay_hdf5_file(
                bridge,
                hdf5_path,
                joint_source="action",
                control_mode="joint",
                **replay_common_kwargs,
            )
            replay_control_mode = "joint"
        elif action in {"p", "P"}:
            print(
                f"\n>> [{label}] HDF5 末端回放（mode=4，seed={replay_eef_ik_seed}，"
                f"orient={replay_orientation_interp}，eef={replay_eef_source}）…",
                flush=True,
            )
            print(f"    数据: {hdf5_path}", flush=True)
            if replay_auto_bake_eef_fk:
                ensure_eef_fk_baked(
                    bridge,
                    hdf5_path,
                    auto_bake=True,
                    eef_source=replay_eef_source,  # type: ignore[arg-type]
                    replay_hz=replay_bake_hz,
                    smooth_window=replay_bake_smooth_window,
                    quat_smooth_window=replay_bake_quat_smooth_window,
                    blend_sec=replay_blend_sec,
                    max_blend_sec=replay_max_blend_sec,
                    force_rebake=replay_force_rebake_eef_fk,
                    return_to_start=replay_return_to_start,
                    return_blend_sec=replay_return_blend_sec,
                    return_max_blend_sec=replay_return_max_blend_sec,
                    return_blend_speed_rad_s=replay_blend_speed,
                    return_settle_sec=replay_return_settle_sec,
                    return_hold_sec=replay_return_hold_sec,
                    return_tol_rad=replay_return_tol_rad,
                )
            else:
                print(
                    "    跳过 bake：使用 observations/eef + 平滑直接末端回放"
                    "（config: auto_bake_before_eef_replay=false）",
                    flush=True,
                )
            last_eef, last_qpos = replay_hdf5_file(
                bridge,
                hdf5_path,
                joint_source="action",
                control_mode="eef",
                **replay_eef_kwargs,
            )
            replay_control_mode = "eef"
        elif action in {"v", "V"}:
            print(
                f"\n>> [{label}] HDF5 末端→关节回放（diff IK→mode=5，"
                f"orient={replay_orientation_interp}，eef={replay_eef_source}）…",
                flush=True,
            )
            print(f"    数据: {hdf5_path}", flush=True)
            print("    提示：先转角度再发 mode=5，不用 SDK mode=4", flush=True)
            last_eef, last_qpos = replay_hdf5_file(
                bridge,
                hdf5_path,
                joint_source="action",
                control_mode="eef_via_joint",
                **replay_eef_kwargs,
            )
            replay_control_mode = "eef_via_joint"
        elif action in {"l", "L"}:
            print(
                f"\n>> [{label}] LeRobot 回放启动（HDF5→parquet 约 30 秒，请等待）…",
                flush=True,
            )
            print(f"    输入 HDF5: {hdf5_path}", flush=True)
            print(f"    临时 parquet: {replay_parquet}", flush=True)
            print(
                f"    正式 LeRobot 训练集需转换: "
                f"{_lerobot_convert_command(output_dir, project_dir, record_hz)}",
                flush=True,
            )
            last_eef, last_qpos = replay_lerobot_from_hdf5(
                bridge,
                hdf5_path,
                record_fps=record_hz,
                **replay_common_kwargs,
            )
            replay_control_mode = "joint"
        else:
            return
    except KeyboardInterrupt:
        print("回放已取消。", flush=True)
    else:
        if last_eef is not None and last_qpos is not None:
            finish_replay_handoff(
                bridge,
                mapper,
                servo,
                last_eef,
                last_qpos,
                replay_period,
                control_mode=replay_control_mode,
                sync_uarm=replay_sync_uarm,
                blend_to_uarm=False,
            )
    finally:
        _warn_ignored_keys(keys, "回放结束")
        print_post_save_menu(
            hdf5_path,
            camera_names=camera_names,
            project_dir=project_dir,
            output_dir=output_dir,
            record_fps=record_hz,
        )
    return _snapshot_arm_qpos(bridge)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def state_to_obs(state: ArmState) -> Dict[str, np.ndarray]:
    qpos = np.asarray(state.joint_pos, dtype=np.float64)
    eef = np.concatenate([np.asarray(state.end_pos, dtype=np.float64), qpos[6:7]])
    return {
        "qpos": qpos,
        "eef": eef,
        "qvel": np.asarray(state.joint_vel, dtype=np.float64),
        "effort": np.asarray(state.joint_cur, dtype=np.float64),
    }


def write_hdf5(
    path: Path,
    task: str,
    buffers: Dict[str, List[Any]],
    camera_names: List[str],
    use_depth: bool,
    record_fps: float = 30.0,
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
        root.attrs["collection_mode"] = "teleop"
        root.attrs["action_space"] = "joint"
        root.attrs["teleop_control_mode"] = "joint"
        root.attrs["camera_names"] = json.dumps(camera_names)
        root.attrs["fps"] = float(record_fps)
        root.attrs["saved_at"] = datetime.now().isoformat(timespec="seconds")
        root.attrs["filename"] = path.name

        obs = root.create_group("observations")
        img_grp = obs.create_group("images")
        for cam_name in camera_names:
            frames = np.stack(images_dict[cam_name], axis=0).astype(np.uint8)
            img_grp.create_dataset(cam_name, data=frames, compression="gzip", compression_opts=1)

        if use_depth and buffers.get("depth") and any(d is not None for d in buffers["depth"]):
            depth_grp = obs.create_group("images_depth")
            depth_stack = []
            for d in buffers["depth"]:
                depth_stack.append(np.zeros_like(buffers["depth"][0]) if d is None else d)
            depth_grp.create_dataset(
                WRIST_CAM,
                data=np.stack(depth_stack, axis=0).astype(np.uint16),
                compression="gzip",
                compression_opts=4,
            )

        for name in ("qpos", "eef", "qvel", "effort"):
            obs.create_dataset(name, data=np.stack(buffers[name], axis=0))

        root.create_dataset("action", data=np.stack(buffers["action"], axis=0))
        root.create_dataset("action_eef", data=np.stack(buffers["eef"], axis=0))
        root.create_dataset("master_deg", data=np.stack(buffers["master_deg"], axis=0))
        root.create_dataset("time", data=np.asarray(buffers["time"], dtype=np.float64))
        root.create_dataset("arm_stamp", data=np.asarray(buffers["arm_stamp"], dtype=np.float64))


@dataclass
class _SaveJob:
    path: Path
    task: str
    buffers: Dict[str, List[Any]]
    camera_names: List[str]
    use_depth: bool
    record_fps: float
    episode_label: int
    open_menu_when_done: bool = False


@dataclass
class _SaveResult:
    path: Path
    episode_label: int
    n_frames: int
    ok: bool
    error: str = ""
    open_menu_when_done: bool = False
    elapsed_sec: float = 0.0


class AsyncHdf5Writer:
    """Background HDF5 write pipeline so teleop can continue recording."""

    def __init__(self, *, max_pending: int = 4) -> None:
        self._jobs: queue.Queue[Optional[_SaveJob]] = queue.Queue(maxsize=max(1, max_pending))
        self._results: queue.Queue[_SaveResult] = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, name="hdf5-save", daemon=True)
        self._thread.start()

    @property
    def pending_count(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, job: _SaveJob) -> None:
        with self._lock:
            self._pending += 1
        backlog = self.pending_count
        print(
            f">> [后台保存] 已排队 {job.path.name}（{len(job.buffers['qpos'])} 帧，"
            f"队列中 {backlog} 条）— 可继续录下一条",
            flush=True,
        )
        try:
            self._jobs.put(job, timeout=120.0)
        except queue.Full:
            with self._lock:
                self._pending = max(0, self._pending - 1)
            raise RuntimeError("后台保存队列已满，请稍候再停录")

    def poll_results(self) -> List[_SaveResult]:
        out: List[_SaveResult] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def shutdown(self, timeout_sec: float = 600.0) -> None:
        pending = self.pending_count
        if pending > 0:
            print(f">> 等待后台保存完成（剩余 {pending} 条）…", flush=True)
        self._jobs.put(None)
        self._thread.join(timeout=timeout_sec)
        if self._thread.is_alive():
            print(">> 警告：后台保存线程超时未结束，部分 episode 可能未写完", flush=True)
        for result in self.poll_results():
            if result.ok:
                print(
                    f">> [后台保存完成] {result.path.name} "
                    f"({result.n_frames} 帧, {result.elapsed_sec:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f">> [后台保存失败] {result.path.name}: {result.error}",
                    flush=True,
                )

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                self._jobs.task_done()
                break
            t0 = time.monotonic()
            n = len(job.buffers.get("qpos", []))
            try:
                write_hdf5(
                    job.path,
                    job.task,
                    job.buffers,
                    job.camera_names,
                    use_depth=job.use_depth,
                    record_fps=job.record_fps,
                )
                result = _SaveResult(
                    path=job.path,
                    episode_label=job.episode_label,
                    n_frames=n,
                    ok=True,
                    open_menu_when_done=job.open_menu_when_done,
                    elapsed_sec=time.monotonic() - t0,
                )
            except Exception as exc:  # noqa: BLE001 — surface to UI thread
                result = _SaveResult(
                    path=job.path,
                    episode_label=job.episode_label,
                    n_frames=n,
                    ok=False,
                    error=str(exc),
                    open_menu_when_done=job.open_menu_when_done,
                    elapsed_sec=time.monotonic() - t0,
                )
            finally:
                # Drop large buffers ASAP.
                job.buffers.clear()
                with self._lock:
                    self._pending = max(0, self._pending - 1)
                self._results.put(result)
                self._jobs.task_done()


def print_post_save_menu(
    saved_path: Path,
    *,
    camera_names: List[str],
    project_dir: Path,
    output_dir: Path,
    record_fps: float,
) -> None:
    lerobot_root = _lerobot_runs_root(project_dir)
    replay_parquet = saved_path.parent / f".replay_{saved_path.stem}.parquet"
    print("\n" + "=" * 60, flush=True)
    print(f"已保存 HDF5: {saved_path}", flush=True)
    if camera_names:
        cams = ", ".join(f"observations/images/{n}" for n in camera_names)
        print(f"  相机数据: {cams}", flush=True)
    print(f"  采集目录: {output_dir}", flush=True)
    print(f"  LeRobot 训练集（转换后）: {lerobot_root}/<日期时间>/", flush=True)
    print(
        f"  转换命令: {_lerobot_convert_command(output_dir, project_dir, record_fps)}",
        flush=True,
    )
    print("=" * 60, flush=True)
    print("  r  HDF5 关节回放（action，mode=5）", flush=True)
    print("  p  HDF5 末端回放（mode=4，observations/eef + 平滑；可选 bake 见 config）", flush=True)
    print("  v  HDF5 末端→关节回放（diff IK→mode=5，避开 mode=4 奇异）", flush=True)
    print("  （菜单期间从臂默认锁定，不跟随 U-Arm；回放结束自动回到轨迹起点）", flush=True)
    print(
        f"  l  LeRobot 回放（临时 parquet: {replay_parquet.name}，约 30 秒，按一次即可）",
        flush=True,
    )
    print("  n  录制新一条（0/Space 开始，s/Enter 停止保存）", flush=True)
    print("  q  退出", flush=True)
    print("  （终端 cbreak 模式不回显按键，见 >> [...] 行确认已收到）", flush=True)
    print("=" * 60 + "\n", flush=True)


def print_session_key_help(*, quick_record: bool = False) -> None:
    print("\n" + "=" * 60)
    print("【采集按键】终端或相机预览窗口均可")
    print("  0 / 空格     开始录制（先强制归位 go_home，约 3–8 秒）")
    print("  s / Enter    停止录制并交后台保存（可马上开下一条）")
    print("  x / Esc      放弃当前录制（不保存，回到可按空格开录）")
    print("  d            删除上一条（未在录制时）")
    print("  q            退出")
    if quick_record:
        print("【快速录制】保存后不进回放菜单")
        print("  n / 0 / 空格 / Enter  立即开始下一轮")
        print("  m                    打开完整回放菜单")
        print("  q                    退出")
    else:
        print("保存后菜单: r=关节  p=末端(mode4)  v=末端→关节  l=LeRobot  n=新一条  q=退出")
    print("=" * 60 + "\n", flush=True)


def print_quick_next_prompt(saved_path: Path, episode_idx: int, *, saving: bool = False) -> None:
    print("\n" + "-" * 60, flush=True)
    saving_note = "（HDF5 后台写入中）" if saving else ""
    print(f"快速录制：已提交 {saved_path.name}{saving_note}", flush=True)
    print("  从臂已重置到启动归位并锁定（与程序启动相同）", flush=True)
    print("  主臂可先摆到下一轮起始姿态（从臂暂不跟随）", flush=True)
    print(
        f"  按  n / 0 / 空格 / Enter  → 再归位同步后开始 episode_{episode_idx}",
        flush=True,
    )
    print("  按  m  → 打开回放菜单   按  q  → 退出", flush=True)
    print("-" * 60 + "\n", flush=True)


def print_recording_started(episode_idx: int) -> None:
    print("\n" + "!" * 60, flush=True)
    print(f"!!! 正在录制 episode_{episode_idx} !!!", flush=True)
    print("!!! 完成后按  s  或  Enter  停止并保存 !!!", flush=True)
    print("!!! 放错物体 / 中途作废：按  x  或  Esc  放弃（不保存） !!!", flush=True)
    print("!" * 60 + "\n", flush=True)


def print_recording_reminder(episode_idx: int) -> None:
    print(
        f"[录制中 episode_{episode_idx}] s/Enter=保存  |  x/Esc=放弃重来",
        flush=True,
    )


def _try_append_record_frame(
    record_buffers: Optional[Dict[str, List[Any]]],
    *,
    bridge: X5ArmBridge,
    command: TeleopCommand,
    mapper: UArmToX5Mapper,
    camera_names: List[str],
    last_rgb_frames: Optional[Dict[str, np.ndarray]],
    last_wrist_depth: Optional[np.ndarray],
    use_depth: bool,
    no_camera: bool,
) -> bool:
    if record_buffers is None:
        return False
    state = bridge.latest_state
    if state is None:
        return False
    if not no_camera and (
        last_rgb_frames is None or len(last_rgb_frames) != len(camera_names)
    ):
        return False
    obs = state_to_obs(state)
    action = mapper.to_qpos7(command)
    record_buffers["qpos"].append(obs["qpos"])
    record_buffers["eef"].append(obs["eef"])
    record_buffers["qvel"].append(obs["qvel"])
    record_buffers["effort"].append(obs["effort"])
    record_buffers["action"].append(action)
    record_buffers["master_deg"].append(command.master_deg)
    if no_camera:
        for name in camera_names:
            record_buffers["images"][name].append(np.zeros((480, 480, 3), dtype=np.uint8))
        record_buffers["depth"].append(None)
    else:
        for name in camera_names:
            record_buffers["images"][name].append(last_rgb_frames[name].copy())  # type: ignore[index]
        record_buffers["depth"].append(
            None if last_wrist_depth is None else last_wrist_depth.copy()
        )
    record_buffers["time"].append(time.time())
    record_buffers["arm_stamp"].append(state.stamp_sec)
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Arm teleop session: record, save, replay.")
    p.add_argument(
        "--config",
        default="config/teleop_uarm.example.json",
        help="Teleop config JSON path.",
    )
    p.add_argument("--output-dir", default=None, help="Override record.output_dir.")
    p.add_argument("--task", default=None, help="Override record.task.")
    p.add_argument("--port", default=None, help="Override uarm.port.")
    p.add_argument("--no-camera", action="store_true", help="Record without camera images.")
    p.add_argument("--no-desktop-camera", action="store_true", help="Wrist D405 only (no D435 desktop).")
    p.add_argument("--wrist-serial", default=None, help="D405 serial (auto-detect if omitted).")
    p.add_argument("--desktop-serial", default=None, help="D435/D435i desktop serial (auto-detect).")
    p.add_argument("--teleop-only", action="store_true", help="Teleop only, no record/replay menu.")
    p.add_argument("--use-depth", action="store_true", help="Store depth in HDF5.")
    p.add_argument("--debug-gripper", action="store_true", help="Print s6 and gripper cmd ~2 Hz.")
    p.add_argument("--replay-hz", type=float, default=50.0, help="Replay control rate. Default: 50")
    p.add_argument(
        "--no-post-save-menu",
        action="store_true",
        help="Save and continue without replay menu (legacy behavior).",
    )
    p.add_argument(
        "--quick-record",
        action="store_true",
        help="After save, one key (n/0/Space/Enter) starts the next episode; skip replay menu.",
    )
    p.add_argument(
        "--replay-file",
        default=None,
        help="Open replay menu on an existing HDF5 (e.g. data/teleop_demos/episode_0.hdf5).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (project_dir / config_path).resolve()
    cfg = load_config(config_path)

    uarm_cfg = cfg["uarm"]
    control_cfg = cfg["control"]
    mapping_cfg = UArmMappingConfig.from_dict(cfg["mapping"])
    record_cfg = cfg["record"]
    replay_cfg = cfg.get("replay") or {}
    startup_cfg = cfg.get("startup") or {}
    camera_cfg = cfg["camera"]

    control_hz = float(control_cfg.get("hz", 50.0))
    control_dt = 1.0 / max(control_hz, 1.0)
    control_mode = int(control_cfg.get("mode", 5))
    record_hz = float(record_cfg.get("hz", 30.0))
    record_period = 1.0 / max(record_hz, 1.0)
    filename_with_timestamp = bool(record_cfg.get("filename_with_timestamp", True))
    async_save = bool(record_cfg.get("async_save", True))
    async_save_max_pending = int(record_cfg.get("async_save_max_pending", 4))
    replay_hz = float(args.replay_hz or replay_cfg.get("hz", 50.0))
    replay_smooth_window = int(replay_cfg.get("smooth_window", 7))
    replay_blend_sec = float(replay_cfg.get("blend_sec", 0.5))
    replay_smooth_gripper = bool(replay_cfg.get("smooth_gripper", False))
    replay_max_blend_sec = float(replay_cfg.get("max_blend_sec", 2.0))
    replay_blend_skip_tol = float(replay_cfg.get("blend_skip_tol_rad", 0.05))
    replay_blend_speed = float(replay_cfg.get("blend_speed_rad_s", 0.8))
    replay_post_hold_sec = float(replay_cfg.get("post_hold_sec", 1.0))
    replay_sync_uarm = bool(replay_cfg.get("sync_uarm_after_replay", True))
    replay_eef_smooth_window = int(replay_cfg.get("eef_smooth_window", 15))
    replay_eef_max_rpy_step_rad = float(replay_cfg.get("eef_max_rpy_step_rad", 0.04))
    replay_eef_ik_seed = str(replay_cfg.get("eef_ik_seed", "action"))
    replay_orientation_interp = str(replay_cfg.get("orientation_interp", "slerp"))
    replay_eef_source = str(replay_cfg.get("eef_source", "observations"))
    replay_eef_ik_seed_smooth_window = int(replay_cfg.get("eef_ik_seed_smooth_window", 15))
    replay_eef_quat_smooth_window = int(replay_cfg.get("eef_quat_smooth_window", 7))
    replay_auto_bake_eef_fk = bool(replay_cfg.get("auto_bake_before_eef_replay", False))
    replay_bake_hz = float(replay_cfg.get("bake_hz", 100.0))
    replay_bake_smooth_window = int(replay_cfg.get("bake_smooth_window", 15))
    replay_bake_quat_smooth_window = int(replay_cfg.get("bake_quat_smooth_window", 7))
    replay_return_to_start = bool(replay_cfg.get("return_to_start_after_replay", True))
    replay_return_blend_sec = float(replay_cfg.get("return_to_start_blend_sec", 1.5))
    replay_return_max_blend_sec = float(replay_cfg.get("return_to_start_max_blend_sec", 6.0))
    replay_return_settle_sec = float(replay_cfg.get("return_to_start_settle_sec", 1.5))
    replay_return_hold_sec = float(replay_cfg.get("return_to_start_hold_sec", 2.5))
    replay_return_tol_rad = float(replay_cfg.get("return_to_start_tol_rad", 0.04))
    replay_start_max_blend_sec = float(replay_cfg.get("start_max_blend_sec", 6.0))
    replay_start_settle_sec = float(replay_cfg.get("start_settle_sec", 0.8))
    replay_start_snap_sec = float(
        replay_cfg.get("start_snap_sec", replay_cfg.get("homing_lock_sec", 2.5))
    )
    replay_homing_lock_sec = float(replay_cfg.get("homing_lock_sec", 2.5))
    follow_uarm_during_menu = bool(replay_cfg.get("follow_uarm_during_menu", False))
    replay_eef_joint_lead_sec = float(replay_cfg.get("eef_joint_lead_sec", 0.0))
    replay_force_rebake_eef_fk = bool(replay_cfg.get("force_rebake_eef_fk", False))
    replay_eef_joint_fallback_enabled = bool(
        replay_cfg.get("eef_joint_fallback_enabled", False)
    )
    replay_eef_joint_fallback_step_rad = float(
        replay_cfg.get("eef_joint_fallback_step_deg", 0.85)
    ) * (np.pi / 180.0)
    replay_eef_joint_fallback_hold_sec = float(
        replay_cfg.get("eef_joint_fallback_hold_sec", 0.8)
    )
    replay_eef_joint_fallback_lookahead_sec = float(
        replay_cfg.get("eef_joint_fallback_lookahead_sec", 0.2)
    )
    replay_singularity_adjust = bool(replay_cfg.get("singularity_adjust", False))
    replay_singularity_joint_fallback = bool(
        replay_cfg.get("singularity_joint_fallback", False)
    )
    replay_eef_ik_seed_closed_loop = bool(
        replay_cfg.get("eef_ik_seed_closed_loop", True)
    )
    replay_eef_ik_seed_hint_blend = float(replay_cfg.get("eef_ik_seed_hint_blend", 0.0))
    from lewm_x5_grasp.singularity_adjuster import SingularityConfig

    replay_singularity_config = SingularityConfig.from_dict(
        replay_cfg.get("singularity") if isinstance(replay_cfg.get("singularity"), dict) else None
    )
    startup_gripper_blend_min = float(startup_cfg.get("gripper_blend_min_sec", 0.5))
    startup_gripper_blend_max = float(startup_cfg.get("gripper_blend_max_sec", 3.0))
    startup_gripper_blend_speed = float(startup_cfg.get("gripper_blend_speed", 2.0))
    startup_gripper_settle = float(
        startup_cfg.get("gripper_settle_sec", mapping_cfg.gripper_open_hold_sec)
    )
    startup_go_home = bool(startup_cfg.get("go_home", True))
    if "go_home_position_deg" in startup_cfg:
        startup_go_home_position = [
            float(v) * (np.pi / 180.0)
            for v in startup_cfg["go_home_position_deg"]
        ]
    else:
        startup_go_home_position = [
            float(v) for v in startup_cfg.get("go_home_position", [0.0] * 6)
        ]
    startup_go_home_position = startup_go_home_position[:6]
    startup_go_home_blend_max = float(startup_cfg.get("go_home_blend_max_sec", 4.0))
    startup_go_home_blend_speed = float(startup_cfg.get("go_home_blend_speed_rad_s", 0.6))
    startup_go_home_before_each_record = bool(
        startup_cfg.get("go_home_before_each_record", True)
    )
    startup_go_home_lock_sec = float(
        startup_cfg.get(
            "go_home_lock_sec",
            replay_cfg.get("homing_lock_sec", replay_cfg.get("start_snap_sec", 2.5)),
        )
    )
    startup_sync_uarm = bool(startup_cfg.get("sync_uarm_before_teleop", True))

    output_dir = Path(args.output_dir or record_cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (project_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task = args.task or record_cfg.get("task", "grab the bottle")
    port = args.port or uarm_cfg.get("port", "/dev/ttyUSB0")

    enable_session = not args.teleop_only
    enable_menu = enable_session and not args.no_post_save_menu
    quick_record = bool(
        args.quick_record or record_cfg.get("quick_record", False)
    )
    if quick_record and args.teleop_only:
        raise SystemExit("--quick-record cannot be used with --teleop-only")
    if quick_record and args.no_post_save_menu:
        # Quick mode replaces legacy no-menu: one-key next episode.
        pass
    if quick_record:
        enable_menu = True  # allow "m" to open full menu when needed

    replay_file_path: Optional[Path] = None
    if args.replay_file:
        if not enable_menu and not quick_record:
            raise SystemExit("--replay-file requires record/replay session (not --teleop-only / --no-post-save-menu)")
        if quick_record:
            print(
                "note: --replay-file opens the full menu first; after that use quick-record flow",
                flush=True,
            )
        replay_file_path = _resolve_hdf5_path(
            args.replay_file,
            output_dir=output_dir,
            project_dir=project_dir,
        )
        if not replay_file_path.is_file():
            raise SystemExit(f"Replay file not found: {replay_file_path}")

    mode4_guard_cfg = dict(cfg.get("mode4_guard") or {"enabled": True})

    rclpy.init()
    bridge = X5ArmBridge(mode4_guard=mode4_guard_cfg)
    mapper = UArmToX5Mapper(mapping_cfg, control_dt=control_dt)
    servo = UArmServoReader(port=port, baudrate=int(uarm_cfg.get("baudrate", 115200)))
    camera: Optional[MultiRealSenseSource] = None
    camera_poller: Optional[CameraPoller] = None
    camera_names: List[str] = []
    if enable_session and not args.no_camera:
        enable_desktop = not args.no_desktop_camera and bool(camera_cfg.get("enable_desktop", True))
        camera = build_teleop_cameras_from_config(
            camera_cfg,
            enable_desktop=enable_desktop,
            wrist_serial=args.wrist_serial,
            desktop_serial=args.desktop_serial,
            wrist_enable_depth=args.use_depth,
        )
        camera_names = camera.camera_names
        print(f"  cameras: {camera_names}", flush=True)
        print("  warming up cameras…", flush=True)
        warm = warmup_cameras(camera, n_frames=int(camera_cfg.get("warmup_frames", 30)))
        for name, frame in warm.items():
            if frame is None:
                print(f"  WARNING: {name} no frames during warmup", flush=True)
            else:
                print(f"  {name} OK: shape={frame.shape} mean={float(frame.mean()):.1f}", flush=True)
        camera_poller = CameraPoller(camera)
        print("  camera preview: main-thread OpenCV (GTK backend)", flush=True)

    reader_hz = float(uarm_cfg.get("read_hz", 50.0))
    reader_thread = threading.Thread(target=servo.read_loop, kwargs={"hz": reader_hz}, daemon=True)
    reader_thread.start()
    time.sleep(0.3)
    boot_angles = servo.get_angles_deg()

    # Probe master DOF motion: dead servo5 + swap_joints_45 parks follower wrist at singularity.
    probe_sec = float(startup_cfg.get("master_health_probe_sec", 1.5))
    probe_n = max(10, int(probe_sec * max(reader_hz, 1.0)))
    probe = np.zeros((probe_n, 7), dtype=np.float64)
    print(
        f"  master health: 请在 {probe_sec:.1f}s 内缓慢活动 U-Arm 各轴（尤其腕部）…",
        flush=True,
    )
    for i in range(probe_n):
        probe[i] = servo.get_angles_deg()
        time.sleep(1.0 / max(reader_hz, 1.0))
    master_std = probe.std(axis=0)
    print(
        "  master std°: "
        + ", ".join(f"s{i}={master_std[i]:.2f}" for i in range(7)),
        flush=True,
    )
    if mapper.maybe_fix_swap_from_master_std(master_std):
        print(
            "  [mapping] 检测到主臂 s5 几乎不动而 s4 有运动："
            "已自动关闭 swap_joints_45，避免从臂腕部 j4 锁在奇异点。",
            flush=True,
        )
    elif mapping_cfg.swap_joints_45 and master_std[5] < mapping_cfg.master_stuck_std_deg:
        print(
            "  [mapping] WARNING: 主臂 s5 几乎不动（std="
            f"{master_std[5]:.2f}°）。swap_joints_45 仍开启时从臂腕部易奇异；"
            "可手动设 swap_joints_45=false，或修好舵机后重采。"
            "（已不再自动改映射，以免跟手错轴/卡顿）",
            flush=True,
        )
    if mapping_cfg.singularity_avoidance:
        print(
            "  [mapping] singularity_avoidance ON (soft gain="
            f"{mapping_cfg.singularity_avoidance_gain:.2f}): "
            f"elbow≥{np.degrees(mapping_cfg.elbow_min_abs_rad):.1f}°, "
            f"wrist≥{np.degrees(mapping_cfg.wrist_min_abs_rad):.1f}°",
            flush=True,
        )
    else:
        print(
            "  [mapping] singularity_avoidance OFF（录制原样跟手；奇异仅回放后处理）",
            flush=True,
        )

    episode_idx = _next_episode_index(output_dir)
    recording = False
    replaying = False
    record_buffers: Optional[Dict[str, List[Any]]] = None
    last_record_time = 0.0
    last_grip_debug_time = 0.0
    last_rgb_frames: Optional[Dict[str, np.ndarray]] = None
    last_wrist_depth: Optional[np.ndarray] = None
    record_frame_warned = False
    menu_pending_path: Optional[Path] = replay_file_path
    menu_banner_printed = False
    session_saved_paths: list[Path] = []
    arm_hold_qpos: Optional[np.ndarray] = None
    last_recording_reminder_time = 0.0
    RECORDING_REMINDER_SEC = 30.0
    quick_awaiting_next = False
    last_saved_for_menu: Optional[Path] = None
    save_writer: Optional[AsyncHdf5Writer] = None
    if enable_session and async_save:
        save_writer = AsyncHdf5Writer(max_pending=async_save_max_pending)
    deferred_menu_path: Optional[Path] = None

    def _handle_save_results() -> None:
        nonlocal menu_pending_path, menu_banner_printed, last_saved_for_menu
        nonlocal arm_hold_qpos, deferred_menu_path
        if save_writer is None:
            return
        for result in save_writer.poll_results():
            if not result.ok:
                print(
                    f"\n>> [后台保存失败] {result.path.name}: {result.error}",
                    flush=True,
                )
                continue
            session_saved_paths.append(result.path)
            print_episode_saved(
                result.path,
                result.n_frames,
                camera_names=camera_names,
                project_dir=project_dir,
                output_dir=output_dir,
                record_fps=record_hz,
            )
            print(
                f">> [后台保存完成] {result.path.name} "
                f"({result.n_frames} 帧, {result.elapsed_sec:.1f}s)"
                + (
                    f"  仍有 {save_writer.pending_count} 条在写"
                    if save_writer.pending_count > 0
                    else ""
                ),
                flush=True,
            )
            last_saved_for_menu = result.path
            if result.open_menu_when_done and enable_menu and not quick_record:
                if recording or replaying or quick_awaiting_next or menu_pending_path is not None:
                    deferred_menu_path = result.path
                    print(
                        ">> 回放菜单已延后（当前忙碌）；空闲后自动弹出",
                        flush=True,
                    )
                else:
                    menu_pending_path = result.path
                    menu_banner_printed = False
                    if arm_hold_qpos is None:
                        arm_hold_qpos = _snapshot_arm_qpos(bridge)
                    print(
                        ">> 菜单期间从臂锁定在当前姿态，不会跟随 U-Arm",
                        flush=True,
                    )

        if (
            deferred_menu_path is not None
            and not recording
            and not replaying
            and not quick_awaiting_next
            and menu_pending_path is None
        ):
            menu_pending_path = deferred_menu_path
            deferred_menu_path = None
            menu_banner_printed = False
            if arm_hold_qpos is None:
                arm_hold_qpos = _snapshot_arm_qpos(bridge)
            print(">> 打开延后的回放菜单", flush=True)

    print("=" * 60)
    print("U-Arm -> ARX X5 遥操作采集会话")
    print(f"  config : {config_path}")
    print(f"  port   : {port}")
    print(f"  servos : {[f's{i}:{boot_angles[i]:+.1f}' for i in range(7)]}")
    print(f"  ctrl   : {control_hz} Hz, teleop=joint (mode={control_mode})")
    print("  record : action=关节角, observations/eef=实测末端（回放 p 键用 EEF+阻尼 IK）", flush=True)
    if enable_session:
        print(f"  record : {record_hz} Hz -> {output_dir}")
        if filename_with_timestamp:
            print("  record : 文件名带时间戳 episode_YYYY-MM-DD_HHMMSS.hdf5（不覆盖）")
        if async_save:
            print(
                f"  save   : 后台流水线（最多排队 {async_save_max_pending} 条，停录后可马上开下一条）",
                flush=True,
            )
        if quick_record:
            print("  mode   : 快速录制（保存后一键开下一轮，跳过回放菜单）", flush=True)
        print(
            f"  replay : {replay_hz} Hz, smooth={replay_smooth_window}, "
            f"eef_rpy_smooth={replay_eef_smooth_window}, "
            f"ik_seed_smooth={replay_eef_ik_seed_smooth_window}, "
            f"eef_quat_smooth={replay_eef_quat_smooth_window}, "
            f"blend={replay_blend_sec}-{replay_max_blend_sec}s, post_hold={replay_post_hold_sec}s"
        )
        print_data_storage_info(
            output_dir=output_dir,
            project_dir=project_dir,
            camera_names=camera_names,
            record_fps=record_hz,
        )
    print("=" * 60)
    print("先启动: ros2 launch arx_x5_controller open_single_arm.launch.py")
    if enable_session:
        print_session_key_help(quick_record=quick_record)
        if replay_file_path is not None:
            print(
                f"回放文件已指定: {_path_display(replay_file_path, project_dir)} "
                f"（启动后按 r / p / v / l 回放，可重复对比）",
                flush=True,
            )
    else:
        print("Keys: q=quit (--teleop-only)")
    print("=" * 60)

    try:
        bridge.wait_for_state(timeout_sec=30.0)
        print("请保持 U-Arm 弹性夹爪自然张开 1 秒…")
        time.sleep(1.0)
        grip_idx = mapping_cfg.gripper_index
        rest_angles = servo.get_angles_deg()
        mapper.set_gripper_rest_deg(rest_angles[grip_idx])
        open_cmd = float(mapping_cfg.gripper_open_cmd)
        print(
            f"  gripper: mode={mapping_cfg.gripper_mode} rest s{grip_idx}={rest_angles[grip_idx]:+.1f}° "
            f"-> open cmd={open_cmd}"
        )

        init_state = bridge.latest_state
        if init_state is not None:
            if startup_go_home:
                prepare_before_recording(
                    bridge,
                    mapper,
                    servo,
                    startup_go_home_position,
                    open_cmd,
                    control_dt,
                    control_mode=control_mode,
                    max_blend_sec=startup_go_home_blend_max,
                    blend_speed_rad_s=startup_go_home_blend_speed,
                    lock_sec=startup_go_home_lock_sec,
                    sync_uarm=startup_sync_uarm,
                    gripper_blend_min_sec=startup_gripper_blend_min,
                    gripper_blend_max_sec=startup_gripper_blend_max,
                    gripper_blend_speed=startup_gripper_blend_speed,
                    gripper_settle_sec=startup_gripper_settle,
                    label="startup",
                )
            else:
                g_before = float(bridge.latest_state.joint_pos[6]) if bridge.latest_state else float(
                    init_state.joint_pos[6]
                )
                print(f"  arm_status gripper before open: {g_before:.3f}")
                startup_blend_gripper_open(
                    bridge,
                    open_cmd,
                    control_dt,
                    control_mode=control_mode,
                    min_sec=startup_gripper_blend_min,
                    max_sec=startup_gripper_blend_max,
                    speed=startup_gripper_blend_speed,
                    settle_sec=startup_gripper_settle,
                )
                rclpy.spin_once(bridge, timeout_sec=0.0)
                if bridge.latest_state is not None:
                    g_after = float(bridge.latest_state.joint_pos[6])
                    print(f"  arm_status gripper after open:  {g_after:.3f}")
                mapper.reset_gripper_state(open_cmd)
                if startup_sync_uarm:
                    master_deg = servo.get_angles_deg()
                    offset = mapper.align_to_follower(init_state.joint_pos, master_deg)
                    print(
                        f"  teleop sync: U-Arm aligned (offset max {offset:.3f} rad)",
                        flush=True,
                    )

        if menu_pending_path is not None:
            arm_hold_qpos = _snapshot_arm_qpos(bridge)
            print(
                ">> 菜单/回放等待：从臂锁定在当前姿态，不会跟随 U-Arm（按 n 恢复跟随）",
                flush=True,
            )

        print("Teleoperation active.", flush=True)
        if enable_session and not recording:
            if menu_pending_path is None:
                if quick_record:
                    print(
                        "就绪【快速录制】：按 0/空格 开始；每轮录制前强制 go_home（同启动）；"
                        "保存后归位锁定，再按 n/0/空格/Enter 开下一轮",
                        flush=True,
                    )
                else:
                    print("就绪：按 0 或 Space 开始录制（停止保存：s / Enter）", flush=True)
            else:
                print("就绪：回放菜单已打开，按 r / p / v / l 回放，n 录制新一条", flush=True)

        with KeyPoller() as keys:

            def reset_to_startup_home(label: str) -> Optional[np.ndarray]:
                """Same go_home + gripper + U-Arm sync as process startup."""
                prepare_before_recording(
                    bridge,
                    mapper,
                    servo,
                    startup_go_home_position,
                    open_cmd,
                    control_dt,
                    control_mode=control_mode,
                    max_blend_sec=startup_go_home_blend_max,
                    blend_speed_rad_s=startup_go_home_blend_speed,
                    lock_sec=startup_go_home_lock_sec,
                    sync_uarm=True,
                    gripper_blend_min_sec=startup_gripper_blend_min,
                    gripper_blend_max_sec=startup_gripper_blend_max,
                    gripper_blend_speed=startup_gripper_blend_speed,
                    gripper_settle_sec=startup_gripper_settle,
                    label=label,
                )
                return _snapshot_arm_qpos(bridge)

            def begin_recording(trigger_ch: str) -> None:
                nonlocal recording, record_buffers, record_frame_warned
                nonlocal last_record_time, last_recording_reminder_time, arm_hold_qpos
                nonlocal quick_awaiting_next
                keys.drain_all()
                quick_awaiting_next = False
                # Always reset like startup before each episode when quick-recording,
                # or when go_home_before_each_record is enabled.
                if quick_record or startup_go_home_before_each_record:
                    snap = reset_to_startup_home(f"record-prep ep{episode_idx}")
                    if snap is not None:
                        arm_hold_qpos = snap
                print(
                    f"\n>> [{_key_label(trigger_ch)}] 开始录制 episode_{episode_idx}",
                    flush=True,
                )
                recording = True
                record_frame_warned = False
                record_buffers = {
                    "qpos": [],
                    "eef": [],
                    "qvel": [],
                    "effort": [],
                    "action": [],
                    "master_deg": [],
                    "images": _empty_image_buffers(camera_names),
                    "depth": [],
                    "time": [],
                    "arm_stamp": [],
                }
                last_record_time = 0.0
                last_recording_reminder_time = time.monotonic()
                print_recording_started(episode_idx)
                # Release hold so teleop follows from the freshly synced home mapping.
                arm_hold_qpos = None

            while rclpy.ok():
                loop_start = time.monotonic()
                rclpy.spin_once(bridge, timeout_sec=0.0)
                _handle_save_results()

                if menu_pending_path is not None and not menu_banner_printed:
                    print_post_save_menu(
                        menu_pending_path,
                        camera_names=camera_names,
                        project_dir=project_dir,
                        output_dir=output_dir,
                        record_fps=record_hz,
                    )
                    menu_banner_printed = True

                cv_key = -1
                follow_uarm = _should_follow_uarm(
                    enable_session=enable_session,
                    menu_pending_path=menu_pending_path,
                    follow_during_menu=follow_uarm_during_menu,
                    replaying=replaying,
                    quick_awaiting_next=quick_awaiting_next,
                )
                if follow_uarm:
                    master_deg = servo.get_angles_deg()
                    command = mapper.map_master_deg(master_deg)
                    publish_teleop_command(
                        bridge,
                        command,
                        joint_control_mode=control_mode,
                    )
                    snap = _snapshot_arm_qpos(bridge)
                    if snap is not None:
                        arm_hold_qpos = snap

                    if camera_poller is not None:
                        cv_key = camera_poller.show_previews()
                        rgb_frames, wrist_depth_snap = camera_poller.snapshot()
                        ready = all(rgb_frames.get(n) is not None for n in camera_names)
                        if ready:
                            last_rgb_frames = {
                                n: rgb_frames[n].copy() for n in camera_names  # type: ignore[union-attr]
                            }
                            if args.use_depth:
                                last_wrist_depth = (
                                    None if wrist_depth_snap is None else wrist_depth_snap.copy()
                                )
                        elif recording and not record_frame_warned:
                            missing = [n for n in camera_names if rgb_frames.get(n) is None]
                            print(f"  WARNING: camera frame missing {missing}; skip record step.", flush=True)
                            record_frame_warned = True

                    if args.debug_gripper:
                        now_dbg = time.monotonic()
                        if now_dbg - last_grip_debug_time >= 0.5:
                            s6 = master_deg[grip_idx]
                            print(
                                f"  [grip] s6={s6:+.1f}° rest={mapper._gripper_rest_deg:+.1f}° "
                                f"press={abs(s6 - mapper._gripper_rest_deg):.1f}° "
                                f"cmd={command.gripper:.3f}",
                                flush=True,
                            )
                            last_grip_debug_time = now_dbg

                    if recording:
                        now_mono = time.monotonic()
                        if now_mono - last_recording_reminder_time >= RECORDING_REMINDER_SEC:
                            print_recording_reminder(episode_idx)
                            last_recording_reminder_time = now_mono

                    if recording and record_buffers is not None:
                        now = time.monotonic()
                        if now - last_record_time >= record_period:
                            if _try_append_record_frame(
                                record_buffers,
                                bridge=bridge,
                                command=command,
                                mapper=mapper,
                                camera_names=camera_names,
                                last_rgb_frames=last_rgb_frames,
                                last_wrist_depth=last_wrist_depth,
                                use_depth=args.use_depth,
                                no_camera=args.no_camera,
                            ):
                                last_record_time = now
                                if len(record_buffers["qpos"]) == 1 and last_rgb_frames:
                                    stats = " ".join(
                                        f"{n}={float(last_rgb_frames[n].mean()):.0f}"
                                        for n in camera_names
                                    )
                                    print(f"  first record frame: {stats}", flush=True)
                else:
                    if arm_hold_qpos is None:
                        arm_hold_qpos = _snapshot_arm_qpos(bridge)
                    if arm_hold_qpos is not None:
                        _publish_hold_qpos(
                            bridge,
                            arm_hold_qpos,
                            joint_control_mode=control_mode,
                        )
                    if camera_poller is not None:
                        cv_key = camera_poller.show_previews()
                        rgb_frames, wrist_depth_snap = camera_poller.snapshot()
                        ready = all(rgb_frames.get(n) is not None for n in camera_names)
                        if ready:
                            last_rgb_frames = {
                                n: rgb_frames[n].copy() for n in camera_names  # type: ignore[union-attr]
                            }
                            if args.use_depth:
                                last_wrist_depth = (
                                    None if wrist_depth_snap is None else wrist_depth_snap.copy()
                                )

                if enable_session and not replaying:
                    polled = _collect_keys(keys, cv_key)

                    if recording and _wants_abort(polled):
                        abort_key = next((k for k in polled if k in ABORT_KEYS), "x")
                        frame_count = len(record_buffers["qpos"]) if record_buffers else 0
                        print(
                            f"\n>> [{_key_label(abort_key)}] 放弃当前录制"
                            f"（已丢弃 {frame_count} 帧，未写入 HDF5）…",
                            flush=True,
                        )
                        recording = False
                        record_buffers = None
                        last_recording_reminder_time = 0.0
                        menu_pending_path = None
                        menu_banner_printed = False
                        quick_awaiting_next = False
                        keys.drain_all()
                        print(">> 重置到启动归位，回到开录前状态…", flush=True)
                        reset_to_startup_home(f"abort-home ep{episode_idx}")
                        arm_hold_qpos = None
                        print(
                            f"\n>> 已取消 episode_{episode_idx}。"
                            f"摆好物体后按  0 / 空格  重新开始录制。\n",
                            flush=True,
                        )
                        polled = []

                    elif recording and _wants_stop(polled):
                        stop_key = next((k for k in polled if k in STOP_KEYS), "")
                        stop_label = "Enter" if stop_key in "\r\n" else (stop_key or "s")
                        frame_count = len(record_buffers["qpos"]) if record_buffers else 0
                        if record_buffers is not None and not record_buffers["qpos"]:
                            _try_append_record_frame(
                                record_buffers,
                                bridge=bridge,
                                command=command,
                                mapper=mapper,
                                camera_names=camera_names,
                                last_rgb_frames=last_rgb_frames,
                                last_wrist_depth=last_wrist_depth,
                                use_depth=args.use_depth,
                                no_camera=args.no_camera,
                            )
                            frame_count = len(record_buffers["qpos"])
                        recording = False
                        if record_buffers and record_buffers["qpos"]:
                            saved_path = _make_episode_save_path(
                                output_dir,
                                use_timestamp=filename_with_timestamp,
                                episode_idx=episode_idx,
                            )
                            n_saved = len(record_buffers["qpos"])
                            buffers_to_save = record_buffers
                            record_buffers = None
                            open_menu_when_done = bool(enable_menu and not quick_record)
                            if save_writer is not None:
                                print(
                                    f"\n>> [{stop_label}] 停止录制，{n_saved} 帧交后台保存"
                                    f"（不阻塞，可继续录）…",
                                    flush=True,
                                )
                                save_writer.submit(
                                    _SaveJob(
                                        path=saved_path,
                                        task=task,
                                        buffers=buffers_to_save,
                                        camera_names=camera_names,
                                        use_depth=args.use_depth,
                                        record_fps=record_hz,
                                        episode_label=episode_idx,
                                        open_menu_when_done=open_menu_when_done,
                                    )
                                )
                            else:
                                print(
                                    f"\n>> [{stop_label}] 停止录制，正在保存 {n_saved} 帧…",
                                    flush=True,
                                )
                                write_hdf5(
                                    saved_path,
                                    task,
                                    buffers_to_save,
                                    camera_names,
                                    use_depth=args.use_depth,
                                    record_fps=record_hz,
                                )
                                session_saved_paths.append(saved_path)
                                print_episode_saved(
                                    saved_path,
                                    n_saved,
                                    camera_names=camera_names,
                                    project_dir=project_dir,
                                    output_dir=output_dir,
                                    record_fps=record_hz,
                                )
                                last_saved_for_menu = saved_path
                                if open_menu_when_done:
                                    menu_pending_path = saved_path
                                    menu_banner_printed = False
                                    arm_hold_qpos = _snapshot_arm_qpos(bridge)
                                    print(
                                        ">> 菜单期间从臂锁定在当前姿态，不会跟随 U-Arm",
                                        flush=True,
                                    )
                            episode_idx += 1
                            last_saved_for_menu = saved_path
                            if quick_record:
                                menu_pending_path = None
                                menu_banner_printed = False
                                print(
                                    ">> 快速录制：重置到启动归位（HDF5 后台写，可马上开下一条）…",
                                    flush=True,
                                )
                                snap = reset_to_startup_home(
                                    f"post-save-home ep{episode_idx - 1}"
                                )
                                arm_hold_qpos = snap
                                quick_awaiting_next = True
                                print_quick_next_prompt(
                                    saved_path,
                                    episode_idx,
                                    saving=save_writer is not None,
                                )
                            elif enable_menu and save_writer is not None:
                                print(
                                    ">> HDF5 后台写入中；写完后自动打开回放菜单。"
                                    "也可先按 0/空格 开录下一条（菜单会延后）。",
                                    flush=True,
                                )
                        else:
                            print("尚无有效帧（请录制至少半秒），未保存。", flush=True)
                            record_buffers = None
                        last_recording_reminder_time = 0.0
                        keys.drain_crlf()
                        _warn_ignored_keys(keys, "停录")
                        polled = [k for k in polled if k not in STOP_KEYS]

                    menu_handled = False
                    if quick_awaiting_next and not recording and menu_pending_path is None:
                        quick_key = next(
                            (
                                k
                                for k in polled
                                if k in QUICK_NEXT_KEYS
                                or k in {"m", "M", "q", "Q", "\x03"}
                            ),
                            None,
                        )
                        if quick_key is not None:
                            if quick_key in QUICK_NEXT_KEYS:
                                begin_recording(quick_key)
                                menu_handled = True
                                polled = []
                            elif quick_key in {"m", "M"}:
                                keys.drain_all()
                                if last_saved_for_menu is None:
                                    print(
                                        f"\n>> [{_key_label(quick_key)}] 尚无已保存 episode，无法打开菜单",
                                        flush=True,
                                    )
                                else:
                                    quick_awaiting_next = False
                                    menu_pending_path = last_saved_for_menu
                                    menu_banner_printed = False
                                    arm_hold_qpos = _snapshot_arm_qpos(bridge)
                                    print(
                                        f"\n>> [{_key_label(quick_key)}] 打开回放菜单",
                                        flush=True,
                                    )
                                menu_handled = True
                                polled = []
                            elif quick_key in {"q", "Q", "\x03"}:
                                print(
                                    f"\n>> [{_key_label(quick_key)}] 退出",
                                    flush=True,
                                )
                                raise KeyboardInterrupt

                    if menu_pending_path is not None:
                        menu_action = _pick_first_menu_key(polled)
                        if menu_action in {"r", "R", "p", "P", "v", "V", "l", "L"}:
                            extra = _count_meaningful_keys(
                                [c for c in polled if c != menu_action and c in MENU_KEYS]
                            )
                            if extra:
                                print(
                                    f">> 忽略同批重复菜单键 ({extra} 个)，只执行 [{_key_label(menu_action)}]",
                                    flush=True,
                                )
                            replaying = True
                            try:
                                held = _run_replay_action(
                                    keys,
                                    bridge=bridge,
                                    mapper=mapper,
                                    servo=servo,
                                    hdf5_path=menu_pending_path,
                                    action=menu_action,
                                    replay_hz=replay_hz,
                                    record_hz=record_hz,
                                    camera_names=camera_names,
                                    project_dir=project_dir,
                                    output_dir=output_dir,
                                    replay_smooth_window=replay_smooth_window,
                                    replay_blend_sec=replay_blend_sec,
                                    replay_smooth_gripper=replay_smooth_gripper,
                                    replay_max_blend_sec=replay_max_blend_sec,
                                    replay_blend_skip_tol=replay_blend_skip_tol,
                                    replay_blend_speed=replay_blend_speed,
                                    replay_post_hold_sec=replay_post_hold_sec,
                                    replay_sync_uarm=replay_sync_uarm,
                                    replay_eef_smooth_window=replay_eef_smooth_window,
                                    replay_eef_max_rpy_step_rad=replay_eef_max_rpy_step_rad,
                                    replay_eef_ik_seed=replay_eef_ik_seed,
                                    replay_orientation_interp=replay_orientation_interp,
                                    replay_eef_source=replay_eef_source,
                                    replay_eef_ik_seed_smooth_window=replay_eef_ik_seed_smooth_window,
                                    replay_eef_quat_smooth_window=replay_eef_quat_smooth_window,
                                    replay_auto_bake_eef_fk=replay_auto_bake_eef_fk,
                                    replay_bake_hz=replay_bake_hz,
                                    replay_bake_smooth_window=replay_bake_smooth_window,
                                    replay_bake_quat_smooth_window=replay_bake_quat_smooth_window,
                                    replay_return_to_start=replay_return_to_start,
                                    replay_return_blend_sec=replay_return_blend_sec,
                                    replay_return_max_blend_sec=replay_return_max_blend_sec,
                                    replay_return_settle_sec=replay_return_settle_sec,
                                    replay_return_hold_sec=replay_return_hold_sec,
                                    replay_return_tol_rad=replay_return_tol_rad,
                                    replay_start_max_blend_sec=replay_start_max_blend_sec,
                                    replay_start_settle_sec=replay_start_settle_sec,
                                    replay_start_snap_sec=replay_start_snap_sec,
                                    replay_eef_joint_lead_sec=replay_eef_joint_lead_sec,
                                    replay_force_rebake_eef_fk=replay_force_rebake_eef_fk,
                                    replay_eef_joint_fallback_enabled=replay_eef_joint_fallback_enabled,
                                    replay_eef_joint_fallback_step_rad=replay_eef_joint_fallback_step_rad,
                                    replay_eef_joint_fallback_hold_sec=replay_eef_joint_fallback_hold_sec,
                                    replay_eef_joint_fallback_lookahead_sec=replay_eef_joint_fallback_lookahead_sec,
                                    replay_singularity_adjust=replay_singularity_adjust,
                                    replay_singularity_joint_fallback=replay_singularity_joint_fallback,
                                    replay_eef_ik_seed_closed_loop=replay_eef_ik_seed_closed_loop,
                                    replay_eef_ik_seed_hint_blend=replay_eef_ik_seed_hint_blend,
                                    replay_singularity_config=replay_singularity_config,
                                )
                                if held is not None:
                                    arm_hold_qpos = held
                                    print(
                                        ">> 回放结束：从臂已锁定在归位姿态，菜单期间不跟随 U-Arm",
                                        flush=True,
                                    )
                            finally:
                                replaying = False
                            menu_handled = True
                        elif menu_action in {"n", "N"}:
                            keys.drain_all()
                            menu_pending_path = None
                            menu_banner_printed = False
                            if quick_record:
                                print(
                                    f"\n>> [{_key_label(menu_action)}] 重置到启动归位…",
                                    flush=True,
                                )
                                snap = reset_to_startup_home(
                                    f"menu-n-home ep{episode_idx}"
                                )
                                arm_hold_qpos = snap
                                quick_awaiting_next = True
                                print_quick_next_prompt(
                                    last_saved_for_menu or Path("episode.hdf5"),
                                    episode_idx,
                                )
                            else:
                                arm_hold_qpos = None
                                offset = mapper.align_to_follower(
                                    bridge.latest_state.joint_pos
                                    if bridge.latest_state
                                    else np.zeros(7),
                                    servo.get_angles_deg(),
                                )
                                print(
                                    f"\n>> [{_key_label(menu_action)}] 可录制新一条：0/Space 开始。"
                                    f"U-Arm 已同步 (offset max {offset:.3f} rad)，可缓慢移动主臂。",
                                    flush=True,
                                )
                            menu_handled = True
                        elif menu_action in {"q", "Q", "\x03"}:
                            print(f"\n>> [{_key_label(menu_action)}] 退出", flush=True)
                            raise KeyboardInterrupt
                        elif any(ch in "\r\n" for ch in polled):
                            keys.drain_all()
                            print(
                                ">> [Enter] 保存后菜单请按 r / l / n / q（Enter 无效）",
                                flush=True,
                            )
                            menu_handled = True

                    if not menu_handled:
                        for ch in polled:
                            if ch in STOP_KEYS:
                                continue

                            if ch in START_KEYS and not recording and menu_pending_path is None:
                                begin_recording(ch)
                            elif ch == "d" and not recording and menu_pending_path is None:
                                keys.drain_all()
                                print(f"\n>> [{_key_label(ch)}] 删除上一条", flush=True)
                                if not session_saved_paths:
                                    print("本会话尚无已保存 episode。", flush=True)
                                else:
                                    target = session_saved_paths.pop()
                                    if target.exists():
                                        target.unlink()
                                        episode_idx = max(0, episode_idx - 1)
                                        print(f"Deleted {target}", flush=True)
                                    else:
                                        print(f"Not found: {target}", flush=True)
                            elif ch in {"q", "\x03"} and menu_pending_path is None and not recording:
                                print(f"\n>> [{_key_label(ch)}] 退出", flush=True)
                                raise KeyboardInterrupt

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, control_dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopping teleoperation.", flush=True)
    finally:
        if save_writer is not None:
            save_writer.shutdown()
        if camera_poller is not None:
            camera_poller.stop()
        servo.close()
        if camera is not None:
            camera.close()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
