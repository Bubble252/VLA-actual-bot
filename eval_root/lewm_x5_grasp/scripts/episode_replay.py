"""Replay saved teleop/teach HDF5 episodes on ARX X5.

Control modes:
  joint         (mode=5): replay recorded joint angles (default).
  eef           (mode=4): replay end-effector xyz+rpy; ARX SDK KDL LMA damped IK + mode4_guard.
  eef_via_joint (mode=5): replay EEF via client differential IK → joints (avoids mode=4 singularity).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lerobot_episode_io import compare_hdf5_parquet, hdf5_to_parquet, load_hdf5_episode, load_parquet_episode

from lewm_x5_grasp.arm_ros2 import X5ArmBridge
from lewm_x5_grasp.eef_to_joint_ik import differential_ik
from lewm_x5_grasp.schemas import ArmCommand
from lewm_x5_grasp.singularity_adjuster import SingularityAdjuster, SingularityConfig

MODE_GO_HOME = 1
MODE_END_CONTROL = 4
MODE_POSITION_CONTROL = 5
ReplayControlMode = Literal["joint", "eef", "eef_via_joint"]
ActionSpace = Literal["joint", "eef"]
EefSource = Literal["auto", "action", "observations", "eef_fk"]
IkSeedSource = Literal["action", "state"]
OrientationInterp = Literal["slerp", "euler"]
GRIPPER_JOINT_IDX = 6
ARM_JOINT_DIMS = 6


def _uses_eef_trajectory(control_mode: ReplayControlMode) -> bool:
    return control_mode in ("eef", "eef_via_joint")


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rate_limit_joints(
    cur_q: Sequence[float], target_q: Sequence[float], max_delta_rad: float
) -> np.ndarray:
    cur = np.asarray(cur_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    tgt = np.asarray(target_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    if max_delta_rad <= 0:
        return tgt.copy()
    out = cur.copy()
    for j in range(ARM_JOINT_DIMS):
        d = float(np.arctan2(np.sin(tgt[j] - cur[j]), np.cos(tgt[j] - cur[j])))
        out[j] = cur[j] + float(np.clip(d, -max_delta_rad, max_delta_rad))
    return out


def _ema_joints(prev_q: Sequence[float], new_q: Sequence[float], alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    prev = np.asarray(prev_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    new = np.asarray(new_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    if a >= 1.0 - 1e-9:
        return new.copy()
    if a <= 1e-9:
        return prev.copy()
    out = prev.copy()
    for j in range(ARM_JOINT_DIMS):
        d = float(np.arctan2(np.sin(new[j] - prev[j]), np.cos(new[j] - prev[j])))
        out[j] = prev[j] + a * d
    return out


class EefViaJointStreamer:
    """Stateful EEF frame stream → differential IK → joint commands (mode=5).

    Defaults for ``ik_max_dq`` / ``max_joint_delta_rad`` / ``joint_ema`` / ``ik_damp``
    come from ``x5_play_params.yaml`` (same file as eval ``eef_via_joint``).
    """

    def __init__(
        self,
        *,
        max_joint_delta_rad: Optional[float] = None,
        joint_ema: Optional[float] = None,
        ik_max_dq: Optional[float] = None,
        ik_damp: Optional[float] = None,
    ) -> None:
        try:
            from x5_play_viz.eef_offset import get_diff_ik_control_params

            p = get_diff_ik_control_params()
        except Exception:
            p = {
                "ik_max_dq": 0.12,
                "max_joint_delta_rad": 0.08,
                "joint_ema": 0.55,
                "ik_damp": 0.02,
                "yaml_path": "",
            }
        self.max_joint_delta_rad = float(
            p["max_joint_delta_rad"]
            if max_joint_delta_rad is None
            else max_joint_delta_rad
        )
        self.joint_ema = float(p["joint_ema"] if joint_ema is None else joint_ema)
        self.ik_max_dq = float(p["ik_max_dq"] if ik_max_dq is None else ik_max_dq)
        self.ik_damp = float(p["ik_damp"] if ik_damp is None else ik_damp)
        self._yaml_path = str(p.get("yaml_path", ""))
        self.last_q: Optional[np.ndarray] = None
        self.prev_eef: Optional[np.ndarray] = None
        self.last_info: dict = {"ok": True, "jump": 0.0}

    def reset(self, seed_q6: Sequence[float]) -> None:
        self.last_q = np.asarray(seed_q6, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS].copy()
        self.prev_eef = None
        self.last_info = {"ok": True, "jump": 0.0}

    def step(self, eef_row: np.ndarray, qpos_row: np.ndarray) -> np.ndarray:
        eef6 = np.asarray(eef_row, dtype=np.float64).reshape(-1)[:6]
        if self.last_q is None:
            self.last_q = np.asarray(qpos_row, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS].copy()
        if self.prev_eef is None:
            self.prev_eef = eef6.copy()
            self.last_info = {"ok": True, "jump": 0.0, "mode": "diff"}
            return self.last_q.copy()
        deef = eef6 - self.prev_eef
        deef[3:6] = _wrap_pi(deef[3:6])
        q_ik, info = differential_ik(
            self.last_q,
            deef,
            damp=self.ik_damp,
            max_dq=self.ik_max_dq,
        )
        q_lim = _rate_limit_joints(self.last_q, q_ik, self.max_joint_delta_rad)
        q_cmd = _ema_joints(self.last_q, q_lim, self.joint_ema)
        self.last_q = q_cmd.copy()
        self.prev_eef = eef6.copy()
        self.last_info = info
        return q_cmd


def _pose_distance_qpos(a: np.ndarray, b: np.ndarray) -> float:
    a6 = np.asarray(a[:ARM_JOINT_DIMS], dtype=np.float64)
    b6 = np.asarray(b[:ARM_JOINT_DIMS], dtype=np.float64)
    delta = np.arctan2(np.sin(b6 - a6), np.cos(b6 - a6))
    return float(np.max(np.abs(delta)))


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _smoothstep(alpha: float) -> float:
    return smoothstep(alpha)


def _read_arm_pose(bridge: X5ArmBridge) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return None, None
    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    cur_eef = np.concatenate(
        [np.asarray(state.end_pos[:6], dtype=np.float64), cur_qpos[6:7]]
    )
    return cur_eef, cur_qpos


def _log_joint_residual(
    label: str,
    cur_qpos: np.ndarray,
    target_qpos: np.ndarray,
) -> float:
    delta = np.arctan2(
        np.sin(np.asarray(target_qpos[:ARM_JOINT_DIMS]) - np.asarray(cur_qpos[:ARM_JOINT_DIMS])),
        np.cos(np.asarray(target_qpos[:ARM_JOINT_DIMS]) - np.asarray(cur_qpos[:ARM_JOINT_DIMS])),
    )
    worst = int(np.argmax(np.abs(delta)))
    max_deg = float(np.max(np.abs(np.degrees(delta))))
    print(
        f"  {label}: joint residual j{worst}={np.degrees(delta[worst]):+.2f}° "
        f"(max {max_deg:.2f}°)",
        flush=True,
    )
    return float(np.max(np.abs(delta)))


def _blend_qpos_angular(
    cur_qpos: np.ndarray,
    target_qpos: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shortest-path joint interpolation (gripper linear)."""
    cur = np.asarray(cur_qpos, dtype=np.float64)[:7]
    target = np.asarray(target_qpos, dtype=np.float64)[:7]
    out = cur.copy()
    for j in range(ARM_JOINT_DIMS):
        delta = np.arctan2(np.sin(target[j] - cur[j]), np.cos(target[j] - cur[j]))
        out[j] = cur[j] + float(alpha) * delta
    out[GRIPPER_JOINT_IDX] = cur[GRIPPER_JOINT_IDX] + float(alpha) * (
        target[GRIPPER_JOINT_IDX] - cur[GRIPPER_JOINT_IDX]
    )
    return out


def snap_to_trajectory_home(
    bridge: X5ArmBridge,
    target_qpos: np.ndarray,
    period: float,
    *,
    label: str = "snap-home",
    blend_speed_rad_s: float = 0.6,
    max_blend_sec: float = 6.0,
    lock_sec: float = 2.5,
    tol_rad: float = 0.003,
) -> float:
    """Move to exact trajectory frame-0 using joint-only mode=5 commands.

    Homing must NOT send observations/eef as end_pos — that fights joint targets
    and leaves a persistent offset (~1–2°) even when joints look close.
    """
    target_qpos = np.asarray(target_qpos, dtype=np.float64)[:7]
    joints6 = target_qpos[:ARM_JOINT_DIMS].tolist()
    gripper = float(target_qpos[GRIPPER_JOINT_IDX])

    _, cur_qpos = _read_arm_pose(bridge)
    if cur_qpos is None:
        return float("inf")

    dist = _pose_distance_qpos(cur_qpos, target_qpos)
    print(
        f"  {label}: offset {dist:.3f} rad ({np.degrees(dist):.2f}° max-joint) "
        f"-> exact frame-0 joints (mode=5, joint-only)",
        flush=True,
    )

    if dist > 0.002:
        duration = adaptive_blend_duration(
            dist,
            min_sec=0.5,
            max_sec=max_blend_sec,
            speed=blend_speed_rad_s,
        )
        n_blend = max(5, int(duration / period))
        print(
            f"  {label}: angular blend over {duration:.2f}s ({n_blend} steps)",
            flush=True,
        )
        for k in range(1, n_blend + 1):
            t0 = time.monotonic()
            alpha = smoothstep(k / n_blend)
            qpos_b = _blend_qpos_angular(cur_qpos, target_qpos, alpha)
            bridge.publish_joint_command(
                qpos_b[:ARM_JOINT_DIMS].tolist(),
                float(qpos_b[GRIPPER_JOINT_IDX]),
                mode=MODE_POSITION_CONTROL,
            )
            rclpy.spin_once(bridge, timeout_sec=0.0)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    n_lock = max(1, int(lock_sec / period))
    print(
        f"  {label}: lock exact joints {lock_sec:.1f}s ({n_lock} steps, no eef cmd)",
        flush=True,
    )
    for _ in range(n_lock):
        t0 = time.monotonic()
        bridge.publish_joint_command(joints6, gripper, mode=MODE_POSITION_CONTROL)
        rclpy.spin_once(bridge, timeout_sec=0.0)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    _, cur_qpos = _read_arm_pose(bridge)
    if cur_qpos is None:
        return float("inf")
    residual = _pose_distance_qpos(cur_qpos, target_qpos)
    _log_joint_residual(f"{label} final", cur_qpos, target_qpos)
    if residual <= tol_rad:
        print(f"  {label}: OK (residual {residual:.4f} rad)", flush=True)
    else:
        print(
            f"  {label}: WARNING residual {residual:.4f} rad "
            f"(>{tol_rad:.4f}); joints commanded exactly, check controller tracking",
            flush=True,
        )
    return residual


def _return_to_trajectory_start(
    bridge: X5ArmBridge,
    start_eef: np.ndarray,
    start_qpos: np.ndarray,
    period: float,
    *,
    blend_sec: float = 1.5,
    max_blend_sec: float = 6.0,
    blend_speed_rad_s: float = 0.8,
    settle_sec: float = 1.5,
    hold_sec: float = 1.0,
    tol_rad: float = 0.04,
    label: str = "return-to-start",
) -> float:
    del start_eef, blend_sec, settle_sec
    print(f"  {label}: 回到轨迹起点（原位）…", flush=True)
    lock_sec = max(2.5, hold_sec)
    return snap_to_trajectory_home(
        bridge,
        start_qpos,
        period,
        label=label,
        blend_speed_rad_s=blend_speed_rad_s,
        max_blend_sec=max_blend_sec,
        lock_sec=lock_sec,
        tol_rad=min(tol_rad, 0.003),
    )


def adaptive_blend_duration(
    delta: float,
    *,
    min_sec: float,
    max_sec: float,
    speed: float,
) -> float:
    return min(max_sec, max(min_sec, abs(float(delta)) / max(float(speed), 1e-6)))


def startup_blend_to_home(
    bridge: X5ArmBridge,
    home_joints6: Sequence[float],
    period: float,
    *,
    control_mode: int = MODE_POSITION_CONTROL,
    min_sec: float = 0.8,
    max_sec: float = 4.0,
    speed: float = 0.6,
    skip_tol: float = 0.05,
) -> bool:
    """Adaptively blend 6 arm joints to go_home_position; gripper unchanged."""
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return False

    home6 = np.asarray(home_joints6, dtype=np.float64)[:ARM_JOINT_DIMS]
    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    target_qpos = cur_qpos.copy()
    target_qpos[:ARM_JOINT_DIMS] = home6

    dist = _pose_distance_qpos(cur_qpos, target_qpos)
    if dist <= skip_tol:
        print(f"  go_home: skip (joint delta {dist:.3f} rad <= {skip_tol:.3f})", flush=True)
        return False

    duration = adaptive_blend_duration(dist, min_sec=min_sec, max_sec=max_sec, speed=speed)
    n_steps = max(5, int(duration / period))
    print(
        f"  go_home blend: {dist:.3f} rad -> home over {duration:.2f}s ({n_steps} steps)",
        flush=True,
    )
    grip_hold = float(cur_qpos[GRIPPER_JOINT_IDX])
    for k in range(1, n_steps + 1):
        t0 = time.monotonic()
        alpha = smoothstep(k / n_steps)
        qpos_b = cur_qpos * (1.0 - alpha) + target_qpos * alpha
        bridge.publish_joint_command(
            joint_pos_6=qpos_b[:ARM_JOINT_DIMS].tolist(),
            gripper=grip_hold,
            mode=control_mode,
        )
        rclpy.spin_once(bridge, timeout_sec=0.0)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    return True


def startup_blend_gripper_open(
    bridge: X5ArmBridge,
    open_cmd: float,
    period: float,
    *,
    control_mode: int = MODE_POSITION_CONTROL,
    min_sec: float = 0.5,
    max_sec: float = 3.0,
    speed: float = 2.0,
    settle_sec: float = 1.0,
) -> None:
    """Gradually open gripper from current /arm_status, then hold at open_cmd."""
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return

    g0 = float(state.joint_pos[6])
    joints = state.joint_pos[:6].tolist()
    delta = abs(open_cmd - g0)
    duration = adaptive_blend_duration(delta, min_sec=min_sec, max_sec=max_sec, speed=speed)
    n_blend = max(1, int(duration / period))
    print(
        f"  gripper blend: {g0:.3f} -> {open_cmd:.3f} over {duration:.2f}s ({n_blend} steps)",
        flush=True,
    )
    for k in range(1, n_blend + 1):
        t0 = time.monotonic()
        alpha = smoothstep(k / n_blend)
        grip = g0 * (1.0 - alpha) + open_cmd * alpha
        bridge.publish_joint_command(
            joint_pos_6=joints,
            gripper=float(grip),
            mode=control_mode,
        )
        rclpy.spin_once(bridge, timeout_sec=0.0)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    if settle_sec > 0.0:
        n_settle = max(1, int(settle_sec / period))
        print(f"  gripper settle at {open_cmd:.3f} for {settle_sec:.1f}s …", flush=True)
        for _ in range(n_settle):
            t0 = time.monotonic()
            bridge.publish_joint_command(
                joint_pos_6=joints,
                gripper=float(open_cmd),
                mode=control_mode,
            )
            rclpy.spin_once(bridge, timeout_sec=0.0)
            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)


def startup_prepare_teleop(
    bridge: X5ArmBridge,
    mapper: Any,
    servo: Any,
    period: float,
    *,
    control_mode: int = MODE_POSITION_CONTROL,
    sync_uarm: bool = True,
    arm_blend_sec: float = 0.5,
    arm_blend_max_sec: float = 2.0,
    arm_blend_speed: float = 0.8,
    arm_blend_skip_tol: float = 0.05,
) -> None:
    """Sync U-Arm mapping to follower and softly blend arm if still misaligned."""
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return

    master_deg = servo.get_angles_deg()
    if sync_uarm:
        offset = mapper.align_to_follower(state.joint_pos, master_deg)
        print(
            f"  teleop sync: U-Arm aligned to follower (runtime offset max {offset:.3f} rad)",
            flush=True,
        )

    command = mapper.map_master_deg(master_deg)
    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    target_qpos = mapper.to_qpos7(command)
    dist = _pose_distance_qpos(cur_qpos, target_qpos)
    if dist <= arm_blend_skip_tol:
        print(f"  arm blend: skip (joint delta {dist:.3f} rad)", flush=True)
        return

    duration = adaptive_blend_duration(
        dist, min_sec=arm_blend_sec, max_sec=arm_blend_max_sec, speed=arm_blend_speed
    )
    n_steps = max(5, int(duration / period))
    print(
        f"  arm blend: {dist:.3f} rad over {duration:.2f}s ({n_steps} steps) before teleop",
        flush=True,
    )
    for k in range(1, n_steps + 1):
        t0 = time.monotonic()
        alpha = smoothstep(k / n_steps)
        qpos_b = cur_qpos * (1.0 - alpha) + target_qpos * alpha
        bridge.publish_joint_command(
            joint_pos_6=qpos_b[:ARM_JOINT_DIMS].tolist(),
            gripper=float(qpos_b[GRIPPER_JOINT_IDX]),
            mode=control_mode,
        )
        rclpy.spin_once(bridge, timeout_sec=0.0)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)
    mapper.align_to_follower(
        bridge.latest_state.joint_pos if bridge.latest_state else target_qpos,
        master_deg,
    )


def prepare_before_recording(
    bridge: X5ArmBridge,
    mapper: Any,
    servo: Any,
    home_joints6: Sequence[float],
    open_cmd: float,
    period: float,
    *,
    control_mode: int = MODE_POSITION_CONTROL,
    max_blend_sec: float = 6.0,
    blend_speed_rad_s: float = 0.6,
    lock_sec: float = 2.5,
    tol_rad: float = 0.003,
    sync_uarm: bool = True,
    gripper_blend_min_sec: float = 0.5,
    gripper_blend_max_sec: float = 3.0,
    gripper_blend_speed: float = 2.0,
    gripper_settle_sec: float = 1.0,
    label: str = "record-prep",
) -> float:
    """Mandatory go_home snap before each recording — never skip by tolerance.

    Keeps the follower at go_home and only re-maps U-Arm offsets (no blend toward
    master), so the arm is not pulled away from home when the master is elsewhere.
    """
    home6 = np.asarray(home_joints6, dtype=np.float64)[:ARM_JOINT_DIMS]
    target_qpos = np.concatenate([home6, [float(open_cmd)]])

    print(f"\n>> [{label}] 从臂归位 go_home（强制 snap + 锁定 {lock_sec:.1f}s）…", flush=True)
    residual = snap_to_trajectory_home(
        bridge,
        target_qpos,
        period,
        label=label,
        blend_speed_rad_s=blend_speed_rad_s,
        max_blend_sec=max_blend_sec,
        lock_sec=lock_sec,
        tol_rad=tol_rad,
    )

    startup_blend_gripper_open(
        bridge,
        open_cmd,
        period,
        control_mode=control_mode,
        min_sec=gripper_blend_min_sec,
        max_sec=gripper_blend_max_sec,
        speed=gripper_blend_speed,
        settle_sec=gripper_settle_sec,
    )
    mapper.reset_gripper_state(open_cmd)

    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if sync_uarm and state is not None:
        master_deg = servo.get_angles_deg()
        offset = mapper.align_to_follower(state.joint_pos, master_deg)
        print(
            f"  {label}: U-Arm 已同步到从臂归位姿态 (offset max {offset:.3f} rad)；"
            f"从臂保持 go_home，请再移动主臂开始示教",
            flush=True,
        )
    print(f">> [{label}] 归位完成", flush=True)
    return residual


def read_hdf5_meta(hdf5_path: Path) -> Dict[str, str]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        return {
            "action_space": str(f.attrs.get("action_space", "joint")),
            "teleop_control_mode": str(f.attrs.get("teleop_control_mode", "joint")),
        }



def _interp_series(t_out: np.ndarray, t_rel: np.ndarray, series: np.ndarray) -> np.ndarray:
    return np.interp(t_out, t_rel, np.asarray(series, dtype=np.float64))


def _interp_angle_series(t_out: np.ndarray, t_rel: np.ndarray, series: np.ndarray) -> np.ndarray:
    angles = np.unwrap(np.asarray(series, dtype=np.float64))
    return np.interp(t_out, t_rel, angles)


def _euler_xyz_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic fixed XYZ (matches scipy 'xyz'): R = Rz(y) @ Ry(p) @ Rx(r)."""
    r, p, y = [float(v) for v in rpy[:3]]
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _quat_normalize(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / s
    w1 = np.sin(t * theta) / s
    return _quat_normalize(w0 * q0 + w1 * q1)


def _quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    x, y, z, w = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _interp_rpy_slerp(t_out: np.ndarray, t_rel: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float64)
    if len(rpy) < 2:
        return np.tile(rpy[0], (len(t_out), 1)) if len(rpy) else np.zeros((len(t_out), 3))
    quats = np.stack([_euler_xyz_to_quat(rpy[i]) for i in range(len(rpy))], axis=0)
    out = np.zeros((len(t_out), 3), dtype=np.float64)
    for i, t in enumerate(t_out):
        j = int(np.searchsorted(t_rel, t, side="right") - 1)
        j = int(np.clip(j, 0, len(t_rel) - 2))
        span = max(float(t_rel[j + 1] - t_rel[j]), 1e-9)
        alpha = float(np.clip((t - t_rel[j]) / span, 0.0, 1.0))
        q = _quat_slerp(quats[j], quats[j + 1], alpha)
        out[i] = _quat_to_euler_xyz(q)
    return out


def hdf5_has_eef_fk(hdf5_path: Path) -> bool:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        obs = f.get("observations")
        return obs is not None and "eef_fk" in obs


def _clip_angle_step(current: float, target: float, max_step: float) -> float:
    delta = float(np.arctan2(np.sin(target - current), np.cos(target - current)))
    if abs(delta) <= max_step:
        return float(target)
    return float(current + np.sign(delta) * max_step)


def limit_eef_orientation_steps(eef: np.ndarray, max_rpy_step_rad: float) -> np.ndarray:
    """Rate-limit roll/pitch/yaw between consecutive replay frames."""
    if max_rpy_step_rad <= 0 or len(eef) < 2 or eef.shape[1] < 6:
        return eef
    out = eef.copy()
    for i in range(1, len(out)):
        for j in range(3, 6):
            out[i, j] = _clip_angle_step(float(out[i - 1, j]), float(out[i, j]), max_rpy_step_rad)
    return out


def smooth_joints_only(
    qpos: np.ndarray,
    window: int,
    *,
    smooth_gripper: bool = False,
) -> np.ndarray:
    """Moving-average smooth arm joints only (IK seed path)."""
    dummy = qpos.copy()
    _, qpos_out = smooth_trajectory(
        dummy,
        qpos,
        window,
        smooth_gripper=smooth_gripper,
    )
    return qpos_out


def smooth_eef_orientation_quat(eef: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smooth orientations on SO(3) via quaternion blending."""
    if window <= 1 or len(eef) < 3 or eef.shape[1] < 6:
        return eef
    win = min(int(window), len(eef))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return eef

    pad = win // 2
    out = eef.copy()
    quats = np.stack([_euler_xyz_to_quat(eef[i, 3:6]) for i in range(len(eef))], axis=0)
    for i in range(len(eef)):
        lo = max(0, i - pad)
        hi = min(len(eef), i + pad + 1)
        ref = quats[i]
        acc = np.zeros(4, dtype=np.float64)
        for j in range(lo, hi):
            q = quats[j].copy()
            if float(np.dot(q, ref)) < 0.0:
                q = -q
            acc += q
        out[i, 3:6] = _quat_to_euler_xyz(_quat_normalize(acc))
    return out


def _blend_control_mode(control_mode: ReplayControlMode) -> ReplayControlMode:
    """EEF blends/holds use joint mode to avoid euler+IK fights near singularities."""
    return "joint" if _uses_eef_trajectory(control_mode) else control_mode


def plan_eef_replay_frame_modes(
    eef_i: np.ndarray,
    qpos_i: np.ndarray,
    period: float,
    *,
    lead_steps: int = 0,
    fallback_enabled: bool = True,
    step_thresh_rad: float = 0.015,
    hold_sec: float = 0.8,
    lookahead_sec: float = 0.2,
    use_rpy_trigger: bool = True,
) -> tuple[list[ReplayControlMode], dict[str, int]]:
    """Precompute per-frame joint vs eef mode for EEF replay.

    Fast lifts (large joint or orientation steps) fall back to joint mode=5 to
    avoid IK branch jumps. ``lead_steps`` covers the fixed lead-in; fallback
    handles motion onsets later in the trajectory (e.g. lift after a long hold).
    """
    n = len(qpos_i)
    modes: list[ReplayControlMode] = ["eef"] * n
    if n == 0:
        return modes, {"joint_frames": 0, "total": 0, "lead_steps": lead_steps}

    lead_steps = max(0, min(int(lead_steps), n))
    for i in range(lead_steps):
        modes[i] = "joint"

    if not fallback_enabled or n < 2:
        joint_frames = sum(1 for m in modes if m == "joint")
        return modes, {"joint_frames": joint_frames, "total": n, "lead_steps": lead_steps}

    steps = np.zeros(n, dtype=np.float64)
    rpy_steps = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        steps[i] = float(np.linalg.norm(qpos_i[i, :6] - qpos_i[i - 1, :6]))
        if eef_i.shape[1] >= 6:
            rpy_steps[i] = float(np.linalg.norm(eef_i[i, 3:6] - eef_i[i - 1, 3:6]))

    hold_steps = max(1, int(hold_sec / max(period, 1e-6)))
    look_steps = max(1, int(lookahead_sec / max(period, 1e-6)))
    trigger = np.zeros(n, dtype=bool)
    for i in range(1, n):
        hot = steps[i] > step_thresh_rad
        if use_rpy_trigger:
            hot = hot or rpy_steps[i] > step_thresh_rad
        if hot:
            lo = max(0, i - look_steps)
            hi = min(n, i + hold_steps)
            trigger[lo:hi] = True

    for i in range(n):
        if modes[i] == "joint" or trigger[i]:
            modes[i] = "joint"

    joint_frames = sum(1 for m in modes if m == "joint")
    return modes, {
        "joint_frames": joint_frames,
        "total": n,
        "lead_steps": lead_steps,
        "fallback_frames": max(0, joint_frames - lead_steps),
    }


def smooth_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    window: int,
    *,
    smooth_gripper: bool = False,
    rpy_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-average smooth; RPY uses unwrap to avoid pi jumps."""
    if window <= 1 or len(qpos) < 3:
        return eef, qpos

    win = min(int(window), len(qpos))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return eef, qpos

    rpy_win = min(int(rpy_window or win), len(qpos))
    if rpy_win % 2 == 0:
        rpy_win -= 1
    rpy_win = max(rpy_win, 3 if len(qpos) >= 3 else 1)

    pad = win // 2
    rpy_pad = rpy_win // 2
    kernel = np.ones(win, dtype=np.float64) / win
    rpy_kernel = np.ones(rpy_win, dtype=np.float64) / rpy_win

    def _smooth_1d(series: np.ndarray, *, angular: bool = False, pad_size: int | None = None, kern=None) -> np.ndarray:
        p = pad if pad_size is None else pad_size
        k = kernel if kern is None else kern
        values = np.unwrap(series) if angular else np.asarray(series, dtype=np.float64)
        padded = np.pad(values, (p, p), mode="edge")
        return np.convolve(padded, k, mode="valid")

    def _smooth_cols(
        data: np.ndarray,
        col_indices: range,
        *,
        angular: bool = False,
        pad_size: int | None = None,
        kern=None,
    ) -> np.ndarray:
        out = data.copy()
        for d in col_indices:
            if d < data.shape[1]:
                out[:, d] = _smooth_1d(
                    data[:, d], angular=angular, pad_size=pad_size, kern=kern
                )
        return out

    joint_cols = range(0, min(ARM_JOINT_DIMS, qpos.shape[1]))
    qpos_out = _smooth_cols(qpos, joint_cols, angular=True)
    if smooth_gripper and qpos.shape[1] > GRIPPER_JOINT_IDX:
        qpos_out[:, GRIPPER_JOINT_IDX] = _smooth_1d(qpos[:, GRIPPER_JOINT_IDX])

    eef_out = eef.copy()
    if eef.shape[1] >= 3:
        eef_out = _smooth_cols(eef_out, range(3))
    if eef.shape[1] >= 6:
        eef_out = _smooth_cols(
            eef_out,
            range(3, 6),
            angular=True,
            pad_size=rpy_pad,
            kern=rpy_kernel,
        )
    if smooth_gripper and eef.shape[1] > GRIPPER_JOINT_IDX:
        eef_out[:, GRIPPER_JOINT_IDX] = qpos_out[:, GRIPPER_JOINT_IDX]
    elif eef.shape[1] > GRIPPER_JOINT_IDX:
        eef_out[:, GRIPPER_JOINT_IDX] = qpos[:, GRIPPER_JOINT_IDX]

    return eef_out, qpos_out


def prepare_replay_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    *,
    out_hz: float,
    smooth_window: int = 7,
    smooth_gripper: bool = False,
    control_mode: ReplayControlMode = "joint",
    eef_smooth_window: int = 15,
    eef_max_rpy_step_rad: float = 0.04,
    eef_ik_seed_smooth_window: int = 15,
    eef_quat_smooth_window: int = 7,
    orientation_interp: OrientationInterp = "slerp",
    singularity_adjust: bool = False,
    singularity_config: SingularityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    eef_i, qpos_i, period = interpolate_trajectory(
        eef, qpos, stamps, out_hz=out_hz, orientation_interp=orientation_interp
    )
    note_parts: list[str] = []
    if _uses_eef_trajectory(control_mode) and orientation_interp == "slerp":
        note_parts.append("orient=slerp")
    rpy_window = None
    if _uses_eef_trajectory(control_mode) and eef_smooth_window > 1:
        rpy_window = eef_smooth_window
    if smooth_window > 1 or rpy_window is not None:
        win = smooth_window if smooth_window > 1 else 7
        eef_i, qpos_i = smooth_trajectory(
            eef_i,
            qpos_i,
            win,
            smooth_gripper=smooth_gripper,
            rpy_window=rpy_window,
        )
        if smooth_window > 1:
            note_parts.append(f"smooth={smooth_window}")
        if rpy_window is not None:
            note_parts.append(f"eef_rpy_smooth={rpy_window}")
    if _uses_eef_trajectory(control_mode) and eef_quat_smooth_window > 1:
        eef_i = smooth_eef_orientation_quat(eef_i, eef_quat_smooth_window)
        note_parts.append(f"eef_quat_smooth={eef_quat_smooth_window}")
    if _uses_eef_trajectory(control_mode) and eef_ik_seed_smooth_window > 1:
        qpos_i = smooth_joints_only(qpos_i, eef_ik_seed_smooth_window)
        note_parts.append(f"ik_seed_smooth={eef_ik_seed_smooth_window}")
    if _uses_eef_trajectory(control_mode) and eef_max_rpy_step_rad > 0:
        eef_i = limit_eef_orientation_steps(eef_i, eef_max_rpy_step_rad)
        note_parts.append(f"rpy_step<={eef_max_rpy_step_rad:.3f}")
    if control_mode == "eef" and singularity_adjust:
        adjuster = SingularityAdjuster(singularity_config)
        eef_i, adj_stats = adjuster.adjust(eef_i, qpos_i)
        note_parts.append(f"singularity({adj_stats.summary()})")
        if adj_stats.warnings:
            for warn in adj_stats.warnings[:3]:
                print(f"  [singularity] warn: {warn}", flush=True)
    return eef_i, qpos_i, period, ", ".join(note_parts)


def interpolate_trajectory(
    eef: np.ndarray,
    qpos: np.ndarray,
    stamps: np.ndarray,
    out_hz: float,
    *,
    orientation_interp: OrientationInterp = "euler",
) -> tuple[np.ndarray, np.ndarray, float]:
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(stamps) < 2:
        return eef, qpos, 1.0 / max(out_hz, 1.0)

    t_rel = stamps - stamps[0]
    duration = max(float(t_rel[-1]), 1.0 / max(out_hz, 1.0))
    n_out = max(2, int(duration * out_hz) + 1)
    t_out = np.linspace(0.0, duration, n_out)

    eef_out = np.zeros((n_out, eef.shape[1]), dtype=np.float64)
    qpos_out = np.zeros((n_out, qpos.shape[1]), dtype=np.float64)

    for d in range(min(ARM_JOINT_DIMS, qpos.shape[1])):
        qpos_out[:, d] = _interp_angle_series(t_out, t_rel, qpos[:, d])
    if qpos.shape[1] > GRIPPER_JOINT_IDX:
        qpos_out[:, GRIPPER_JOINT_IDX] = _interp_series(t_out, t_rel, qpos[:, GRIPPER_JOINT_IDX])

    for d in range(min(3, eef.shape[1])):
        eef_out[:, d] = _interp_series(t_out, t_rel, eef[:, d])
    if eef.shape[1] >= 6:
        if orientation_interp == "slerp":
            eef_out[:, 3:6] = _interp_rpy_slerp(t_out, t_rel, eef[:, 3:6])
        else:
            for d in range(3, 6):
                eef_out[:, d] = _interp_angle_series(t_out, t_rel, eef[:, d])
    if eef.shape[1] > GRIPPER_JOINT_IDX:
        eef_out[:, GRIPPER_JOINT_IDX] = qpos_out[:, GRIPPER_JOINT_IDX]

    return eef_out, qpos_out, 1.0 / max(out_hz, 1.0)


def resolve_ik_seed(
    bridge: X5ArmBridge,
    qpos_hint: np.ndarray,
    *,
    closed_loop: bool = True,
    hint_blend: float = 0.0,
) -> np.ndarray:
    """Pick IK seed for mode=4: prefer live joint feedback for branch continuity."""
    hint = np.asarray(qpos_hint[:ARM_JOINT_DIMS], dtype=np.float64)
    if not closed_loop:
        return hint
    state = bridge.latest_state
    if state is None:
        return hint
    actual = np.asarray(state.joint_pos[:ARM_JOINT_DIMS], dtype=np.float64)
    blend = float(np.clip(hint_blend, 0.0, 1.0))
    if blend <= 0.0:
        return actual
    out = actual.copy()
    for j in range(ARM_JOINT_DIMS):
        delta = np.arctan2(np.sin(hint[j] - actual[j]), np.cos(hint[j] - actual[j]))
        out[j] = actual[j] + blend * delta
    return out


def publish_replay_frame(
    bridge: X5ArmBridge,
    eef_row: np.ndarray,
    qpos_row: np.ndarray,
    *,
    control_mode: ReplayControlMode = "joint",
    ik_seed_closed_loop: bool = True,
    ik_seed_hint_blend: float = 0.0,
    eef_via_joint_streamer: Optional[EefViaJointStreamer] = None,
) -> None:
    gripper = float(
        eef_row[GRIPPER_JOINT_IDX]
        if eef_row.shape[0] > GRIPPER_JOINT_IDX
        else qpos_row[GRIPPER_JOINT_IDX]
    )
    if control_mode == "eef":
        ik_seed = resolve_ik_seed(
            bridge,
            qpos_row,
            closed_loop=ik_seed_closed_loop,
            hint_blend=ik_seed_hint_blend,
        )
        bridge.publish_command(
            ArmCommand(
                xyzrpy=eef_row[:6].tolist(),
                gripper=gripper,
                mode=MODE_END_CONTROL,
                joint_pos=ik_seed.tolist(),
            )
        )
        return

    if control_mode == "eef_via_joint":
        if eef_via_joint_streamer is None:
            raise ValueError("eef_via_joint requires eef_via_joint_streamer")
        q6 = eef_via_joint_streamer.step(eef_row, qpos_row)
        bridge.publish_joint_command(q6.tolist(), gripper, mode=MODE_POSITION_CONTROL)
        return

    bridge.publish_command(
        ArmCommand(
            xyzrpy=eef_row[:6].tolist(),
            gripper=gripper,
            joint_pos=qpos_row[:ARM_JOINT_DIMS].tolist(),
            mode=MODE_POSITION_CONTROL,
        )
    )


def _hold_pose(
    bridge: X5ArmBridge,
    eef_row: np.ndarray,
    qpos_row: np.ndarray,
    period: float,
    hold_sec: float,
    *,
    control_mode: ReplayControlMode = "joint",
    ik_seed_closed_loop: bool = True,
    ik_seed_hint_blend: float = 0.0,
    eef_via_joint_streamer: Optional[EefViaJointStreamer] = None,
) -> None:
    if hold_sec <= 0.0:
        return
    # eef_via_joint: keep streaming same EEF (Δ≈0) so we hold converted joints, not recorded qpos.
    if control_mode == "eef_via_joint":
        hold_mode: ReplayControlMode = "eef_via_joint"
        streamer = eef_via_joint_streamer
    else:
        hold_mode = _blend_control_mode(control_mode)
        streamer = None
    n_hold = max(1, int(hold_sec / period))
    for _ in range(n_hold):
        t0 = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)
        publish_replay_frame(
            bridge,
            eef_row,
            qpos_row,
            control_mode=hold_mode,
            ik_seed_closed_loop=ik_seed_closed_loop,
            ik_seed_hint_blend=ik_seed_hint_blend,
            eef_via_joint_streamer=streamer,
        )
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)


def _blend_to_start(
    bridge: X5ArmBridge,
    eef_i: np.ndarray,
    qpos_i: np.ndarray,
    period: float,
    *,
    control_mode: ReplayControlMode = "joint",
    blend_sec: float = 0.5,
    max_blend_sec: float = 6.0,
    blend_skip_tol: float = 0.05,
    blend_speed_rad_s: float = 0.8,
    settle_sec: float = 0.8,
    snap_sec: float = 2.5,
) -> float:
    """Snap to exact frame-0 joints before replay (never skip)."""
    del eef_i, control_mode, blend_sec, blend_skip_tol, settle_sec
    return snap_to_trajectory_home(
        bridge,
        qpos_i[0],
        period,
        label="pre-replay snap",
        blend_speed_rad_s=blend_speed_rad_s,
        max_blend_sec=max_blend_sec,
        lock_sec=max(2.5, snap_sec),
    )


def _blend_from_end(
    bridge: X5ArmBridge,
    eef_end: np.ndarray,
    qpos_end: np.ndarray,
    target_eef: np.ndarray,
    target_qpos: np.ndarray,
    period: float,
    *,
    control_mode: ReplayControlMode = "joint",
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    blend_speed_rad_s: float = 0.8,
) -> float:
    del eef_end, qpos_end, target_eef, control_mode, blend_sec
    return snap_to_trajectory_home(
        bridge,
        target_qpos,
        period,
        label="post-blend",
        blend_speed_rad_s=blend_speed_rad_s,
        max_blend_sec=max_blend_sec,
        lock_sec=2.0,
    )


def _run_replay_loop(
    bridge: X5ArmBridge,
    eef_i: np.ndarray,
    qpos_i: np.ndarray,
    period: float,
    replay_hz: float,
    label: str,
    *,
    control_mode: ReplayControlMode = "joint",
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    blend_skip_tol: float = 0.05,
    blend_speed_rad_s: float = 0.8,
    post_hold_sec: float = 1.0,
    smooth_note: str = "",
    return_to_start: bool = False,
    return_blend_sec: float = 1.5,
    return_max_blend_sec: float = 6.0,
    return_blend_speed_rad_s: float = 0.8,
    return_settle_sec: float = 1.5,
    return_hold_sec: float = 1.0,
    return_tol_rad: float = 0.04,
    start_max_blend_sec: float = 6.0,
    start_settle_sec: float = 0.8,
    start_snap_sec: float = 1.0,
    eef_joint_lead_sec: float = 0.0,
    eef_joint_fallback_enabled: bool = False,
    eef_joint_fallback_step_rad: float = 0.015,
    eef_joint_fallback_hold_sec: float = 0.8,
    eef_joint_fallback_lookahead_sec: float = 0.2,
    eef_ik_seed_closed_loop: bool = True,
    eef_ik_seed_hint_blend: float = 0.0,
    singularity_joint_fallback: bool = False,
    singularity_config: SingularityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_play = len(qpos_i)
    print(f"{label}: {n_play} steps @ {replay_hz:.0f} Hz. Ctrl+C to abort.", flush=True)
    if smooth_note:
        print(f"  preprocess: {smooth_note}", flush=True)
    period_scales = np.ones(n_play, dtype=np.float64)
    eef_via_joint_streamer: Optional[EefViaJointStreamer] = None
    if control_mode == "eef":
        print(
            "  control: mode=4 END_CONTROL, "
            f"{'closed-loop' if eef_ik_seed_closed_loop else 'recorded'} IK seed + mode4_guard",
            flush=True,
        )
        cfg = singularity_config or SingularityConfig()
        if cfg.slowdown_enabled and cfg.slowdown_factor > 1.0:
            period_scales = SingularityAdjuster(cfg).period_scale(qpos_i)
            n_slow = int(np.sum(period_scales > 1.0 + 1e-9))
            if n_slow > 0:
                print(
                    f"  singularity slowdown: {n_slow}/{n_play} frames ×"
                    f"{cfg.slowdown_factor:.1f} (EEF path unchanged)",
                    flush=True,
                )
        if eef_joint_lead_sec > 0.0:
            print(
                f"  note: first {eef_joint_lead_sec:.2f}s uses joint mode=5 (eef joint lead-in)",
                flush=True,
            )
        if eef_joint_fallback_enabled:
            print(
                f"  note: fast motion > {np.degrees(eef_joint_fallback_step_rad):.2f}°/step "
                f"falls back to joint mode=5 (hold {eef_joint_fallback_hold_sec:.1f}s, "
                f"lookahead {eef_joint_fallback_lookahead_sec:.1f}s)",
                flush=True,
            )
        if singularity_joint_fallback:
            print(
                "  warn: singularity_joint_fallback is ignored "
                "(offline joint-mask path removed; use recorded EEF as-is)",
                flush=True,
            )
    elif control_mode == "eef_via_joint":
        print(
            "  control: EEF→diff IK→joints mode=5 "
            "(client differential IK; no SDK mode=4)",
            flush=True,
        )
    _blend_to_start(
        bridge,
        eef_i,
        qpos_i,
        period,
        control_mode=control_mode,
        blend_sec=blend_sec,
        max_blend_sec=max(start_max_blend_sec, max_blend_sec),
        blend_skip_tol=blend_skip_tol,
        blend_speed_rad_s=blend_speed_rad_s,
        settle_sec=start_settle_sec,
        snap_sec=start_snap_sec,
    )
    if control_mode == "eef_via_joint":
        eef_via_joint_streamer = EefViaJointStreamer()
        print(
            f"  eef_via_joint: ik_max_dq={eef_via_joint_streamer.ik_max_dq} "
            f"max_joint_delta={eef_via_joint_streamer.max_joint_delta_rad} "
            f"ema={eef_via_joint_streamer.joint_ema} damp={eef_via_joint_streamer.ik_damp} "
            f"yaml={eef_via_joint_streamer._yaml_path or '(fallback)'}",
            flush=True,
        )
        seed_q = qpos_i[0][:ARM_JOINT_DIMS]
        cur_eef, cur_q = _read_arm_pose(bridge)
        if cur_q is not None:
            seed_q = cur_q[:ARM_JOINT_DIMS]
        eef_via_joint_streamer.reset(seed_q)
    lead_steps = 0
    frame_modes: list[ReplayControlMode] | None = None
    if control_mode == "eef":
        if eef_joint_lead_sec > 0.0:
            lead_steps = min(n_play, max(1, int(eef_joint_lead_sec / period)))
        frame_modes, mode_stats = plan_eef_replay_frame_modes(
            eef_i,
            qpos_i,
            period,
            lead_steps=lead_steps,
            fallback_enabled=eef_joint_fallback_enabled,
            step_thresh_rad=eef_joint_fallback_step_rad,
            hold_sec=eef_joint_fallback_hold_sec,
            lookahead_sec=eef_joint_fallback_lookahead_sec,
        )
        if mode_stats["joint_frames"] > 0:
            pct = 100.0 * mode_stats["joint_frames"] / max(mode_stats["total"], 1)
            print(
                f"  eef fallback plan: {mode_stats['joint_frames']}/{mode_stats['total']} "
                f"frames joint ({pct:.1f}%; lead={mode_stats['lead_steps']}, "
                f"fast-motion≈{mode_stats.get('fallback_frames', 0)})",
                flush=True,
            )
    try:
        for i in range(n_play):
            t0 = time.monotonic()
            rclpy.spin_once(bridge, timeout_sec=0.0)
            if frame_modes is not None:
                frame_mode = frame_modes[i]
            elif control_mode == "eef" and i < lead_steps:
                frame_mode = "joint"
            else:
                frame_mode = control_mode
            publish_replay_frame(
                bridge,
                eef_i[i],
                qpos_i[i],
                control_mode=frame_mode,
                ik_seed_closed_loop=eef_ik_seed_closed_loop,
                ik_seed_hint_blend=eef_ik_seed_hint_blend,
                eef_via_joint_streamer=eef_via_joint_streamer,
            )
            if i % max(1, int(replay_hz)) == 0:
                print(f"  {label} {i}/{n_play}", flush=True)
            frame_period = float(period) * float(period_scales[i])
            elapsed = time.monotonic() - t0
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)
    except KeyboardInterrupt:
        print("Replay aborted.", flush=True)
        raise
    last_eef = eef_i[-1].copy()
    last_qpos = qpos_i[-1].copy()
    if (
        control_mode == "eef_via_joint"
        and eef_via_joint_streamer is not None
        and eef_via_joint_streamer.last_q is not None
    ):
        last_qpos = last_qpos.copy()
        last_qpos[:ARM_JOINT_DIMS] = eef_via_joint_streamer.last_q
    if post_hold_sec > 0.0:
        print(f"  hold end pose {post_hold_sec:.1f}s …", flush=True)
        _hold_pose(
            bridge,
            last_eef,
            last_qpos,
            period,
            post_hold_sec,
            control_mode=control_mode,
            ik_seed_closed_loop=eef_ik_seed_closed_loop,
            ik_seed_hint_blend=eef_ik_seed_hint_blend,
            eef_via_joint_streamer=eef_via_joint_streamer,
        )
    if return_to_start and len(qpos_i) > 0:
        start_q = qpos_i[0].copy()
        start_e = eef_i[0].copy()
        _return_to_trajectory_start(
            bridge,
            start_e,
            start_q,
            period,
            blend_sec=return_blend_sec,
            max_blend_sec=return_max_blend_sec,
            blend_speed_rad_s=return_blend_speed_rad_s,
            settle_sec=return_settle_sec,
            hold_sec=return_hold_sec,
            tol_rad=return_tol_rad,
        )
        last_eef = start_e.copy()
        last_qpos = start_q.copy()
    print("Replay finished.", flush=True)
    return last_eef, last_qpos


def finish_replay_handoff(
    bridge: X5ArmBridge,
    mapper: Any,
    servo: Any,
    last_eef: np.ndarray,
    last_qpos: np.ndarray,
    period: float,
    *,
    control_mode: ReplayControlMode = "joint",
    sync_uarm: bool = True,
    blend_to_uarm: bool = False,
) -> None:
    """Re-sync U-Arm mapping after replay; optionally blend arm toward U-Arm."""
    rclpy.spin_once(bridge, timeout_sec=0.0)
    state = bridge.latest_state
    if state is None:
        return

    master_deg = servo.get_angles_deg()
    if sync_uarm:
        offset = mapper.align_to_follower(state.joint_pos, master_deg)
        print(
            f">> 回放结束：U-Arm 已同步到从臂当前姿态 (offset max {offset:.3f} rad)",
            flush=True,
        )
        print(">> 请保持 U-Arm 不动，再缓慢继续遥操作。", flush=True)
        if not blend_to_uarm:
            return
        cmd = mapper.map_master_deg(master_deg)
        target_qpos = mapper.to_qpos7(cmd)
        target_eef = np.concatenate([state.end_pos[:6], target_qpos[6:7]])
        _blend_from_end(
            bridge,
            last_eef,
            last_qpos,
            target_eef,
            target_qpos,
            period,
            control_mode=control_mode,
            blend_sec=0.4,
            max_blend_sec=1.5,
        )


def load_episode_arrays(
    hdf5_path: Path,
    joint_source: Literal["action", "state"] = "action",
    *,
    control_mode: ReplayControlMode = "joint",
    eef_source: EefSource = "auto",
    eef_ik_seed: IkSeedSource = "action",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = read_hdf5_meta(hdf5_path)
    action_space = meta["action_space"]
    if eef_source == "auto":
        if action_space == "eef":
            resolved_eef_source = "action"
        else:
            resolved_eef_source = "observations"
    else:
        resolved_eef_source = eef_source

    ep = load_hdf5_episode(hdf5_path, load_eef_fk=True)
    action = np.asarray(ep["action"], dtype=np.float64)
    state = np.asarray(ep["qpos"], dtype=np.float64)
    if _uses_eef_trajectory(control_mode):
        if resolved_eef_source == "action":
            eef = action.copy()
        elif resolved_eef_source == "eef_fk":
            if ep.get("eef_fk") is None:
                raise ValueError(
                    f"{hdf5_path}: eef_fk missing. Run: "
                    f"python3 scripts/bake_eef_fk.py {hdf5_path}"
                )
            eef = np.asarray(ep["eef_fk"], dtype=np.float64)
        else:
            eef = np.asarray(ep["eef"], dtype=np.float64)
        # Seed / blend joints: prefer recorded action joints when available.
        qpos = action if eef_ik_seed == "action" else state
    else:
        qpos = action if joint_source == "action" else state
        eef = np.asarray(ep["eef"], dtype=np.float64)

    if eef.shape[0] != qpos.shape[0]:
        raise ValueError(f"{hdf5_path}: eef/qpos length mismatch")
    return eef, qpos, ep["stamps"]


def replay_hdf5_file(
    bridge: X5ArmBridge,
    hdf5_path: Path,
    replay_hz: float = 50.0,
    joint_source: Literal["action", "state"] = "action",
    control_mode: ReplayControlMode = "joint",
    eef_source: EefSource = "auto",
    *,
    smooth_window: int = 7,
    smooth_gripper: bool = False,
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    blend_skip_tol: float = 0.05,
    blend_speed_rad_s: float = 0.8,
    post_hold_sec: float = 1.0,
    eef_smooth_window: int = 15,
    eef_max_rpy_step_rad: float = 0.04,
    eef_ik_seed: IkSeedSource = "action",
    eef_ik_seed_smooth_window: int = 15,
    eef_quat_smooth_window: int = 7,
    orientation_interp: OrientationInterp = "slerp",
    return_to_start: bool = False,
    return_blend_sec: float = 1.5,
    return_max_blend_sec: float = 6.0,
    return_settle_sec: float = 1.5,
    return_hold_sec: float = 1.0,
    return_tol_rad: float = 0.04,
    start_max_blend_sec: float = 6.0,
    start_settle_sec: float = 0.8,
    start_snap_sec: float = 1.0,
    eef_joint_lead_sec: float = 0.0,
    eef_joint_fallback_enabled: bool = False,
    eef_joint_fallback_step_rad: float = 0.015,
    eef_joint_fallback_hold_sec: float = 0.8,
    eef_joint_fallback_lookahead_sec: float = 0.2,
    singularity_adjust: bool = False,
    eef_ik_seed_closed_loop: bool = True,
    eef_ik_seed_hint_blend: float = 0.0,
    singularity_joint_fallback: bool = False,
    singularity_config: SingularityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    meta = read_hdf5_meta(hdf5_path)
    eef, qpos, stamps = load_episode_arrays(
        hdf5_path,
        joint_source=joint_source,
        control_mode=control_mode,
        eef_source=eef_source,
        eef_ik_seed=eef_ik_seed,
    )
    eef_i, qpos_i, period, smooth_note = prepare_replay_trajectory(
        eef,
        qpos,
        stamps,
        out_hz=replay_hz,
        smooth_window=smooth_window,
        smooth_gripper=smooth_gripper,
        control_mode=control_mode,
        eef_smooth_window=eef_smooth_window,
        eef_max_rpy_step_rad=eef_max_rpy_step_rad,
        eef_ik_seed_smooth_window=eef_ik_seed_smooth_window,
        eef_quat_smooth_window=eef_quat_smooth_window,
        orientation_interp=orientation_interp,
        singularity_adjust=singularity_adjust,
        singularity_config=singularity_config,
    )
    if _uses_eef_trajectory(control_mode):
        if eef_source == "auto":
            src = (
                "action"
                if meta["action_space"] == "eef"
                else "observations/eef"
            )
        elif eef_source == "eef_fk":
            src = "observations/eef_fk"
        elif eef_source == "observations":
            src = "observations/eef"
        else:
            src = "action"
        if control_mode == "eef_via_joint":
            mode_label = f"eef→joint diff-IK ({src}, seed={eef_ik_seed})"
        else:
            mode_label = f"eef+IK ({src}, seed={eef_ik_seed})"
    else:
        mode_label = f"joints={joint_source}"
    return _run_replay_loop(
        bridge,
        eef_i,
        qpos_i,
        period,
        replay_hz,
        f"HDF5 replay ({hdf5_path.name}, {mode_label})",
        control_mode=control_mode,
        blend_sec=blend_sec,
        max_blend_sec=max_blend_sec,
        blend_skip_tol=blend_skip_tol,
        blend_speed_rad_s=blend_speed_rad_s,
        post_hold_sec=post_hold_sec,
        smooth_note=smooth_note,
        return_to_start=return_to_start,
        return_blend_sec=return_blend_sec,
        return_max_blend_sec=return_max_blend_sec,
        return_blend_speed_rad_s=blend_speed_rad_s,
        return_settle_sec=return_settle_sec,
        return_hold_sec=return_hold_sec,
        return_tol_rad=return_tol_rad,
        start_max_blend_sec=start_max_blend_sec,
        start_settle_sec=start_settle_sec,
        start_snap_sec=start_snap_sec,
        eef_joint_lead_sec=eef_joint_lead_sec,
        eef_joint_fallback_enabled=eef_joint_fallback_enabled,
        eef_joint_fallback_step_rad=eef_joint_fallback_step_rad,
        eef_joint_fallback_hold_sec=eef_joint_fallback_hold_sec,
        eef_joint_fallback_lookahead_sec=eef_joint_fallback_lookahead_sec,
        eef_ik_seed_closed_loop=eef_ik_seed_closed_loop,
        eef_ik_seed_hint_blend=eef_ik_seed_hint_blend,
        singularity_joint_fallback=singularity_joint_fallback,
        singularity_config=singularity_config,
    )


def replay_lerobot_from_hdf5(
    bridge: X5ArmBridge,
    hdf5_path: Path,
    replay_hz: float = 50.0,
    record_fps: float = 20.0,
    control_mode: ReplayControlMode = "joint",
    *,
    smooth_window: int = 7,
    smooth_gripper: bool = False,
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    blend_skip_tol: float = 0.05,
    blend_speed_rad_s: float = 0.8,
    post_hold_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = hdf5_path.parent / f".replay_{hdf5_path.stem}.parquet"
    print(f"  [1/3] 读取 HDF5: {hdf5_path.name}", flush=True)
    print(f"  [2/3] 转换 parquet（大文件约 30 秒，请勿重复按键）…", flush=True)
    t0 = time.monotonic()
    hdf5_to_parquet(hdf5_path, parquet_path, fps=record_fps)
    print(f"  [2/3] parquet 完成 ({time.monotonic() - t0:.1f}s)", flush=True)
    print(f"        临时文件: {parquet_path}", flush=True)
    print(
        "        正式 LeRobot 训练集（mp4+parquet）需 run_convert_to_lerobot.sh 导出到 "
        "data/arx_x5_real/runs/<日期>/",
        flush=True,
    )
    diffs = compare_hdf5_parquet(hdf5_path, parquet_path)
    print(
        "  LeRobot convert check: "
        f"max|hdf5.action - parquet.action| = {diffs['action_max_abs_diff']:.2e}, "
        f"max|hdf5.qpos - parquet.state| = {diffs['state_max_abs_diff']:.2e}",
        flush=True,
    )

    data = load_parquet_episode(parquet_path)
    action = data["action"]
    ep = load_hdf5_episode(hdf5_path)
    eef = ep["eef"].astype(np.float64)
    stamps = ep["stamps"]

    print(f"  [3/3] 开始机械臂回放 @ {replay_hz:.0f} Hz …", flush=True)
    eef_i, qpos_i, period, smooth_note = prepare_replay_trajectory(
        eef,
        action,
        stamps,
        out_hz=replay_hz,
        smooth_window=smooth_window,
        smooth_gripper=smooth_gripper,
    )
    mode_label = "eef+IK" if control_mode == "eef" else "parquet action joints"
    return _run_replay_loop(
        bridge,
        eef_i,
        qpos_i,
        period,
        replay_hz,
        f"LeRobot replay ({parquet_path.name}, {mode_label})",
        control_mode=control_mode,
        blend_sec=blend_sec,
        max_blend_sec=max_blend_sec,
        blend_skip_tol=blend_skip_tol,
        blend_speed_rad_s=blend_speed_rad_s,
        post_hold_sec=post_hold_sec,
        smooth_note=smooth_note,
    )


def needs_eef_fk_bake(hdf5_path: Path, eef_source: EefSource = "auto") -> bool:
    """True when eef replay expects eef_fk but HDF5 has not been baked yet."""
    if eef_source in {"observations", "action", "auto"}:
        return False
    if eef_source == "eef_fk":
        return not hdf5_has_eef_fk(hdf5_path)
    return False


def write_eef_fk_to_hdf5(
    hdf5_path: Path,
    eef_fk: np.ndarray,
    *,
    output: Path | None = None,
    quat_smooth_window: int = 0,
) -> Path:
    import shutil

    import h5py

    target = (output or hdf5_path).resolve()
    if target != hdf5_path.resolve():
        shutil.copy2(hdf5_path, target)

    with h5py.File(target, "a") as f:
        obs = f.require_group("observations")
        if "eef_fk" in obs:
            del obs["eef_fk"]
        obs.create_dataset(
            "eef_fk",
            data=np.asarray(eef_fk, dtype=np.float32),
            compression="gzip",
            compression_opts=1,
        )
        f.attrs["eef_fk_baked"] = True
        f.attrs["eef_fk_bake_quat_smooth"] = int(quat_smooth_window)
    return target


def bake_eef_fk(
    bridge: X5ArmBridge,
    hdf5_path: Path,
    *,
    replay_hz: float = 100.0,
    smooth_window: int = 15,
    quat_smooth_window: int = 7,
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    write_back: bool = True,
    return_to_start: bool = True,
    return_blend_sec: float = 1.5,
    return_max_blend_sec: float = 6.0,
    return_blend_speed_rad_s: float = 0.8,
    return_settle_sec: float = 1.5,
    return_hold_sec: float = 2.5,
    return_tol_rad: float = 0.04,
) -> np.ndarray:
    """Joint-replay action on the arm, record FK end_pos → observations/eef_fk."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        action = np.asarray(f["action"], dtype=np.float64)
        eef_obs = np.asarray(f["observations/eef"], dtype=np.float64)
        stamps = (
            np.asarray(f["arm_stamp"], dtype=np.float64)
            if "arm_stamp" in f
            else np.asarray(f["time"], dtype=np.float64)
        )

    eef_dummy = eef_obs.copy()
    _, qpos_i, period, note = prepare_replay_trajectory(
        eef_dummy,
        action,
        stamps,
        out_hz=replay_hz,
        smooth_window=smooth_window,
        control_mode="joint",
    )
    print(
        f"  [bake] 关节回放录 FK: {len(qpos_i)} 步 @ {replay_hz:.0f} Hz ({note})",
        flush=True,
    )
    print(
        "  [bake] 用 mode=5 走 action，记录控制器 FK 末端 → observations/eef_fk",
        flush=True,
    )

    eef_fk_bake: list[np.ndarray] = []
    _blend_to_start(
        bridge,
        eef_dummy,
        qpos_i,
        period,
        control_mode="joint",
        blend_sec=blend_sec,
        max_blend_sec=max_blend_sec,
    )
    for i in range(len(qpos_i)):
        t0 = time.monotonic()
        rclpy.spin_once(bridge, timeout_sec=0.0)
        publish_replay_frame(
            bridge,
            eef_dummy[min(i, len(eef_dummy) - 1)],
            qpos_i[i],
            control_mode="joint",
        )
        state = bridge.latest_state
        if state is not None:
            row = np.concatenate(
                [np.asarray(state.end_pos[:6], dtype=np.float64), qpos_i[i, 6:7]]
            )
            eef_fk_bake.append(row)
        if i % max(1, int(replay_hz)) == 0:
            print(f"  [bake] {i}/{len(qpos_i)}", flush=True)
        elapsed = time.monotonic() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    if return_to_start and len(qpos_i) > 0 and eef_fk_bake:
        start_e = np.asarray(eef_fk_bake[0], dtype=np.float64)
        start_q = qpos_i[0].copy()
        _return_to_trajectory_start(
            bridge,
            start_e,
            start_q,
            period,
            blend_sec=return_blend_sec,
            max_blend_sec=return_max_blend_sec,
            blend_speed_rad_s=return_blend_speed_rad_s,
            settle_sec=return_settle_sec,
            hold_sec=return_hold_sec,
            tol_rad=return_tol_rad,
            label="[bake] return-to-start",
        )

    baked = np.stack(eef_fk_bake, axis=0) if eef_fk_bake else np.zeros((0, 7))
    t_rel = stamps - stamps[0]
    duration = max(float(t_rel[-1]), 1e-6)
    t_bake = np.linspace(0.0, duration, len(baked))
    eef_out = np.zeros_like(eef_obs)
    for i, t in enumerate(t_rel):
        j = int(np.searchsorted(t_bake, t))
        j = min(max(j, 0), len(baked) - 1)
        eef_out[i] = baked[j]

    if quat_smooth_window > 1:
        before = float(np.max(np.abs(np.diff(np.degrees(eef_out[:, 3:6]), axis=0))))
        eef_out = smooth_eef_orientation_quat(eef_out, quat_smooth_window)
        after = float(np.max(np.abs(np.diff(np.degrees(eef_out[:, 3:6]), axis=0))))
        print(
            f"  [bake] quat smooth w={quat_smooth_window}: max|Δrpy| {before:.2f}° → {after:.2f}°/帧",
            flush=True,
        )

    obs_diff = float(np.max(np.abs(eef_out[:, :6] - eef_obs[:, :6])))
    rpy_diff = float(np.max(np.abs(eef_out[:, 3:6] - eef_obs[:, 3:6])))
    print(
        f"  [bake] 完成 max|eef_fk−eef_obs| xyz={obs_diff:.4f} m, "
        f"rpy={np.degrees(rpy_diff):.2f}°",
        flush=True,
    )

    if write_back:
        write_eef_fk_to_hdf5(
            hdf5_path,
            eef_out.astype(np.float32),
            quat_smooth_window=quat_smooth_window,
        )
        print(f"  [bake] 已写入 {hdf5_path} observations/eef_fk", flush=True)
    return eef_out.astype(np.float32)


def ensure_eef_fk_baked(
    bridge: X5ArmBridge,
    hdf5_path: Path,
    *,
    auto_bake: bool = False,
    eef_source: EefSource = "auto",
    replay_hz: float = 100.0,
    smooth_window: int = 15,
    quat_smooth_window: int = 7,
    blend_sec: float = 0.5,
    max_blend_sec: float = 2.0,
    force_rebake: bool = False,
    return_to_start: bool = True,
    return_blend_sec: float = 1.5,
    return_max_blend_sec: float = 6.0,
    return_blend_speed_rad_s: float = 0.8,
    return_settle_sec: float = 1.5,
    return_hold_sec: float = 2.5,
    return_tol_rad: float = 0.04,
) -> bool:
    """Bake eef_fk when needed. Returns True if bake ran."""
    if not auto_bake:
        return False
    if eef_source == "observations":
        return False
    if force_rebake and hdf5_has_eef_fk(hdf5_path):
        print("  [bake] force_rebake_eef_fk=true，将覆盖已有 observations/eef_fk", flush=True)
    elif not needs_eef_fk_bake(hdf5_path, eef_source):
        if hdf5_has_eef_fk(hdf5_path):
            print("  [bake] 已有 observations/eef_fk，跳过", flush=True)
        return False
    print(
        "\n>> [bake] 末端回放前自动生成 eef_fk（约 1 分钟，机械臂先走一遍关节轨迹）…",
        flush=True,
    )
    bake_eef_fk(
        bridge,
        hdf5_path,
        replay_hz=replay_hz,
        smooth_window=smooth_window,
        quat_smooth_window=quat_smooth_window,
        blend_sec=blend_sec,
        max_blend_sec=max_blend_sec,
        write_back=True,
        return_to_start=return_to_start,
        return_blend_sec=return_blend_sec,
        return_max_blend_sec=return_max_blend_sec,
        return_blend_speed_rad_s=return_blend_speed_rad_s,
        return_settle_sec=return_settle_sec,
        return_hold_sec=return_hold_sec,
        return_tol_rad=return_tol_rad,
    )
    return True
