#!/usr/bin/env python3
"""Run a LIBERO-trained StarVLA policy on ARX X5 (single arm, RealSense cameras).

Architecture:
  Server (GPU):  starVLA policy server  (WebSocket)
  Laptop:        this script            (cameras + ROS2 /arm_cmd)

The LIBERO checkpoint expects two images [agentview, wrist].
  - Default: duplicate D405 wrist into both slots (debug).
  - --desktop-camera: D435i desktop (cam_high) + D405 wrist (cam_wrist).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rclpy

try:
    import cv2  # for rollout video recording (--record-dir)
except Exception:  # pragma: no cover - recording optional
    cv2 = None

DEFAULT_STARVLA_ROOT = "/9950backfile/zhangjh/wh/starvla_wh/starVLA"
MODE_END_CONTROL = 4
MODE_POSITION_CONTROL = 5
ARM_JOINT_DIMS = 6
GRIPPER_JOINT_IDX = 6
# Same default as teleop_uarm.example.json startup.go_home_position_deg
DEFAULT_GO_HOME_DEG = (90.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_GO_HOME_GRIPPER = 4.5

# Optional 100Hz interpolation publish thread (interp_publish.py).
# Only activated when --publish-hz > 0; otherwise original direct publish is used.
try:
    from interp_publish import InterpTarget, InterpPublishThread
except ImportError:
    InterpTarget = None  # type: ignore
    InterpPublishThread = None  # type: ignore

# Optional MuJoCo play-viz bus (see lewm_x5_grasp/scripts/tool/).
_PLAY_VIZ_IMPORT_WARNED = False
_EEF_OFFSET_WARNED = False


def _current_xyz_from_sdk(sdk_end_pos_xyz: np.ndarray) -> np.ndarray:
    """SDK flange xyz + YAML eef_xyz_offset_m → current_xyz (offset frame).

    Canonical params: .../lewm_x5_grasp/scripts/x5_play_params.yaml
    (or env X5_PLAY_PARAMS). Prefer x5_play_viz.eef_offset when on PYTHONPATH.
    """
    global _EEF_OFFSET_WARNED
    try:
        # Shared loader with MuJoCo viz (same YAML).
        from x5_play_viz.eef_offset import current_xyz_with_offset

        return current_xyz_with_offset(sdk_end_pos_xyz)
    except Exception as exc:
        if not _EEF_OFFSET_WARNED:
            print(
                "[eef-offset] x5_play_viz.eef_offset unavailable (%s); "
                "trying scripts/x5_play_params.yaml directly"
                % exc,
                flush=True,
            )
            _EEF_OFFSET_WARNED = True
    # Fallback: read the same YAML without importing the package.
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    candidates = [
        os.environ.get("X5_PLAY_PARAMS", "").strip(),
        str(
            Path.home()
            / "project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/scripts/x5_play_params.yaml"
        ),
        str(Path(__file__).resolve().parent / "x5_play_params.yaml"),
    ]
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if yaml is not None:
            data = yaml.safe_load(text)
            off = np.asarray(data["eef_xyz_offset_m"], dtype=np.float64).reshape(-1)[:3]
        else:
            import ast
            import re

            m = re.search(r"eef_xyz_offset_m\s*:\s*(\[[^\]]+\])", text)
            if not m:
                continue
            off = np.asarray(ast.literal_eval(m.group(1)), dtype=np.float64).reshape(-1)[:3]
        return np.asarray(sdk_end_pos_xyz, dtype=np.float64).reshape(-1)[:3] + off
    # Last resort: zero offset (should not happen if YAML is present).
    if not _EEF_OFFSET_WARNED:
        print("[eef-offset] WARNING: x5_play_params.yaml not found; offset=0", flush=True)
        _EEF_OFFSET_WARNED = True
    return np.asarray(sdk_end_pos_xyz, dtype=np.float64).reshape(-1)[:3]


def _eef_with_xyz_offset(sdk_eef6: np.ndarray) -> np.ndarray:
    """Full xyzrpy: xyz := SDK xyz + YAML offset; rpy unchanged.

    Used as the eef_via_joint integration base (current) so
    target = current_offset + Δ, matching logged current_xyz / target_xyz.
    """
    out = np.asarray(sdk_eef6, dtype=np.float64).reshape(-1)[:6].copy()
    out[:3] = _current_xyz_from_sdk(out[:3])
    return out


def _apply_dxyz_xy_remap(dxyz: np.ndarray) -> np.ndarray:
    """Δxyz remap from YAML: (x,y,z)->(-y,x,z). Used for IK + clipped_dxyz logs."""
    try:
        from x5_play_viz.eef_offset import apply_dxyz_xy_remap

        return apply_dxyz_xy_remap(dxyz)
    except Exception:
        # Fallback: read same YAML flag / apply fixed map when enabled.
        enabled = True
        try:
            import yaml  # type: ignore

            p = Path.home() / (
                "project/ARX_X5_clean_server46(7.12get-data)/"
                "lewm_x5_grasp/scripts/x5_play_params.yaml"
            )
            env = os.environ.get("X5_PLAY_PARAMS", "").strip()
            if env:
                p = Path(env).expanduser()
            if p.is_file():
                data = yaml.safe_load(p.read_text(encoding="utf-8"))
                enabled = bool(data.get("dxyz_xy_remap_enabled", True))
        except Exception:
            enabled = True
        v = np.asarray(dxyz, dtype=np.float64).reshape(-1)[:3].copy()
        if not enabled:
            return v
        return np.array([-float(v[1]), float(v[0]), float(v[2])], dtype=np.float64)


def _play_viz_enabled(args: argparse.Namespace) -> bool:
    flag = getattr(args, "play_viz", None)
    if flag is True:
        return True
    if flag is False:
        return False
    return os.environ.get("X5_PLAY_VIZ", "").strip().lower() in ("1", "true", "yes", "on")


def _maybe_publish_play_viz(
    *,
    q_cmd: Optional[Sequence[float]],
    joint_pos: Optional[Sequence[float]],
    end_pos: Optional[Sequence[float]],
    cam_high: Optional[np.ndarray],
    cam_wrist: Optional[np.ndarray],
    meta: Optional[dict],
) -> None:
    """Best-effort publish to /tmp/x5_play_viz; never raises into control loop."""
    global _PLAY_VIZ_IMPORT_WARNED
    try:
        from x5_play_viz.client_hook import try_publish
    except ImportError:
        if not _PLAY_VIZ_IMPORT_WARNED:
            print(
                "[play-viz] x5_play_viz not importable — add scripts/tool to PYTHONPATH "
                "(see tool/doc/04_真机启动命令_含UI.md)",
                flush=True,
            )
            _PLAY_VIZ_IMPORT_WARNED = True
        return
    try_publish(
        q_cmd=None if q_cmd is None else np.asarray(q_cmd, dtype=np.float64),
        joint_pos=None if joint_pos is None else np.asarray(joint_pos, dtype=np.float64),
        end_pos=None if end_pos is None else np.asarray(end_pos, dtype=np.float64),
        cam_high=cam_high,
        cam_wrist=cam_wrist,
        meta=meta or {},
    )


def _smoothstep(alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


def _pose_distance_qpos(a: np.ndarray, b: np.ndarray) -> float:
    a6 = np.asarray(a[:ARM_JOINT_DIMS], dtype=np.float64)
    b6 = np.asarray(b[:ARM_JOINT_DIMS], dtype=np.float64)
    delta = np.arctan2(np.sin(b6 - a6), np.cos(b6 - a6))
    return float(np.max(np.abs(delta)))


def _blend_qpos_angular(
    cur_qpos: np.ndarray, target_qpos: np.ndarray, alpha: float
) -> np.ndarray:
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


def _adaptive_blend_duration(
    delta: float, *, min_sec: float, max_sec: float, speed: float
) -> float:
    return min(max_sec, max(min_sec, abs(float(delta)) / max(float(speed), 1e-6)))


def _rate_limit_joints(
    cur_q: Sequence[float],
    target_q: Sequence[float],
    max_delta_rad: float,
) -> np.ndarray:
    """Move toward target but cap per-step joint change (shortest-path)."""
    cur = np.asarray(cur_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    tgt = np.asarray(target_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    if max_delta_rad <= 0:
        return tgt.copy()
    out = cur.copy()
    for j in range(ARM_JOINT_DIMS):
        d = float(np.arctan2(np.sin(tgt[j] - cur[j]), np.cos(tgt[j] - cur[j])))
        d = float(np.clip(d, -max_delta_rad, max_delta_rad))
        out[j] = cur[j] + d
    return out


def _ema_joints(
    prev_q: Sequence[float],
    new_q: Sequence[float],
    alpha: float,
) -> np.ndarray:
    """Exponential blend toward new_q (alpha in [0,1]; 1=follow new fully)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if a >= 1.0 - 1e-9:
        return np.asarray(new_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS].copy()
    if a <= 1e-9:
        return np.asarray(prev_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS].copy()
    prev = np.asarray(prev_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    new = np.asarray(new_q, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS]
    out = prev.copy()
    for j in range(ARM_JOINT_DIMS):
        d = float(np.arctan2(np.sin(new[j] - prev[j]), np.cos(new[j] - prev[j])))
        out[j] = prev[j] + a * d
    return out


def _startup_go_home(
    node,
    *,
    home_joints6: Sequence[float],
    gripper_open: float,
    control_hz: float,
    blend_speed_rad_s: float,
    max_blend_sec: float,
    lock_sec: float,
    dry_run: bool,
    settle_tol_rad: float = 0.05,
    settle_timeout_sec: float = 8.0,
) -> None:
    """Smooth mode=5 blend to collection go_home, then open gripper + lock.

    Mirrors teleop_record / episode_replay.prepare_before_recording (no U-Arm sync).
    After lock, closed-loop confirm each joint reaches home within settle_tol_rad;
    keep re-publishing home until converged or settle_timeout_sec (open-loop blend
    alone can leave a joint short, e.g. J5 observed -96deg instead of 0).
    """
    period = 1.0 / max(control_hz, 1.0)
    rclpy.spin_once(node, timeout_sec=0.0)
    state = node.latest_state
    if state is None:
        print("[go_home] no /arm_status yet; skip", flush=True)
        return

    home6 = np.asarray(home_joints6, dtype=np.float64)[:ARM_JOINT_DIMS]
    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    target_qpos = np.concatenate([home6, [float(gripper_open)]])
    dist = _pose_distance_qpos(cur_qpos, target_qpos)
    print(
        f"[go_home] offset {dist:.3f} rad ({np.degrees(dist):.2f}° max-joint) "
        f"-> home={list(np.round(home6, 3))} gripper={gripper_open:.2f} "
        f"dry_run={dry_run}",
        flush=True,
    )

    if dist > 0.002:
        duration = _adaptive_blend_duration(
            dist, min_sec=0.5, max_sec=max_blend_sec, speed=blend_speed_rad_s
        )
        n_blend = max(5, int(duration / period))
        print(
            f"[go_home] angular blend {duration:.2f}s ({n_blend} steps @ {control_hz:.0f}Hz)",
            flush=True,
        )
        for k in range(1, n_blend + 1):
            t0 = time.perf_counter()
            alpha = _smoothstep(k / n_blend)
            qpos_b = _blend_qpos_angular(cur_qpos, target_qpos, alpha)
            if not dry_run:
                node.publish_joint_command(
                    qpos_b[:ARM_JOINT_DIMS].tolist(),
                    float(qpos_b[GRIPPER_JOINT_IDX]),
                    mode=MODE_POSITION_CONTROL,
                )
            rclpy.spin_once(node, timeout_sec=0.0)
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    # Gripper blend to open (joints hold home)
    rclpy.spin_once(node, timeout_sec=0.0)
    state = node.latest_state
    if state is not None:
        g0 = float(state.joint_pos[6]) if state.joint_pos.shape[0] > 6 else float(gripper_open)
        joints = home6.tolist()
        g_delta = abs(float(gripper_open) - g0)
        g_dur = _adaptive_blend_duration(g_delta, min_sec=0.5, max_sec=3.0, speed=2.0)
        n_g = max(1, int(g_dur / period))
        print(
            f"[go_home] gripper {g0:.3f} -> {gripper_open:.3f} over {g_dur:.2f}s",
            flush=True,
        )
        for k in range(1, n_g + 1):
            t0 = time.perf_counter()
            alpha = _smoothstep(k / n_g)
            grip = g0 * (1.0 - alpha) + float(gripper_open) * alpha
            if not dry_run:
                node.publish_joint_command(joints, float(grip), mode=MODE_POSITION_CONTROL)
            rclpy.spin_once(node, timeout_sec=0.0)
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

    n_lock = max(1, int(lock_sec / period))
    print(f"[go_home] lock home {lock_sec:.1f}s ({n_lock} steps)", flush=True)
    for _ in range(n_lock):
        t0 = time.perf_counter()
        if not dry_run:
            node.publish_joint_command(
                home6.tolist(), float(gripper_open), mode=MODE_POSITION_CONTROL
            )
        rclpy.spin_once(node, timeout_sec=0.0)
        elapsed = time.perf_counter() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    # Closed-loop settle: keep holding home until every joint is within tol.
    if not dry_run:
        settle_deadline = time.perf_counter() + float(settle_timeout_sec)
        while True:
            rclpy.spin_once(node, timeout_sec=0.0)
            state = node.latest_state
            if state is None:
                break
            cur = np.asarray(state.joint_pos[:ARM_JOINT_DIMS], dtype=np.float64)
            per_joint_err = np.abs(cur - home6)
            worst = float(per_joint_err.max())
            if worst <= settle_tol_rad:
                print(
                    f"[go_home] settled max-joint-err={worst:.4f} rad "
                    f"({np.degrees(worst):.2f} deg)",
                    flush=True,
                )
                break
            if time.perf_counter() >= settle_deadline:
                bad = int(np.argmax(per_joint_err))
                print(
                    f"[go_home][warn] settle timeout {settle_timeout_sec:.1f}s: "
                    f"J{bad} err={worst:.4f} rad ({np.degrees(worst):.2f} deg) "
                    f"cur={list(np.round(cur,3))}",
                    flush=True,
                )
                break
            node.publish_joint_command(
                home6.tolist(), float(gripper_open), mode=MODE_POSITION_CONTROL
            )
            time.sleep(period)

    rclpy.spin_once(node, timeout_sec=0.0)
    state = node.latest_state
    if state is not None:
        residual = _pose_distance_qpos(state.joint_pos[:7], target_qpos)
        print(f"[go_home] done residual={residual:.4f} rad", flush=True)
    else:
        print("[go_home] done", flush=True)


def _ensure_starvla_imports(starvla_root: str) -> None:
    root = os.path.abspath(starvla_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _binarize_gripper_open(open_val: np.ndarray) -> float:
    v = float(np.asarray(open_val, dtype=np.float32).reshape(-1)[0])
    return 1.0 - 2.0 * (v > 0.5)


def _clip_xyz(
    xyz: np.ndarray,
    min_xyz: Sequence[float],
    max_xyz: Sequence[float],
) -> np.ndarray:
    out = xyz.copy()
    out[0] = np.clip(out[0], min_xyz[0], max_xyz[0])
    out[1] = np.clip(out[1], min_xyz[1], max_xyz[1])
    out[2] = np.clip(out[2], min_xyz[2], max_xyz[2])
    return out


def _map_gripper(gripper_sign: float, open_val: float, close_val: float) -> float:
    return close_val if gripper_sign > 0 else open_val


def _closedness_to_qpos(
    closedness: float, qpos_open: float, qpos_closed: float
) -> float:
    """OpenPI closedness [0,1] (0=open, 1=closed) -> ARX qpos (linear).

    qpos = qpos_open - closedness * (qpos_open - qpos_closed).
    Inverse of StateGripperQposToOpenPI: closedness=(qpos_open-qpos)/span.
    Matches gripper_map.py. 0 -> qpos_open (开), 1 -> qpos_closed (闭).
    """
    c = float(np.clip(closedness, 0.0, 1.0))
    return qpos_open - c * (qpos_open - qpos_closed)


def _qpos_to_closedness(qpos: float, qpos_open: float, qpos_closed: float) -> float:
    """ARX qpos -> OpenPI closedness [0,1] (0=open, 1=closed). Inverse of above."""
    span = qpos_open - qpos_closed
    if abs(span) < 1e-9:
        return 0.0
    return float(np.clip((qpos_open - qpos) / span, 0.0, 1.0))


def _preprocess_image(
    img: np.ndarray,
    flip_180: bool,
    rotate_ccw_deg: float = 0.0,
) -> np.ndarray:
    arr = np.ascontiguousarray(img)
    if abs(rotate_ccw_deg) > 1e-6:
        k = int(round(rotate_ccw_deg / 90.0)) % 4
        if k == 1:
            arr = np.rot90(arr, k=1)
        elif k == 2:
            arr = arr[::-1, ::-1]
        elif k == 3:
            arr = np.rot90(arr, k=3)
    if flip_180:
        arr = arr[::-1, ::-1]
    return arr


def _publish_arm_command(
    node,
    *,
    xyzrpy: Sequence[float],
    joint_pos: Sequence[float],
    gripper: float,
    mode: int,
    dry_run: bool,
) -> None:
    from lewm_x5_grasp.schemas import ArmCommand

    command = ArmCommand(
        xyzrpy=[float(v) for v in xyzrpy[:6]],
        gripper=float(gripper),
        joint_pos=[float(v) for v in joint_pos[:6]],
        mode=int(mode),
    )
    if not dry_run:
        node.publish_command(command)


def _eef_after_base_rotation(eef6: np.ndarray, delta_rad: float) -> np.ndarray:
    """Approximate FK for pure base-joint rotation about +Z."""
    x, y, z, roll, pitch, yaw = [float(v) for v in eef6[:6]]
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    return np.array(
        [c * x - s * y, s * x + c * y, z, roll, pitch, yaw + delta_rad],
        dtype=np.float64,
    )


def _enter_position_control(node, dry_run: bool) -> None:
    rclpy.spin_once(node, timeout_sec=0.0)
    state = node.latest_state
    if state is None:
        return
    _publish_arm_command(
        node,
        xyzrpy=state.end_pos[:6],
        joint_pos=state.joint_pos[:6],
        gripper=(
            float(state.joint_pos[6])
            if state.joint_pos.shape[0] > 6
            else 0.0
        ),
        mode=MODE_POSITION_CONTROL,
        dry_run=dry_run,
    )


def _publish_replay_frame(
    node,
    eef_row: np.ndarray,
    qpos_row: np.ndarray,
    dry_run: bool,
) -> None:
    """Match record_teach: send consistent EEF + joint in mode=5."""
    _publish_arm_command(
        node,
        xyzrpy=eef_row[:6],
        joint_pos=qpos_row[:6],
        gripper=float(qpos_row[6]),
        mode=MODE_POSITION_CONTROL,
        dry_run=dry_run,
    )


def _hold_replay_frame(
    node,
    eef_row: np.ndarray,
    qpos_row: np.ndarray,
    hold_sec: float,
    control_hz: float,
    dry_run: bool,
) -> None:
    steps = max(5, int(hold_sec * control_hz))
    dt = 1.0 / max(control_hz, 1.0)
    for _ in range(steps):
        rclpy.spin_once(node, timeout_sec=0.0)
        _publish_replay_frame(node, eef_row, qpos_row, dry_run=dry_run)
        time.sleep(dt)


def _blend_joint0_ccw(
    node,
    joint_index: int,
    delta_rad: float,
    move_sec: float,
    control_hz: float,
    hold_sec: float,
    dry_run: bool,
) -> None:
    """Blend joint0 + matching EEF like record_teach _replay_blend_to_start."""
    _enter_position_control(node, dry_run=dry_run)
    time.sleep(0.3)

    rclpy.spin_once(node, timeout_sec=0.0)
    state = node.latest_state
    if state is None:
        return

    cur_qpos = np.asarray(state.joint_pos[:7], dtype=np.float64)
    cur_eef = np.asarray(state.end_pos[:6], dtype=np.float64)
    target_qpos = cur_qpos.copy()
    target_qpos[joint_index] += delta_rad
    target_eef = _eef_after_base_rotation(cur_eef, delta_rad)

    if abs(target_qpos[joint_index] - cur_qpos[joint_index]) < 0.01:
        return

    steps = max(10, int(move_sec * control_hz))
    dt = 1.0 / max(control_hz, 1.0)
    node.get_logger().info(
        "Rotating base joint[%d]: %.3f -> %.3f rad over %.1f s @ %.0f Hz "
        "(mode=5, paired eef+qpos)"
        % (
            joint_index,
            cur_qpos[joint_index],
            target_qpos[joint_index],
            move_sec,
            control_hz,
        )
    )
    for k in range(1, steps + 1):
        alpha = k / steps
        qpos_b = cur_qpos * (1.0 - alpha) + target_qpos * alpha
        eef_b = _eef_after_base_rotation(cur_eef, alpha * delta_rad)
        t0 = time.monotonic()
        rclpy.spin_once(node, timeout_sec=0.0)
        _publish_replay_frame(node, eef_b, qpos_b, dry_run=dry_run)
        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    _hold_replay_frame(
        node=node,
        eef_row=target_eef,
        qpos_row=target_qpos,
        hold_sec=hold_sec,
        control_hz=control_hz,
        dry_run=dry_run,
    )


def _warmup_end_control_from_feedback(
    node,
    warmup_sec: float,
    control_hz: float,
    dry_run: bool,
) -> None:
    if warmup_sec <= 0.0:
        return
    steps = max(5, int(warmup_sec * control_hz))
    dt = 1.0 / max(control_hz, 1.0)
    node.get_logger().info(
        "Optional mode=4 warmup at measured pose for %.1f s" % warmup_sec
    )
    for _ in range(steps):
        rclpy.spin_once(node, timeout_sec=0.0)
        state = node.latest_state
        if state is None:
            time.sleep(dt)
            continue
        gripper = (
            float(state.joint_pos[6])
            if state.joint_pos.shape[0] > 6
            else 0.0
        )
        _publish_arm_command(
            node,
            xyzrpy=state.end_pos[:6],
            joint_pos=state.joint_pos[:6],
            gripper=gripper,
            mode=MODE_END_CONTROL,
            dry_run=dry_run,
        )
        time.sleep(dt)


# Joint-clean run dataset_statistics.json → state min/max (min_max → [-1,1])
_JOINT_STATE_MIN = np.array(
    [
        0.5128936767578125,
        -0.01697540283203125,
        -0.01392364501953125,
        -0.7455940246582031,
        -1.2823305130004883,
        -1.7217903137207031,
        1.9707026481628418,
    ],
    dtype=np.float32,
)
_JOINT_STATE_MAX = np.array(
    [
        2.7018003463745117,
        2.2066450119018555,
        2.3531322479248047,
        1.2788963317871094,
        0.9718093872070312,
        1.588273048400879,
        4.4880595207214355,
    ],
    dtype=np.float32,
)

# LIBERO/franka-style EEF state for this ckpt: [x,y,z,roll,pitch,yaw,gripper] (7,)
# Server meta lists pad in state_keys, but DiT state proj is 7x1024 — do NOT send pad.
# Fallback ranges from local ARX teleop HDF5; override via --state-stats-json.
_EEF_STATE_MIN = np.array(
    [-0.35, -0.05, -0.12, -2.2, -0.85, -3.2, 1.5],
    dtype=np.float32,
)
_EEF_STATE_MAX = np.array(
    [0.15, 0.50, 0.35, 2.0, 0.85, 3.2, 4.6],
    dtype=np.float32,
)

# Scratch 8D (arxx5_scratch_0813): [xyz, rotvec, +g, -g]. Prefer --state-stats-json.
_EEF8_STATE_MIN = np.array(
    [-0.35, -0.05, -0.12, -2.2, -0.85, -3.2, 1.5, -4.6],
    dtype=np.float32,
)
_EEF8_STATE_MAX = np.array(
    [0.15, 0.50, 0.35, 2.0, 0.85, 3.2, 4.6, -1.5],
    dtype=np.float32,
)


def _normalize_state_minmax(
    state: Sequence[float],
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Match training StateActionTransform mode=min_max → [-1, 1]."""
    x = np.asarray(state, dtype=np.float32).reshape(-1)
    lo = np.asarray(lo, dtype=np.float32).reshape(-1)
    hi = np.asarray(hi, dtype=np.float32).reshape(-1)
    n = min(x.shape[0], lo.shape[0], hi.shape[0])
    x = x[:n]
    lo = lo[:n]
    hi = hi[:n]
    denom = hi - lo
    out = np.zeros(n, dtype=np.float32)
    ok = np.abs(denom) > 1e-8
    out[ok] = np.clip((x[ok] - lo[ok]) / denom[ok] * 2.0 - 1.0, -1.0, 1.0)
    # Constant dims (e.g. pad=0): keep 0 after normalize.
    return out


def _state_raw_and_norm(
    state_vec: Sequence[float],
    *,
    normalize: bool,
    state_lo: np.ndarray,
    state_hi: np.ndarray,
) -> tuple[list[float], list[float]]:
    """Raw + as-sent (minmax or raw) for offset A/B dumps."""
    raw = [float(x) for x in np.asarray(state_vec, dtype=np.float32).reshape(-1)]
    if normalize:
        norm = _normalize_state_minmax(state_vec, state_lo, state_hi)
        sent = [float(x) for x in norm.reshape(-1)]
    else:
        n = min(len(raw), int(np.asarray(state_lo).reshape(-1).shape[0]))
        sent = raw[:n]
    return raw, sent


def _eef_offset_meta() -> dict:
    """What offset YAML the process is using (for dump / startup logs)."""
    try:
        from x5_play_viz.eef_offset import get_eef_xyz_offset, resolve_params_yaml

        off = get_eef_xyz_offset()
        return {
            "eef_xyz_offset_m": [float(x) for x in off.reshape(-1)[:3]],
            "eef_offset_yaml": str(resolve_params_yaml()),
        }
    except Exception as exc:
        # Fallback path may still apply offset; report probe result.
        probe = _current_xyz_from_sdk(np.zeros(3, dtype=np.float64))
        return {
            "eef_xyz_offset_m": [float(x) for x in probe.reshape(-1)[:3]],
            "eef_offset_yaml": f"(fallback; {exc})",
        }


def _maybe_write_pred_jsonl(
    pred_log,
    *,
    closed_loop: bool,
    step: int,
    action_repeat: int,
    inferred: bool,
    row: dict,
) -> None:
    if pred_log is None or not inferred:
        return
    if closed_loop and step % max(1, action_repeat) != 0:
        return
    import json

    pred_log.write(json.dumps(row, ensure_ascii=False) + "\n")
    pred_log.flush()


class _RolloutRecorder:
    """Record per-step camera video (mp4) + trajectory (jsonl) for later analysis.

    Writes into ``<record_dir>/<YYYYMMDD_HHMMSS>/``:
      - ``cam_high.avi`` / ``cam_wrist.avi`` (MJPG, plays in any player without codecs;
        cam_high only if dual)
      - ``traj.jsonl``: one row per control step with wall-time, joint pos, EEF pose,
        raw model action, applied gripper cmd, ik_ok, rtc info, etc.
      - ``meta.json``: run config snapshot (task, ckpt, hz, action_space, gripper mode).
    Recording is best-effort: any per-step failure is logged and recording continues.
    """

    def __init__(self, record_dir: str, *, fps: float, meta: dict) -> None:
        import json as _json

        if cv2 is None:
            raise RuntimeError("cv2 not available; cannot record video")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.out = Path(record_dir).expanduser() / ts
        self.out.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps), 1.0)
        # MJPG (Motion-JPEG .avi): universally playable, no external codec needed
        # (mp4v/H.264 often missing decoder on the control laptop).
        self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._ext = ".avi"
        self._vw = {}  # name -> VideoWriter
        self._traj = open(self.out / "traj.jsonl", "a", encoding="utf-8")
        with open(self.out / "meta.json", "w", encoding="utf-8") as f:
            _json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[record] rollout → {self.out} (fps={self.fps:.1f})", flush=True)

    def _writer(self, name: str, frame: np.ndarray):
        vw = self._vw.get(name)
        if vw is None:
            h, w = frame.shape[:2]
            vw = cv2.VideoWriter(
                str(self.out / f"{name}{self._ext}"), self._fourcc, self.fps, (w, h)
            )
            self._vw[name] = vw
        return vw

    def write(self, name: str, frame_rgb: np.ndarray) -> None:
        try:
            if frame_rgb is None:
                return
            # image_source.read() returns RGB; OpenCV VideoWriter expects BGR.
            # Convert so recorded video colors match reality (else orange looks blue).
            if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3:
                bgr = frame_rgb[:, :, ::-1]
            else:
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
            self._writer(name, bgr).write(bgr)
        except Exception as exc:  # noqa: BLE001
            print(f"[record][warn] video {name} write failed: {exc}", flush=True)

    def log(self, row: dict) -> None:
        import json as _json

        try:
            self._traj.write(_json.dumps(row, ensure_ascii=False) + "\n")
            self._traj.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[record][warn] traj log failed: {exc}", flush=True)

    def close(self) -> None:
        for vw in self._vw.values():
            try:
                vw.release()
            except Exception:  # noqa: BLE001
                pass
        self._vw.clear()
        try:
            self._traj.close()
        except Exception:  # noqa: BLE001
            pass
        print(f"[record] saved → {self.out}", flush=True)


def _resolve_state_modality(action_space: str, override: str = "joint") -> str:
    """Default = joint qpos. EEF (SDK end_pos) only when explicitly requested."""
    del action_space  # kept for call-site compatibility; does not auto-switch
    o = (override or "joint").strip().lower()
    if o in ("eef", "auto_eef"):
        return "eef"
    # "auto" and "joint" both → joint (safe default; EEF needs an explicit switch)
    return "joint"


def _resolve_eef_gripper(
    end_pos: Sequence[float],
    joint_pos: Optional[Sequence[float]] = None,
) -> float:
    ep = np.asarray(end_pos, dtype=np.float64).reshape(-1)
    if joint_pos is not None and len(joint_pos) > 6:
        return float(joint_pos[6])
    if ep.shape[0] > 6:
        return float(ep[6])
    return 0.0


def _eef_xyzrpy_for_state(
    end_pos: Sequence[float],
    joint_pos: Optional[Sequence[float]] = None,
    *,
    source: str = "sdk_offset",
) -> np.ndarray:
    """Build xyzrpy (6,) for policy proprio.

    ``sdk_offset`` (legacy): SDK flange xyz + YAML ``eef_xyz_offset_m``, SDK rpy.
    ``eef_link_fk``: URDF tip via ``eef_link_fk(q)`` — matches ``bake_lerobot_eef_from_fk``.
    """
    src = (source or "sdk_offset").strip().lower()
    if src in ("eef_link_fk", "fk_tip", "tip", "eef_link"):
        if joint_pos is None or len(joint_pos) < 6:
            raise ValueError("eef_link_fk state needs joint_pos[:6]")
        from lewm_x5_grasp.x5_fk import eef_link_fk

        return np.asarray(eef_link_fk(joint_pos[:6]), dtype=np.float64).reshape(6)

    ep = np.asarray(end_pos, dtype=np.float64).reshape(-1)
    eef6 = np.zeros(6, dtype=np.float64)
    eef6[: min(6, ep.shape[0])] = ep[:6]
    return _eef_with_xyz_offset(eef6)


def _sdk_eef_state7(
    end_pos: Sequence[float],
    joint_pos: Optional[Sequence[float]] = None,
    *,
    eef_state_source: str = "sdk_offset",
) -> np.ndarray:
    """EEF → policy state [x,y,z,roll,pitch,yaw,gripper] (7,).

    Default ``sdk_offset`` keeps legacy 1p/2p behavior. Pass
    ``eef_state_source=eef_link_fk`` for 0806 bake-aligned tip.
    No pad (2p DiT state Linear is 7×1024).
    """
    xyzrpy = _eef_xyzrpy_for_state(
        end_pos, joint_pos, source=eef_state_source
    ).astype(np.float32)
    grip = _resolve_eef_gripper(end_pos, joint_pos)
    # bake prefers joint gripper; for sdk_offset keep historical preference: end_pos[6] first
    src = (eef_state_source or "sdk_offset").strip().lower()
    if src in ("sdk_offset", "offset", "sdk"):
        ep = np.asarray(end_pos, dtype=np.float64).reshape(-1)
        if ep.shape[0] > 6:
            grip = float(ep[6])
        elif joint_pos is not None and len(joint_pos) > 6:
            grip = float(joint_pos[6])
        else:
            grip = 0.0
    return np.concatenate([xyzrpy, np.array([grip], dtype=np.float32)])


def _euler_xyz_to_rotvec(rpy: Sequence[float]) -> np.ndarray:
    """Extrinsic xyz Euler (rad) → rotvec; matches ``scipy Rotation.from_euler('xyz')``.

    Pure NumPy — avoids broken system SciPy vs user NumPy 2.x ABI mismatch.
    """
    rx, ry, rz = (float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(-1)[:3])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    # R = Rz @ Ry @ Rx (scipy extrinsic 'xyz')
    R = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    tr = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(np.pi - theta) < 1e-6:
        # Near-π: more stable axis from diagonal
        xx = max(0.0, (R[0, 0] + 1.0) * 0.5)
        yy = max(0.0, (R[1, 1] + 1.0) * 0.5)
        zz = max(0.0, (R[2, 2] + 1.0) * 0.5)
        axis = np.array([np.sqrt(xx), np.sqrt(yy), np.sqrt(zz)], dtype=np.float64)
        # Fix signs from off-diagonals
        if R[2, 1] < R[1, 2]:
            axis[0] = -axis[0]
        if R[0, 2] < R[2, 0]:
            axis[1] = -axis[1]
        if R[1, 0] < R[0, 1]:
            axis[2] = -axis[2]
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return axis * (theta / n)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis *= theta / (2.0 * np.sin(theta))
    return axis


def _sdk_eef_state8(
    end_pos: Sequence[float],
    joint_pos: Optional[Sequence[float]] = None,
    *,
    eef_state_source: str = "sdk_offset",
    state_gripper_mode: str = "sign",
    gripper_qpos_open: float = 4.491874016821384,
    gripper_qpos_closed: float = -0.06599569314857945,
) -> np.ndarray:
    """EEF → scratch 8D state.

    ``sign`` (0813): [xyz, rotvec, +g, -g] raw qpos.
    ``openpi`` (ljn pick stage2): [xyz, rotvec, pad=0, gripper=closedness] where
    closedness = (qpos_open - qpos)/(qpos_open - qpos_closed) clipped to [0,1].
    Matches StateGripperQposToOpenPI: pad cleared to 0, gripper -> closedness.
    See ``_sdk_eef_state7`` / ``--eef-state-source`` for xyzrpy frame.
    """
    s7 = _sdk_eef_state7(
        end_pos, joint_pos, eef_state_source=eef_state_source
    )
    out = np.zeros(8, dtype=np.float32)
    out[0:3] = s7[0:3]
    out[3:6] = _euler_xyz_to_rotvec(s7[3:6]).astype(np.float32)
    g = float(s7[6])
    if state_gripper_mode == "openpi":
        out[6] = 0.0  # pad cleared (StateGripperQposToOpenPI)
        out[7] = _qpos_to_closedness(g, gripper_qpos_open, gripper_qpos_closed)
    else:
        out[6] = g
        out[7] = -g
    return out


def _load_state_minmax_from_json(
    path: str,
    *,
    unnorm_key: str,
    modality: str,
    state_dim: int = 7,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Optional override from dataset_statistics.json (ckpt sibling)."""
    try:
        import json
        from pathlib import Path as _P

        p = _P(path).expanduser()
        if not p.is_file():
            print(f"[state] stats file not found: {p}", flush=True)
            return None
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        block = raw.get(unnorm_key, raw)
        st = block.get("state", block)
        lo = st.get("min") or st.get("q01")
        hi = st.get("max") or st.get("q99")
        if lo is None or hi is None:
            print(f"[state] no state min/max in {p} key={unnorm_key}", flush=True)
            return None
        lo_a = np.asarray(lo, dtype=np.float32).reshape(-1)
        hi_a = np.asarray(hi, dtype=np.float32).reshape(-1)
        want = int(state_dim)
        if lo_a.shape[0] < want or hi_a.shape[0] < want:
            print(
                f"[state] stats dim {lo_a.shape[0]} < want {want} in {p}; ignore",
                flush=True,
            )
            return None
        # Legacy 7D EEF: if stats are 8-D (…, pad, gripper), drop pad → [xyzrpy, gripper]
        if want == 7 and lo_a.shape[0] >= 8 and modality == "eef":
            lo_a = np.concatenate([lo_a[:6], lo_a[7:8]])
            hi_a = np.concatenate([hi_a[:6], hi_a[7:8]])
        print(
            f"[state] loaded {modality} min/max dim={want} from {p} ({unnorm_key})",
            flush=True,
        )
        return lo_a[:want], hi_a[:want]
    except Exception as exc:
        print(f"[state] failed to load stats ({exc})", flush=True)
        return None


def _normalize_state_seq_minmax(
    state_seq: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Normalize (T, D) with shared per-dim min/max → [-1, 1]."""
    x = np.asarray(state_seq, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    lo = np.asarray(lo, dtype=np.float32).reshape(-1)
    hi = np.asarray(hi, dtype=np.float32).reshape(-1)
    d = min(x.shape[1], lo.shape[0], hi.shape[0])
    x = x[:, :d]
    denom = hi[:d] - lo[:d]
    out = np.zeros_like(x, dtype=np.float32)
    ok = np.abs(denom) > 1e-8
    out[:, ok] = np.clip(
        (x[:, ok] - lo[ok]) / denom[ok] * 2.0 - 1.0, -1.0, 1.0
    )
    return out


def _build_example_dict(
    wrist_rgb: np.ndarray,
    task: str,
    camera_mode: str,
    flip_180: bool,
    agent_rgb: Optional[np.ndarray] = None,
    agent_rotate_ccw_deg: float = 0.0,
    state_vec: Optional[Sequence[float]] = None,
    state_seq: Optional[np.ndarray] = None,
    normalize_state: bool = True,
    state_lo: Optional[np.ndarray] = None,
    state_hi: Optional[np.ndarray] = None,
) -> dict:
    wrist = _preprocess_image(wrist_rgb, flip_180=flip_180)
    if camera_mode == "dual":
        if agent_rgb is None:
            raise ValueError("camera_mode=dual requires desktop/agent camera frame.")
        agent = _preprocess_image(
            agent_rgb,
            flip_180=flip_180,
            rotate_ccw_deg=agent_rotate_ccw_deg,
        )
        # Train ArxX5Joint: video_keys = [cam_high, cam_wrist] → image_0, image_1
        images = [agent, wrist]
    elif camera_mode in ("duplicate", "wrist_as_agent"):
        images = [wrist, wrist]
    else:
        raise ValueError(f"Unknown camera_mode: {camera_mode}")

    out: dict = {"image": images, "lang": task}
    if state_seq is not None:
        lo = _JOINT_STATE_MIN if state_lo is None else state_lo
        hi = _JOINT_STATE_MAX if state_hi is None else state_hi
        seq = np.asarray(state_seq, dtype=np.float32)
        if seq.ndim == 1:
            seq = seq.reshape(1, -1)
        if normalize_state:
            st = _normalize_state_seq_minmax(seq, lo, hi)
        else:
            d = min(seq.shape[1], np.asarray(lo).reshape(-1).shape[0])
            st = seq[:, :d]
        out["state"] = st
    elif state_vec is not None:
        lo = _JOINT_STATE_MIN if state_lo is None else state_lo
        hi = _JOINT_STATE_MAX if state_hi is None else state_hi
        if normalize_state:
            st = _normalize_state_minmax(state_vec, lo, hi)
        else:
            st = np.asarray(state_vec, dtype=np.float32).reshape(-1)
            n = min(st.shape[0], np.asarray(lo).reshape(-1).shape[0])
            st = st[:n]
        # Training batch state is [B, T, D] with T=1 (D=7 joint or EEF xyzrpy+gripper)
        out["state"] = st.reshape(1, -1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO StarVLA policy on ARX X5 via WebSocket + ROS2."
    )
    parser.add_argument(
        "--starvla-root",
        default=os.environ.get("STARVLA_ROOT", DEFAULT_STARVLA_ROOT),
        help="Path to starVLA repo (for ModelClient imports).",
    )
    parser.add_argument("--host", default=os.environ.get("POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLICY_PORT", "5694")))
    parser.add_argument(
        "--unnorm-key",
        default=os.environ.get("UNNORM_KEY", "franka"),
        help="Must match checkpoint dataset_statistics key (EEF: franka; joint: new_embodiment).",
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "pick and place the can"),
        help="English instruction aligned with LeRobot tasks.jsonl.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--action-space",
        choices=("joint", "eef_via_joint", "eef"),
        default=os.environ.get("ACTION_SPACE", "joint"),
        help=(
            "joint: abs_qpos → mode=5; "
            "eef_via_joint: OSC Δ → IK → mode=5 (推荐给 EEF 模型，避开 mode=4 奇异); "
            "eef: mode=4 直发末端（实机不推荐）."
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=float(os.environ.get("CONTROL_HZ", "20")),
        help="EEF models: 20; joint models: 30.",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=0.05,
        help="EEF OSC physical scale for xyz (LIBERO conversion uses 0.05).",
    )
    parser.add_argument(
        "--rot-scale",
        type=float,
        default=0.5,
        help="EEF OSC physical scale for rotation (LIBERO conversion uses 0.5).",
    )
    parser.add_argument(
        "--frame-align",
        default=os.environ.get("FRAME_ALIGN", "none"),
        choices=("none", "neg_x_z1"),
        help="Match training dataset frame tag. For 1p/2p libero-aligned EEF "
        "(x'=-x, z'=z+1, R'=SRS): use neg_x_z1 to inverse action deltas to ARX. "
        "Joint 3p: none.",
    )
    parser.add_argument(
        "--latency-log",
        action="store_true",
        default=True,
        help="Print T_infer / T_loop summary (default on).",
    )
    parser.add_argument("--no-latency-log", action="store_false", dest="latency_log")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions only; do not publish /arm_cmd.",
    )
    parser.add_argument(
        "--flip-180",
        action="store_true",
        help="Apply LIBERO sim image flip [::-1, ::-1] to D405 frames.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("duplicate", "wrist_as_agent", "dual"),
        default="duplicate",
        help="duplicate/wrist_as_agent: D405 only; dual: D435i desktop + D405 wrist.",
    )
    parser.add_argument(
        "--desktop-camera",
        action="store_true",
        help="Enable D435/D435i desktop camera as LIBERO agentview (cam_high).",
    )
    parser.add_argument(
        "--desktop-serial",
        default=None,
        help="Desktop RealSense serial (auto-detect D435/D435i if omitted).",
    )
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--gripper-close", type=float, default=1.2)
    parser.add_argument(
        "--gripper-mode",
        choices=("binary", "continuous_openpi"),
        default="binary",
        help="Gripper output mode. binary (default, 0813): threshold closedness>0.5 "
        "to {--gripper-open, --gripper-close}. continuous_openpi (ljn pick stage2): "
        "map closedness [0,1] linearly to qpos via --gripper-qpos-open/closed. "
        "0=open, 1=closed (OpenPI convention).",
    )
    parser.add_argument(
        "--gripper-qpos-open",
        type=float,
        default=4.491874016821384,
        help="continuous_openpi: qpos at fully open (closedness=0). Default arx_x5_ljn.",
    )
    parser.add_argument(
        "--gripper-qpos-closed",
        type=float,
        default=-0.06599569314857945,
        help="continuous_openpi: qpos at fully closed (closedness=1). Default arx_x5_ljn.",
    )
    parser.add_argument(
        "--workspace-min",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="SDK/base_link flange xyz min. Default: x5_play_params.yaml "
        "workspace_min_xyz (else -0.3 0.11 0).",
    )
    parser.add_argument(
        "--workspace-max",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="SDK/base_link flange xyz max. Default: x5_play_params.yaml "
        "workspace_max_xyz (else 0.3 0.55 0.7).",
    )
    parser.add_argument(
        "--d405-width",
        type=int,
        default=640,
        help="Match train video width 640 (was 480 square crop — cuts desktop FOV).",
    )
    parser.add_argument(
        "--d405-height",
        type=int,
        default=480,
        help="Match train video height 480.",
    )
    parser.add_argument(
        "--send-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send proprio state (required for joint DiT; train uses state tokens). "
        "Default on. --no-send-state for debug only.",
    )
    parser.add_argument(
        "--state-modality",
        choices=("joint", "eef", "auto"),
        default=os.environ.get("STATE_MODALITY", "joint"),
        help="Proprio content. Default joint (qpos). "
        "Pass eef (or STATE_MODALITY=eef) to send "
        "[current_xyz(+eef_xyz_offset_m), rpy, gripper] (7-D) for 1p/2p. "
        "'auto' is treated as joint.",
    )
    parser.add_argument(
        "--send-eef-state",
        action="store_true",
        help="Shortcut: --state-modality eef "
        "(offset-frame current_xyz + SDK rpy/gripper, 7-D by default; "
        "use --eef-state-8d for scratch 8D rotvec).",
    )
    parser.add_argument(
        "--eef-state-8d",
        action="store_true",
        help="EEF proprio as scratch 8D [xyz, rotvec, +g, -g] (requires --send-eef-state "
        "or --state-modality eef). Use with --state-history-len 16 for arxx5_scratch_0813.",
    )
    parser.add_argument(
        "--eef-state-source",
        choices=("sdk_offset", "eef_link_fk"),
        default=os.environ.get("EEF_STATE_SOURCE", "sdk_offset"),
        help="EEF proprio frame for --send-eef-state. "
        "sdk_offset (default/legacy): SDK end_pos xyz + YAML eef_xyz_offset_m. "
        "eef_link_fk: joint URDF tip (link6+0.145m), matches bake_lerobot_eef_from_fk / 0806.",
    )
    parser.add_argument(
        "--state-history-len",
        type=int,
        default=int(os.environ.get("STATE_HISTORY_LEN", "1")),
        help="Temporal length T for state sent to policy (default 1). "
        "Scratch 8D train uses 16 (state_indices=range(-16,0)). "
        "When T>1, left-pad by repeating the earliest frame until full.",
    )
    parser.add_argument(
        "--state-stats-json",
        default=os.environ.get("STATE_STATS_JSON", ""),
        help="Optional dataset_statistics.json to override state min/max "
        "(prefer ckpt sibling under unnorm_key).",
    )
    parser.add_argument(
        "--normalize-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="min_max normalize state to [-1,1] (joint-clean or EEF stats; default on).",
    )
    parser.add_argument(
        "--state-gripper-mode",
        choices=("sign", "openpi"),
        default="sign",
        help="state[6:8] convention for --send-eef-state / --eef-state-8d. "
        "sign (default, 0813): [+g, -g] raw qpos. "
        "openpi (ljn pick stage2): [pad=0, gripper=closedness] where "
        "closedness=(qpos_open-qpos)/(qpos_open-qpos_closed) clipped to [0,1]. "
        "Use openpi with --no-normalize-state (train state is not normalized).",
    )
    parser.add_argument(
        "--save-obs-dir",
        default=None,
        help="If set, dump cam_high/cam_wrist JPEGs every --save-obs-every steps for FOV check.",
    )
    parser.add_argument("--save-obs-every", type=int, default=8)
    parser.add_argument("--d405-fps", type=int, default=30)
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live camera preview windows during eval (cam_wrist + cam_high if dual).",
    )
    parser.add_argument(
        "--go-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before policy loop: smooth blend to collection go_home (default: on). "
        "Use --no-go-home to skip.",
    )
    parser.add_argument(
        "--go-home-deg",
        type=float,
        nargs=6,
        default=list(DEFAULT_GO_HOME_DEG),
        metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
        help="Home joint angles in degrees (default: 90 0 0 0 0 0, same as teleop).",
    )
    parser.add_argument(
        "--go-home-gripper",
        type=float,
        default=DEFAULT_GO_HOME_GRIPPER,
        help="Gripper cmd at home (default 4.5, same as teleop gripper_open_cmd).",
    )
    parser.add_argument(
        "--go-home-hz",
        type=float,
        default=30.0,
        help="Command rate during go_home blend/lock.",
    )
    parser.add_argument(
        "--go-home-blend-speed",
        type=float,
        default=0.6,
        help="Adaptive blend speed rad/s (teleop default 0.6).",
    )
    parser.add_argument(
        "--go-home-blend-max-sec",
        type=float,
        default=4.0,
        help="Max blend duration seconds.",
    )
    parser.add_argument(
        "--go-home-lock-sec",
        type=float,
        default=2.5,
        help="Hold exact home after blend (teleop default 2.5).",
    )
    parser.add_argument(
        "--go-home-settle-tol",
        type=float,
        default=0.05,
        help="Closed-loop go_home settle tolerance rad per joint (default 0.05=2.9deg).",
    )
    parser.add_argument(
        "--go-home-settle-timeout",
        type=float,
        default=8.0,
        help="Max seconds to keep re-publishing home until joints settle (default 8).",
    )
    parser.add_argument(
        "--init-joint0-ccw-deg",
        type=float,
        default=0.0,
        help="After go_home, extra base joint CCW degrees (0=skip). Prefer --go-home.",
    )
    parser.add_argument(
        "--init-face-desktop",
        action="store_true",
        help="Deprecated shorthand: if --no-go-home, rotate joint0 +90°. "
        "With default --go-home (j0=90°), usually unnecessary.",
    )
    parser.add_argument(
        "--init-joint-index",
        type=int,
        default=0,
        help="Which joint to rotate for init (default 0 = bottom/base motor).",
    )
    parser.add_argument(
        "--init-move-sec",
        type=float,
        default=3.0,
        help="Seconds to rotate base joint into init pose.",
    )
    parser.add_argument(
        "--init-control-hz",
        type=float,
        default=30.0,
        help="Command rate for init joint move (independent of --control-hz).",
    )
    parser.add_argument(
        "--init-hold-sec",
        type=float,
        default=0.5,
        help="Hold final joint pose (mode=5) after rotation.",
    )
    parser.add_argument(
        "--init-settle-sec",
        type=float,
        default=0.0,
        help="Optional mode=4 hold at measured pose after init (0=skip; may snap).",
    )
    parser.add_argument(
        "--agent-rotate-ccw-deg",
        type=float,
        default=0.0,
        help="Rotate desktop/agentview image CCW (e.g. 90 if D435i is mounted sideways).",
    )
    parser.add_argument(
        "--num-ddim-steps",
        type=int,
        default=4,
        help="Diffusion steps on server (training default 4; lower=faster, less freeze).",
    )
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=2,
        help="Hold each predicted action for N control cycles (default 2). "
        "With --closed-loop this is 'replan every N steps'.",
    )
    parser.add_argument(
        "--max-joint-delta-rad",
        type=float,
        default=None,
        help="Per-cycle joint rate limit (rad). 0=off. Default: x5_play_params.yaml "
        "max_joint_delta_rad (shared with replay v).",
    )
    parser.add_argument(
        "--ik-max-dq",
        type=float,
        default=None,
        help="differential_ik ||Δq|| cap (rad). Default: x5_play_params.yaml ik_max_dq.",
    )
    parser.add_argument(
        "--ik-damp",
        type=float,
        default=None,
        help="differential_ik DLS damp. Default: x5_play_params.yaml ik_damp.",
    )
    parser.add_argument(
        "--eef-soft-entry-sec",
        type=float,
        default=1.0,
        help="For eef_via_joint: after go_home, smoothstep-blend into first IK "
        "targets over this many seconds (0=off). Reduces start jerk.",
    )
    parser.add_argument(
        "--rot-delta-semantics",
        choices=("euler", "rotvec"),
        default="euler",
        help="euler (0813): model rot_delta applied element-wise to extrinsic-XYZ "
        "rpy. rotvec (ljn pick stage2): training actions are element-wise rotvec "
        "diffs = world-frame rotation increments; convert via Rv(d)@R_cur before "
        "IK. At yaw≈90° euler mode executes a roll input as world-pitch (drift).",
    )
    parser.add_argument(
        "--ik-fk",
        choices=("link6", "eef_link"),
        default="link6",
        help="FK point for the differential-IK Jacobian. link6 (legacy): flange. "
        "eef_link: tip = link6+0.145m along +X (train-aligned point; use with "
        "--eef-state-source eef_link_fk).",
    )
    parser.add_argument(
        "--eef-joint-ema",
        type=float,
        default=None,
        help="For eef_via_joint: EMA weight on new IK joints in [0,1] "
        "(1=no filter; 0=freeze). Default: x5_play_params.yaml joint_ema "
        "(shared with replay v).",
    )
    parser.add_argument(
        "--eef-ik-fail-delta-rad",
        type=float,
        default=0.03,
        help="For eef_via_joint: tighter per-step joint cap when IK not ok "
        "(avoids singularity branch jumps).",
    )
    parser.add_argument(
        "--closed-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Receding-horizon: only execute action[0] of each chunk, then re-infer "
        "with fresh images/state (default on). Open-loop 8-step chunks often ignore "
        "object offset and keep going to the mean demo side. --no-closed-loop to disable.",
    )
    parser.add_argument(
        "--rtc-mode",
        choices=("off", "track_b"),
        default=os.environ.get("RTC_MODE", "off"),
        help="off=ModelClient (default). track_b=async leftover + rtc_* fields "
        "(needs P3-patched server; forces --no-closed-loop, action-repeat=1).",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=int(os.environ.get("RTC_EXECUTION_HORIZON", "4")),
        help="RTC execution_horizon (overlap); keep <= chunk size.",
    )
    parser.add_argument(
        "--rtc-prefetch-remaining",
        type=int,
        default=int(os.environ.get("RTC_PREFETCH_REMAINING", "4")),
        help="Prefetch next chunk when queue remaining <= this.",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=float(os.environ.get("PUBLISH_HZ", "0.0")),
        help="0=direct publish (original behavior). >0=enable 100Hz-style "
        "interpolation publish thread (e.g. 100). See interp_publish.py.",
    )
    parser.add_argument(
        "--dump-pred-jsonl",
        default=None,
        help="Append per-infer rows: state_raw/state_sent, world_vector, "
        "clipped_dxyz (eef_via_joint) for offset A/B. Prefer one file per run.",
    )
    parser.add_argument(
        "--log-state-every",
        type=int,
        default=0,
        help="Every N infer steps, print state_raw + state_sent (normalized) + "
        "world_vector. 0=off. Use with --send-eef-state for offset A/B.",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="If set, record rollout into <dir>/<timestamp>/: cam_high/cam_wrist mp4 "
        "+ traj.jsonl (per-step joints/eef/action/gripper) + meta.json. For analysis.",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=0.0,
        help="Video fps for --record-dir (default 0 = use --control-hz).",
    )
    parser.add_argument(
        "--play-viz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish q_cmd/images to /tmp/x5_play_viz for MuJoCo UI "
        "(default: on if env X5_PLAY_VIZ=1). Pair with: bash run_viz.sh --bus",
    )
    args = parser.parse_args()

    if args.init_face_desktop:
        args.init_joint0_ccw_deg = 90.0

    if _play_viz_enabled(args):
        print(
            "[play-viz] ON → writing /tmp/x5_play_viz/latest.npz "
            "(run: bash .../tool/run_viz.sh --bus)",
            flush=True,
        )

    _ensure_starvla_imports(args.starvla_root)
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    from lewm_x5_grasp.arm_ros2 import X5ArmBridge
    from lewm_x5_grasp.image_source import (
        RealSenseD405ImageSource,
        build_record_cameras,
    )
    from lewm_x5_grasp.schemas import ArmCommand

    if args.desktop_camera:
        args.camera_mode = "dual"

    if args.camera_mode != "dual":
        print(
            "[CRITICAL] camera_mode=%s — desktop cam_high is NOT used; "
            "both VLM slots are wrist. Joint model was trained with "
            "[cam_high, cam_wrist]. Re-run with --desktop-camera "
            "(and confirm --display shows two windows)."
            % args.camera_mode
        )
    if args.action_space == "joint" and not args.send_state:
        print(
            "[CRITICAL] --no-send-state: DiT was trained WITH state tokens; "
            "inference without state changes the action-head layout and often "
            "collapses to a stereotyped motion (e.g. always one side)."
        )

    sleep_sec = 1.0 / max(args.control_hz, 0.5)
    save_obs_dir = None
    if args.save_obs_dir:
        from pathlib import Path

        save_obs_dir = Path(args.save_obs_dir)
        save_obs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[obs] dumping frames to {save_obs_dir}")
    ws_src = "cli"
    if args.workspace_min is not None and args.workspace_max is not None:
        min_xyz = tuple(float(v) for v in args.workspace_min)
        max_xyz = tuple(float(v) for v in args.workspace_max)
    else:
        try:
            from x5_play_viz.eef_offset import get_workspace_xyz_limits, resolve_params_yaml

            lo, hi = get_workspace_xyz_limits()
            min_xyz = tuple(float(v) for v in lo.reshape(-1)[:3])
            max_xyz = tuple(float(v) for v in hi.reshape(-1)[:3])
            ws_src = f"yaml:{resolve_params_yaml()}"
            if args.workspace_min is not None:
                min_xyz = tuple(float(v) for v in args.workspace_min)
                ws_src = "cli-min+yaml-max"
            if args.workspace_max is not None:
                max_xyz = tuple(float(v) for v in args.workspace_max)
                ws_src = "yaml-min+cli-max" if args.workspace_min is None else "cli"
        except Exception as exc:
            min_xyz = (-0.3, 0.11, 0.0)
            max_xyz = (0.3, 0.55, 0.7)
            if args.workspace_min is not None:
                min_xyz = tuple(float(v) for v in args.workspace_min)
            if args.workspace_max is not None:
                max_xyz = tuple(float(v) for v in args.workspace_max)
            ws_src = f"fallback ({exc})"
    print(
        f"[workspace] SDK/base_link xyz clip min={list(min_xyz)} max={list(max_xyz)} "
        f"src={ws_src}",
        flush=True,
    )

    # Shared differential-IK knobs: YAML defaults; CLI overrides when passed.
    try:
        from x5_play_viz.eef_offset import get_diff_ik_control_params

        _ikp = get_diff_ik_control_params()
        _ikp_src = f"yaml:{_ikp.get('yaml_path', '')}"
    except Exception as exc:
        _ikp = {
            "ik_max_dq": 0.12,
            "max_joint_delta_rad": 0.08,
            "joint_ema": 0.55,
            "ik_damp": 0.02,
        }
        _ikp_src = f"fallback ({exc})"
    if args.max_joint_delta_rad is None:
        args.max_joint_delta_rad = float(_ikp["max_joint_delta_rad"])
    if args.eef_joint_ema is None:
        args.eef_joint_ema = float(_ikp["joint_ema"])
    if getattr(args, "ik_max_dq", None) is None:
        args.ik_max_dq = float(_ikp["ik_max_dq"])
    if getattr(args, "ik_damp", None) is None:
        args.ik_damp = float(_ikp["ik_damp"])
    print(
        f"[diff-ik] ik_max_dq={args.ik_max_dq} max_joint_delta_rad={args.max_joint_delta_rad} "
        f"joint_ema={args.eef_joint_ema} ik_damp={args.ik_damp} src={_ikp_src}",
        flush=True,
    )

    if args.action_space == "joint" and abs(args.control_hz - 30.0) > 1e-3:
        print(
            f"[warn] joint model usually trains at 30Hz; got control_hz={args.control_hz}"
        )
    if args.action_space in ("eef", "eef_via_joint") and abs(args.control_hz - 20.0) > 1e-3:
        print(
            f"[warn] EEF models for today are 20Hz (forgot to change in train); "
            f"got control_hz={args.control_hz}"
        )
    if args.action_space == "eef":
        print(
            "[warn] --action-space eef uses mode=4 SDK IK; "
            "real-robot singularity issues known. Prefer eef_via_joint or joint."
        )
    if args.action_space == "eef_via_joint":
        print(
            "[exec] eef_via_joint: policy EEF Δ → client DLS-IK → mode=5 joints "
            f"(soft_entry={args.eef_soft_entry_sec:.1f}s ema={args.eef_joint_ema:.2f} "
            f"max_dq={args.max_joint_delta_rad} fail_dq={args.eef_ik_fail_delta_rad}). "
            "Do NOT use mode=4 for EEF models."
        )
    if args.action_space in ("eef", "eef_via_joint"):
        print(
            f"[exec] frame_align={args.frame_align} "
            "(neg_x_z1: inverse x-flip + R→SRS on action Δ before arm cmd; "
            "none: raw model deltas)"
        )
        if args.frame_align == "none":
            print(
                "[exec] frame_align=none (raw ARX deltas). "
                "Only 1p / libero-aligned EEF data needs neg_x_z1."
            )

    state_modality = _resolve_state_modality(args.action_space, args.state_modality)
    if bool(getattr(args, "send_eef_state", False)):
        state_modality = "eef"
    use_eef8 = bool(getattr(args, "eef_state_8d", False)) and state_modality == "eef"
    eef_state_source = str(getattr(args, "eef_state_source", "sdk_offset") or "sdk_offset")
    state_dim = 8 if use_eef8 else (7 if state_modality == "eef" else 7)
    if state_modality == "joint":
        state_dim = 7
    state_history_len = max(1, int(getattr(args, "state_history_len", 1) or 1))
    state_lo: np.ndarray
    state_hi: np.ndarray
    loaded = None
    if args.state_stats_json:
        loaded = _load_state_minmax_from_json(
            args.state_stats_json,
            unnorm_key=str(args.unnorm_key),
            modality=state_modality,
            state_dim=state_dim,
        )
    if loaded is not None:
        state_lo, state_hi = loaded
    elif use_eef8:
        state_lo, state_hi = _EEF8_STATE_MIN.copy(), _EEF8_STATE_MAX.copy()
    elif state_modality == "eef":
        state_lo, state_hi = _EEF_STATE_MIN.copy(), _EEF_STATE_MAX.copy()
    else:
        state_lo, state_hi = _JOINT_STATE_MIN.copy(), _JOINT_STATE_MAX.copy()
    if args.send_state:
        layout = "eef8_rotvec" if use_eef8 else state_modality
        src_note = f" eef_source={eef_state_source}" if state_modality == "eef" else ""
        print(
            f"[state] modality={layout} dim={int(state_lo.shape[0])} "
            f"history_T={state_history_len} normalize={bool(args.normalize_state)}"
            f"{src_note} "
            f"(default=joint; EEF needs --send-eef-state; scratch 8D needs --eef-state-8d)"
        )
        if args.action_space in ("eef", "eef_via_joint") and state_modality != "eef":
            print(
                "[state] NOTE: action_space is EEF but state=joint. "
                "For 1p/2p train-aligned proprio, add --send-eef-state.",
                flush=True,
            )
        if use_eef8 and state_history_len < 16:
            print(
                f"[state] WARN: scratch train used T=16; you set T={state_history_len}. "
                "Prefer --state-history-len 16.",
                flush=True,
            )
    off_meta = _eef_offset_meta()
    print(
        "[eef-offset] eef_xyz_offset_m=%s yaml=%s "
        "(restart client after editing YAML; play-viz cache is process-local)"
        % (off_meta["eef_xyz_offset_m"], off_meta["eef_offset_yaml"]),
        flush=True,
    )

    rtc_mode = str(getattr(args, "rtc_mode", "off") or "off").strip().lower()
    print(
        f"Connecting policy server ws://{args.host}:{args.port} "
        f"unnorm_key={args.unnorm_key} action_space={args.action_space} "
        f"control_hz={args.control_hz} dry_run={args.dry_run} rtc_mode={rtc_mode}"
    )
    if rtc_mode == "off":
        client = ModelClient(
            unnorm_key=args.unnorm_key,
            host=args.host,
            port=args.port,
            policy_setup="franka",
            num_ddim_steps=int(args.num_ddim_steps),
        )
    else:
        # RTC track_b client lives under get-data RTC/ (PYTHONPATH or RTC_ROOT).
        rtc_root = os.environ.get("RTC_ROOT", "").strip()
        if not rtc_root:
            # sibling get-data checkout next to EVAL_ROOT, or env GETDATA_ROOT
            for cand in (
                os.environ.get("GETDATA_ROOT", ""),
                os.path.join(
                    os.path.dirname(os.path.abspath(args.starvla_root or "")),
                    "ARX_X5_clean_server46(7.12get-data)",
                ),
                os.path.expanduser(
                    "~/project/ARX_X5_clean_server46(7.12get-data)"
                ),
            ):
                if cand and os.path.isdir(os.path.join(cand, "RTC", "rtc_arx")):
                    rtc_root = os.path.join(os.path.abspath(cand), "RTC")
                    break
        if rtc_root and rtc_root not in sys.path:
            sys.path.insert(0, rtc_root)
        try:
            from rtc_arx import RTCDeltaEefClient
        except ImportError as e:
            raise SystemExit(
                "rtc-mode=track_b needs RTC/rtc_arx on PYTHONPATH "
                "(set GETDATA_ROOT or RTC_ROOT to …/ARX_X5_clean_server46(7.12get-data)). "
                f"Import error: {e}"
            ) from e
        client = RTCDeltaEefClient(
            host=args.host,
            port=args.port,
            unnorm_key=args.unnorm_key,
            rtc_mode="track_b",
            execution_horizon=int(args.rtc_execution_horizon),
            prefetch_remaining=int(args.rtc_prefetch_remaining),
            control_hz=float(args.control_hz),
            starvla_root=args.starvla_root,
        )
        print(
            f"[rtc] track_b on: exec_horizon={args.rtc_execution_horizon} "
            f"prefetch_at={args.rtc_prefetch_remaining} "
            f"(forces open-loop chunk consume; closed-loop would discard leftover)",
            flush=True,
        )
    client.reset(task_description=args.task)
    action_repeat = max(1, int(args.action_repeat))
    closed_loop = bool(args.closed_loop)
    if rtc_mode == "track_b":
        if closed_loop:
            print(
                "[rtc] forcing --no-closed-loop (RTC needs leftover queue)",
                flush=True,
            )
        closed_loop = False
        if action_repeat != 1:
            print(
                f"[rtc] forcing --action-repeat 1 (was {action_repeat})",
                flush=True,
            )
        action_repeat = 1
    if closed_loop:
        # Replan every action_repeat control cycles; only use first action of chunk.
        infer_period_s = action_repeat / max(args.control_hz, 0.5)
        print(
            f"[exec] CLOSED-LOOP on: replan every {action_repeat} steps "
            f"(~{infer_period_s:.2f}s) using action[0] only; "
            f"max_joint_delta_rad={args.max_joint_delta_rad} ddim={args.num_ddim_steps}"
        )
        print(
            "[exec] If arm still always goes one side with dual-cam+state OK, "
            "model likely mode-collapsed (mean trajectory) — try mid ckpt or tell 师姐."
        )
    else:
        print(
            f"[exec] open-loop chunk: action_repeat={action_repeat} "
            f"max_joint_delta_rad={args.max_joint_delta_rad} "
            f"infer every ~{8 * action_repeat / max(args.control_hz, 0.5):.2f}s"
        )
    t_infer_ms: list[float] = []
    t_loop_ms: list[float] = []
    pred_log = open(args.dump_pred_jsonl, "a", encoding="utf-8") if args.dump_pred_jsonl else None
    if pred_log:
        print(f"[exec] dumping predictions → {args.dump_pred_jsonl}")
        import json as _json

        pred_log.write(
            _json.dumps(
                {
                    "event": "run_meta",
                    "task": args.task,
                    "action_space": args.action_space,
                    "state_modality": state_modality,
                    "eef_state_8d": use_eef8,
                    "state_history_len": state_history_len,
                    "normalize_state": bool(args.normalize_state),
                    "frame_align": args.frame_align,
                    "closed_loop": closed_loop,
                    **off_meta,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        pred_log.flush()
    log_state_every = max(0, int(args.log_state_every))
    if log_state_every > 0:
        print(
            f"[state] console log every {log_state_every} infer step(s) "
            "(state_raw / state_sent / world_vector)",
            flush=True,
        )
    cached_target_q: Optional[np.ndarray] = None
    state_hist: list[np.ndarray] = []
    cached_gripper: Optional[float] = None
    # eef_via_joint continuity (seed IK from last command, not raw measure)
    last_cmd_q: Optional[np.ndarray] = None
    soft_entry_steps = max(
        0, int(round(float(args.eef_soft_entry_sec) * max(args.control_hz, 0.5)))
    )
    entry_q0: Optional[np.ndarray] = None

    # --- Rollout recorder (video + per-step traj.jsonl) ---
    recorder: Optional[_RolloutRecorder] = None
    if args.record_dir:
        if cv2 is None:
            print("[record][warn] cv2 unavailable; --record-dir ignored", flush=True)
        else:
            rec_fps = float(args.record_fps) if args.record_fps else float(args.control_hz)
            try:
                recorder = _RolloutRecorder(
                    args.record_dir,
                    fps=rec_fps,
                    meta={
                        "task": args.task,
                        "action_space": args.action_space,
                        "state_modality": state_modality,
                        "state_history_len": state_history_len,
                        "normalize_state": bool(args.normalize_state),
                        "state_gripper_mode": args.state_gripper_mode,
                        "gripper_mode": args.gripper_mode,
                        "gripper_qpos_open": args.gripper_qpos_open,
                        "gripper_qpos_closed": args.gripper_qpos_closed,
                        "frame_align": args.frame_align,
                        "control_hz": float(args.control_hz),
                        "closed_loop": closed_loop,
                        "rtc_mode": rtc_mode,
                        "rtc_execution_horizon": int(args.rtc_execution_horizon),
                        "camera_mode": args.camera_mode,
                        "dry_run": bool(args.dry_run),
                        "unnorm_key": args.unnorm_key,
                        **off_meta,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[record][warn] recorder init failed: {exc}", flush=True)
                recorder = None

    if args.camera_mode == "dual":
        image_source = build_record_cameras(
            wrist_serial=None,
            desktop_serial=args.desktop_serial,
            enable_desktop=True,
            width=args.d405_width,
            height=args.d405_height,
            fps=args.d405_fps,
            display=args.display,
        )
        for _ in range(10):
            frames = image_source.read_all_ready(timeout_ms=1000)
            if frames is not None:
                break
    else:
        image_source = RealSenseD405ImageSource(
            width=args.d405_width,
            height=args.d405_height,
            fps=args.d405_fps,
            display=args.display,
            align_depth=False,
        )
        for _ in range(10):
            if image_source.read(timeout_ms=1000) is not None:
                break

    rclpy.init()
    node = X5ArmBridge()

    try:
        node.wait_for_state()

        if args.go_home:
            home_rad = [float(np.deg2rad(v)) for v in args.go_home_deg]
            node.get_logger().info(
                "Startup go_home (teleop-style smooth mode=5) before policy loop"
            )
            _startup_go_home(
                node,
                home_joints6=home_rad,
                gripper_open=float(args.go_home_gripper),
                control_hz=float(args.go_home_hz),
                blend_speed_rad_s=float(args.go_home_blend_speed),
                max_blend_sec=float(args.go_home_blend_max_sec),
                lock_sec=float(args.go_home_lock_sec),
                dry_run=args.dry_run,
                settle_tol_rad=float(args.go_home_settle_tol),
                settle_timeout_sec=float(args.go_home_settle_timeout),
            )
        elif abs(args.init_joint0_ccw_deg) > 1e-6:
            _blend_joint0_ccw(
                node=node,
                joint_index=args.init_joint_index,
                delta_rad=np.deg2rad(args.init_joint0_ccw_deg),
                move_sec=args.init_move_sec,
                control_hz=args.init_control_hz,
                hold_sec=args.init_hold_sec,
                dry_run=args.dry_run,
            )
            _warmup_end_control_from_feedback(
                node=node,
                warmup_sec=args.init_settle_sec,
                control_hz=args.init_control_hz,
                dry_run=args.dry_run,
            )

        node.get_logger().info(
            "StarVLA->X5 loop started. action_space=%s pos_scale=%.3f rot_scale=%.3f hz=%.1f"
            % (args.action_space, args.pos_scale, args.rot_scale, args.control_hz)
        )

        # --- 100Hz interpolation publish thread (optional) ---
        _interp_target = None
        _interp_thread = None
        if args.publish_hz > 0:
            if InterpTarget is None or InterpPublishThread is None:
                print("[warn] --publish-hz>0 but interp_publish not importable; "
                      "falling back to direct publish", flush=True)
            else:
                _interp_target = InterpTarget()
                # Initialize with current joint position to avoid initial jump
                _init_q = np.asarray(
                    node.latest_state.joint_pos[:6] if node.latest_state is not None
                    else np.zeros(6),
                    dtype=np.float64,
                )
                _interp_target.update(_init_q, 0.0, np.zeros(6))
                _interp_thread = InterpPublishThread(
                    node,
                    _interp_target,
                    publish_hz=float(args.publish_hz),
                    interp_duration=1.0 / max(args.control_hz, 1e-3),
                    safety_rate_limit_rad=0.04,
                )
                _interp_thread.start()
                print(
                    f"[interp] enabled: publish_hz={args.publish_hz:.0f} "
                    f"interp_duration={1000.0/max(args.control_hz,1e-3):.0f}ms",
                    flush=True,
                )

        for step in range(args.max_steps):
            t0 = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)
            arm_state = node.latest_state
            if arm_state is None:
                time.sleep(sleep_sec)
                continue

            if args.camera_mode == "dual":
                frames = image_source.read_all_ready(timeout_ms=1000)
                if frames is None:
                    node.get_logger().warning(
                        "Camera returned incomplete frames; skipping step."
                    )
                    time.sleep(sleep_sec)
                    continue
                wrist_rgb = frames["cam_wrist"]
                agent_rgb = frames["cam_high"]
            else:
                wrist_rgb = image_source.read(timeout_ms=1000)
                agent_rgb = None
                if wrist_rgb is None:
                    node.get_logger().warning("D405 returned no frame; skipping step.")
                    time.sleep(sleep_sec)
                    continue

            q7 = np.asarray(arm_state.joint_pos[:7], dtype=np.float32)
            if q7.shape[0] < 7:
                # pad gripper from end_pos if needed
                g = float(arm_state.end_pos[6]) if arm_state.end_pos.shape[0] > 6 else 0.0
                q7 = np.concatenate([q7[:6], [g]]).astype(np.float32)

            # 1p/2p: offset-frame current_xyz (+ raw rpy, gripper) → 7-D state.
            # scratch 8D: [xyz, rotvec, +g, -g] with optional T=16 history.
            # --eef-state-source sdk_offset|eef_link_fk selects xyzrpy frame.
            if state_modality == "eef":
                if use_eef8:
                    state_vec = _sdk_eef_state8(
                        arm_state.end_pos,
                        q7,
                        eef_state_source=eef_state_source,
                        state_gripper_mode=args.state_gripper_mode,
                        gripper_qpos_open=args.gripper_qpos_open,
                        gripper_qpos_closed=args.gripper_qpos_closed,
                    )
                else:
                    state_vec = _sdk_eef_state7(
                        arm_state.end_pos,
                        q7,
                        eef_state_source=eef_state_source,
                    )
            else:
                state_vec = q7

            state_seq_raw: Optional[np.ndarray] = None
            if args.send_state and state_vec is not None:
                state_hist.append(np.asarray(state_vec, dtype=np.float32).reshape(-1))
                if len(state_hist) > state_history_len:
                    del state_hist[:-state_history_len]
                # Left-pad by repeating earliest frame until T full (train-aligned length).
                buf = list(state_hist)
                while len(buf) < state_history_len:
                    buf.insert(0, buf[0].copy())
                state_seq_raw = np.stack(buf, axis=0)

            if save_obs_dir is not None and step % max(1, args.save_obs_every) == 0:
                try:
                    from PIL import Image as _PILImage

                    _PILImage.fromarray(wrist_rgb).save(
                        save_obs_dir / f"step{step:04d}_cam_wrist.jpg"
                    )
                    if agent_rgb is not None:
                        _PILImage.fromarray(agent_rgb).save(
                            save_obs_dir / f"step{step:04d}_cam_high.jpg"
                        )
                except Exception as exc:
                    node.get_logger().warning("save-obs failed: %s" % exc)

            example = _build_example_dict(
                wrist_rgb=wrist_rgb,
                task=args.task,
                camera_mode=args.camera_mode,
                flip_180=args.flip_180,
                agent_rgb=agent_rgb,
                agent_rotate_ccw_deg=args.agent_rotate_ccw_deg,
                state_vec=None if state_seq_raw is not None else (
                    state_vec if args.send_state else None
                ),
                state_seq=state_seq_raw if args.send_state else None,
                normalize_state=bool(args.normalize_state),
                state_lo=state_lo,
                state_hi=state_hi,
            )
            # Closed-loop: re-infer every action_repeat steps, hold action[0] between.
            # Open-loop / RTC track_b: one client.step per control tick (RTC pops queue).
            t_infer = 0.0
            model_step = step // action_repeat
            inferred_this_step = False
            last_rtc_info: Optional[dict] = None
            response: Optional[dict] = None
            if closed_loop:
                do_infer = step % action_repeat == 0 or cached_target_q is None
                model_step = step // action_repeat  # replan index for logs
                if do_infer:
                    inferred_this_step = True
                    client.raw_actions = None
                    t_infer0 = time.perf_counter()
                    response = client.step(example=example, step=0)
                    t_infer = time.perf_counter() - t_infer0
                    raw_action = response["raw_action"]
                    world_delta = np.asarray(
                        raw_action["world_vector"], dtype=np.float32
                    ).reshape(3)
                    rot_delta = np.asarray(
                        raw_action["rotation_delta"], dtype=np.float32
                    ).reshape(3)
                    grip_raw = float(
                        np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(
                            -1
                        )[0]
                    )
                    cached_target_q = np.concatenate(
                        [world_delta, rot_delta]
                    ).astype(np.float64)
                    cached_gripper = grip_raw
                assert cached_target_q is not None and cached_gripper is not None
                target_q = cached_target_q
                grip_raw = cached_gripper
                world_delta = target_q[:3].astype(np.float32)
                rot_delta = target_q[3:6].astype(np.float32)
            else:
                inferred_this_step = True
                t_infer0 = time.perf_counter()
                response = client.step(example=example, step=model_step)
                t_infer = time.perf_counter() - t_infer0
                raw_action = response["raw_action"]
                world_delta = np.asarray(
                    raw_action["world_vector"], dtype=np.float32
                ).reshape(3)
                rot_delta = np.asarray(
                    raw_action["rotation_delta"], dtype=np.float32
                ).reshape(3)
                grip_raw = float(
                    np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1)[
                        0
                    ]
                )
                target_q = np.concatenate([world_delta, rot_delta]).astype(np.float64)
            if isinstance(response, dict):
                last_rtc_info = response.get("rtc")

            world_delta_pre_align = np.asarray(world_delta, dtype=np.float32).copy()
            rot_delta_pre_align = np.asarray(rot_delta, dtype=np.float32).copy()

            # LIBERO-like → ARX inverse (training used neg_x_z1 on EEF datasets)
            if args.action_space in ("eef", "eef_via_joint") and args.frame_align != "none":
                from lewm_x5_grasp.frame_align import action_deltas_libero_to_arx

                world_delta, rot_delta = action_deltas_libero_to_arx(
                    world_delta, rot_delta, mode=args.frame_align
                )

            # dxyz_xy_remap is applied after workspace clip (see eef_via_joint), so we can
            # log both pre- and post-remap clipped_dxyz.

            state_raw_list: Optional[list[float]] = None
            state_sent_list: Optional[list[float]] = None
            if args.send_state and state_vec is not None and (
                inferred_this_step
                and (
                    pred_log is not None
                    or (
                        log_state_every > 0
                        and (model_step % log_state_every == 0)
                    )
                )
            ):
                state_raw_list, state_sent_list = _state_raw_and_norm(
                    state_vec,
                    normalize=bool(args.normalize_state),
                    state_lo=state_lo,
                    state_hi=state_hi,
                )
                if log_state_every > 0 and model_step % log_state_every == 0 and inferred_this_step:
                    node.get_logger().info(
                        "STATE step=%d/%d modality=%s offset=%s "
                        "sdk_xyz=[%.4f, %.4f, %.4f] "
                        "state_raw_xyz=[%.4f, %.4f, %.4f] "
                        "state_sent_xyz=[%.4f, %.4f, %.4f] "
                        "world_vector=[%.5f, %.5f, %.5f]"
                        % (
                            step,
                            model_step,
                            state_modality,
                            off_meta["eef_xyz_offset_m"],
                            float(arm_state.end_pos[0]),
                            float(arm_state.end_pos[1]),
                            float(arm_state.end_pos[2]),
                            state_raw_list[0],
                            state_raw_list[1],
                            state_raw_list[2],
                            state_sent_list[0],
                            state_sent_list[1],
                            state_sent_list[2],
                            float(world_delta[0]),
                            float(world_delta[1]),
                            float(world_delta[2]),
                        )
                    )

            if args.action_space == "joint":
                # Absolute qpos target (6 joints); gripper separate
                cur_q = np.asarray(arm_state.joint_pos[:6], dtype=np.float64)
                joint_pos = _rate_limit_joints(
                    cur_q, target_q, float(args.max_joint_delta_rad)
                )
                _maybe_write_pred_jsonl(
                    pred_log,
                    closed_loop=closed_loop,
                    step=step,
                    action_repeat=action_repeat,
                    inferred=inferred_this_step,
                    row={
                        "event": "pred",
                        "task": args.task,
                        "step": step,
                        "model_step": model_step,
                        "action_space": "joint",
                        "state_modality": state_modality,
                        "t_infer_ms": round(t_infer * 1000.0, 1),
                        "state_raw": state_raw_list,
                        "state_sent": state_sent_list,
                        "cur_q": [float(x) for x in cur_q],
                        "target_q": [float(x) for x in target_q],
                        "cmd_q": [float(x) for x in joint_pos],
                        "gripper": float(grip_raw),
                        "closed_loop": closed_loop,
                        **off_meta,
                    },
                )
                gripper = (
                    _closedness_to_qpos(grip_raw, args.gripper_qpos_open, args.gripper_qpos_closed)
                    if args.gripper_mode == "continuous_openpi"
                    else float(grip_raw)
                )
                cur_eef = np.asarray(arm_state.end_pos[:6], dtype=np.float64)
                command = ArmCommand(
                    xyzrpy=[float(v) for v in cur_eef],
                    gripper=gripper,
                    joint_pos=[float(v) for v in joint_pos[:6]],
                    mode=MODE_POSITION_CONTROL,
                )
                node.get_logger().info(
                    "step=%d/%d mode=5 qpos=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] "
                    "tgt=[%.3f, %.3f, %.3f] gripper=%.3f T_infer=%.1fms"
                    % (
                        step,
                        model_step,
                        joint_pos[0],
                        joint_pos[1],
                        joint_pos[2],
                        joint_pos[3],
                        joint_pos[4],
                        joint_pos[5],
                        target_q[0],
                        target_q[1],
                        target_q[2],
                        gripper,
                        t_infer * 1000.0,
                    )
                )
            elif args.action_space == "eef_via_joint":
                from lewm_x5_grasp.eef_to_joint_ik import eef_delta_to_joint_cmd

                if args.gripper_mode == "continuous_openpi":
                    gripper = _closedness_to_qpos(
                        grip_raw, args.gripper_qpos_open, args.gripper_qpos_closed
                    )
                else:
                    gripper_sign = _binarize_gripper_open(
                        np.asarray([grip_raw], dtype=np.float32)
                    )
                    gripper = _map_gripper(
                        gripper_sign, args.gripper_open, args.gripper_close
                    )
                # Raw SDK flange pose (workspace limits are defined in this frame).
                cur_eef_sdk = np.asarray(arm_state.end_pos[:6], dtype=np.float64)
                # Integration base matches --eef-state-source (train-aligned tip vs legacy offset).
                if eef_state_source in ("eef_link_fk", "fk_tip", "tip", "eef_link"):
                    cur_eef = _eef_xyzrpy_for_state(
                        arm_state.end_pos,
                        arm_state.joint_pos,
                        source="eef_link_fk",
                    )
                else:
                    # Offset-frame current: xyz += eef_xyz_offset_m (YAML).
                    cur_eef = _eef_with_xyz_offset(cur_eef_sdk)
                cur_q = np.asarray(arm_state.joint_pos[:6], dtype=np.float64)
                if last_cmd_q is None:
                    last_cmd_q = cur_q.copy()
                if entry_q0 is None:
                    entry_q0 = cur_q.copy()
                # Integrate Δ in proprio frame, then clip xyz back in SDK/flange frame.
                tentative = cur_eef.copy()
                tentative[:3] = cur_eef[:3] + world_delta * args.pos_scale
                # Map to flange xyz for clip (min_xyz/max_xyz stay in SDK coordinates).
                tentative_flange = tentative[:3] - (cur_eef[:3] - cur_eef_sdk[:3])
                tentative_flange = _clip_xyz(tentative_flange, min_xyz, max_xyz)
                tentative[:3] = tentative_flange + (cur_eef[:3] - cur_eef_sdk[:3])
                clipped_delta_pre = (tentative[:3] - cur_eef[:3]) / max(args.pos_scale, 1e-8)
                # YAML dxyz_xy_remap on clipped Δxyz: (x,y,z)->(-y,x,z). IK + post log use remapped.
                clipped_delta = _apply_dxyz_xy_remap(clipped_delta_pre)
                # Differential IK: Δxyzrpy→Δq at last_cmd_q. cur_eef here is offset-frame
                # (only builds target_eef absolute; Jacobian uses joints / URDF).
                # world_delta here is post-remap clipped Δ (pre-scale units).
                # rotvec semantics (ljn stage2): model rot_delta is a world-frame
                # rotation-vector increment; convert to the euler increment the
                # Jacobian understands (element-wise euler add is wrong at yaw≈90°).
                rot_delta_ik = rot_delta
                rot_scale_ik = float(args.rot_scale)
                if args.rot_delta_semantics == "rotvec":
                    from lewm_x5_grasp.eef_to_joint_ik import (
                        rotvec_delta_to_euler_increment,
                    )

                    rot_delta_ik = rotvec_delta_to_euler_increment(
                        rot_delta, last_cmd_q, rot_scale=args.rot_scale
                    )
                    rot_scale_ik = 1.0  # scale already applied in the conversion
                _ik_fk_fn = None
                if args.ik_fk == "eef_link":
                    from lewm_x5_grasp.x5_fk import eef_link_fk as _ik_fk_fn
                q_ik, target_eef, ik_info = eef_delta_to_joint_cmd(
                    cur_eef=cur_eef,
                    cur_q=last_cmd_q,
                    world_delta=clipped_delta,
                    rot_delta=rot_delta_ik,
                    pos_scale=args.pos_scale,
                    rot_scale=rot_scale_ik,
                    seed_q=last_cmd_q,
                    use_differential=True,
                    damp=float(args.ik_damp),
                    max_dq=float(args.ik_max_dq),
                    fk_fn=_ik_fk_fn,
                )
                ik_ok = bool(ik_info.get("ok", False))
                jump = float(ik_info.get("jump", 0.0))
                max_dq = float(args.max_joint_delta_rad)
                if not ik_ok:
                    max_dq = min(max_dq, float(args.eef_ik_fail_delta_rad))
                q_limited = _rate_limit_joints(last_cmd_q, q_ik, max_dq)
                q6 = _ema_joints(last_cmd_q, q_limited, float(args.eef_joint_ema))
                # Soft entry: blend from post-go_home pose into policy IK
                if soft_entry_steps > 0 and step < soft_entry_steps and entry_q0 is not None:
                    alpha = _smoothstep((step + 1) / soft_entry_steps)
                    q6 = _ema_joints(entry_q0, q6, alpha)
                last_cmd_q = q6.copy()
                command = ArmCommand(
                    xyzrpy=[float(v) for v in target_eef],
                    gripper=float(gripper),
                    joint_pos=[float(v) for v in q6],
                    mode=MODE_POSITION_CONTROL,
                )
                # current_xyz / target_xyz both in offset frame (current = SDK+offset).
                current_xyz = cur_eef[:3]
                # Physical clipped Δxyz: pre-remap (after workspace) vs post-remap (fed to IK).
                clipped_dxyz_pre = np.asarray(clipped_delta_pre, dtype=np.float64).reshape(
                    3
                ) * float(args.pos_scale)
                clipped_dxyz = np.asarray(clipped_delta, dtype=np.float64).reshape(3) * float(
                    args.pos_scale
                )
                clipped_d_rpy = np.asarray(rot_delta, dtype=np.float64).reshape(3) * float(
                    args.rot_scale
                )
                g_fb = (
                    float(arm_state.joint_pos[6])
                    if np.asarray(arm_state.joint_pos).reshape(-1).shape[0] > 6
                    else float("nan")
                )
                node.get_logger().info(
                    "step=%d/%d mode=5(eef→Δik) current_xyz=[%.3f, %.3f, %.3f] "
                    "target_xyz=[%.3f, %.3f, %.3f] "
                    "clipped_dxyz_pre=[%.4f, %.4f, %.4f] "
                    "clipped_dxyz=[%.4f, %.4f, %.4f] "
                    "clipped_drpy=[%.4f, %.4f, %.4f] "
                    "qpos=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f] ik_ok=%s "
                    "dq=%.3f T_infer=%.1fms | "
                    "grip_raw=%.3f open=%s g_cmd=%.2f g_fb=%.3f | "
                    "rtc={mode:%s,q:%s,delay:%s,srv:%s}"
                    % (
                        step,
                        model_step,
                        current_xyz[0],
                        current_xyz[1],
                        current_xyz[2],
                        target_eef[0],
                        target_eef[1],
                        target_eef[2],
                        clipped_dxyz_pre[0],
                        clipped_dxyz_pre[1],
                        clipped_dxyz_pre[2],
                        clipped_dxyz[0],
                        clipped_dxyz[1],
                        clipped_dxyz[2],
                        clipped_d_rpy[0],
                        clipped_d_rpy[1],
                        clipped_d_rpy[2],
                        q6[0],
                        q6[1],
                        q6[2],
                        q6[3],
                        q6[4],
                        q6[5],
                        ik_ok,
                        jump,
                        t_infer * 1000.0,
                        float(grip_raw),
                        bool(float(grip_raw) > 0.5),
                        float(gripper),
                        g_fb,
                        (last_rtc_info or {}).get("mode", rtc_mode),
                        (last_rtc_info or {}).get("queue_remaining"),
                        (last_rtc_info or {}).get("last_delay"),
                        ((last_rtc_info or {}).get("server") or {}).get("applied"),
                    )
                )
                _maybe_write_pred_jsonl(
                    pred_log,
                    closed_loop=closed_loop,
                    step=step,
                    action_repeat=action_repeat,
                    inferred=inferred_this_step,
                    row={
                        "event": "pred",
                        "task": args.task,
                        "step": step,
                        "model_step": model_step,
                        "action_space": "eef_via_joint",
                        "state_modality": state_modality,
                        "t_infer_ms": round(t_infer * 1000.0, 1),
                        "sdk_xyz": [
                            float(arm_state.end_pos[0]),
                            float(arm_state.end_pos[1]),
                            float(arm_state.end_pos[2]),
                        ],
                        "state_raw": state_raw_list,
                        "state_sent": state_sent_list,
                        "world_vector_pre_align": [
                            float(x) for x in world_delta_pre_align
                        ],
                        "world_vector": [float(x) for x in world_delta],
                        "rotation_delta_pre_align": [
                            float(x) for x in rot_delta_pre_align
                        ],
                        "rotation_delta": [float(x) for x in rot_delta],
                        "clipped_dxyz_pre": [float(x) for x in clipped_dxyz_pre],
                        "clipped_dxyz": [float(x) for x in clipped_dxyz],
                        "current_xyz": [float(x) for x in current_xyz],
                        "target_xyz": [float(x) for x in target_eef[:3]],
                        "ik_ok": ik_ok,
                        "gripper": float(grip_raw),
                        "closed_loop": closed_loop,
                        **off_meta,
                    },
                )
            else:
                cur = np.asarray(arm_state.end_pos[:6], dtype=np.float64)
                target = cur.copy()
                target[:3] = cur[:3] + world_delta * args.pos_scale
                target[3:6] = cur[3:6] + rot_delta * args.rot_scale
                target[:3] = _clip_xyz(target[:3], min_xyz, max_xyz)
                if args.gripper_mode == "continuous_openpi":
                    gripper = _closedness_to_qpos(
                        grip_raw, args.gripper_qpos_open, args.gripper_qpos_closed
                    )
                else:
                    gripper_sign = _binarize_gripper_open(
                        np.asarray([grip_raw], dtype=np.float32)
                    )
                    gripper = _map_gripper(
                        gripper_sign, args.gripper_open, args.gripper_close
                    )
                command = ArmCommand(
                    xyzrpy=[float(v) for v in target],
                    gripper=float(gripper),
                    mode=MODE_END_CONTROL,
                )
                node.get_logger().info(
                    "step=%d mode=4 delta_xyz=[%.4f, %.4f, %.4f] target_xyz=[%.3f, %.3f, %.3f] "
                    "gripper=%.2f T_infer=%.1fms"
                    % (
                        step,
                        world_delta[0],
                        world_delta[1],
                        world_delta[2],
                        target[0],
                        target[1],
                        target[2],
                        gripper,
                        t_infer * 1000.0,
                    )
                )
                _maybe_write_pred_jsonl(
                    pred_log,
                    closed_loop=closed_loop,
                    step=step,
                    action_repeat=action_repeat,
                    inferred=inferred_this_step,
                    row={
                        "event": "pred",
                        "task": args.task,
                        "step": step,
                        "model_step": model_step,
                        "action_space": "eef",
                        "state_modality": state_modality,
                        "t_infer_ms": round(t_infer * 1000.0, 1),
                        "state_raw": state_raw_list,
                        "state_sent": state_sent_list,
                        "world_vector": [float(x) for x in world_delta],
                        "rotation_delta": [float(x) for x in rot_delta],
                        "target_xyz": [float(x) for x in target[:3]],
                        "gripper": float(grip_raw),
                        "closed_loop": closed_loop,
                        **off_meta,
                    },
                )

            if not args.dry_run:
                if _interp_target is not None and _interp_thread is not None:
                    # Interpolation publish path: update shared target, thread publishes at 100Hz
                    _jpos = getattr(command, "joint_pos", None)
                    if _jpos is not None:
                        _interp_target.update(
                            q6=_jpos,
                            gripper=float(command.gripper),
                            eef=command.xyzrpy if command.xyzrpy is not None else np.zeros(6),
                        )
                    else:
                        # mode=4 command without joint_pos: direct publish as fallback
                        node.publish_command(command)
                else:
                    # Original direct publish path (unchanged)
                    node.publish_command(command)

            if _play_viz_enabled(args):
                q_for_fk = getattr(command, "joint_pos", None)
                if q_for_fk is None:
                    q_for_fk = arm_state.joint_pos[:6]
                meta = {
                    "step": int(step),
                    "model_step": int(model_step),
                    "action_space": args.action_space,
                    "frame_align": args.frame_align,
                    "t_infer_ms": round(t_infer * 1000.0, 1),
                    "source": "eval",
                }
                if args.action_space == "eef_via_joint":
                    meta["ik_ok"] = bool(locals().get("ik_ok", False))
                    te = locals().get("target_eef")
                    if te is not None:
                        meta["target_xyz"] = [float(x) for x in np.asarray(te)[:3]]
                    # Offset-frame current/target (same as logs). Bus end_pos stays raw SDK;
                    # MuJoCo --show-sdk applies YAML offset at draw time.
                    meta["current_xyz"] = [
                        float(x) for x in _current_xyz_from_sdk(arm_state.end_pos[:3])
                    ]
                _maybe_publish_play_viz(
                    q_cmd=q_for_fk,
                    joint_pos=arm_state.joint_pos[:6],
                    # Raw SDK end_pos on the bus; viz applies EEF_XYZ_OFFSET when drawing SDK axes.
                    end_pos=arm_state.end_pos[:6],
                    cam_high=agent_rgb,
                    cam_wrist=wrist_rgb,
                    meta=meta,
                )

            # --- Rollout record: video frames + per-step traj row ---
            if recorder is not None:
                try:
                    recorder.write("cam_wrist", wrist_rgb)
                    if agent_rgb is not None:
                        recorder.write("cam_high", agent_rgb)
                    _ep = np.asarray(arm_state.end_pos, dtype=np.float64).reshape(-1)
                    _jp = np.asarray(arm_state.joint_pos, dtype=np.float64).reshape(-1)
                    recorder.log(
                        {
                            "step": int(step),
                            "model_step": int(model_step),
                            "wall_time": round(time.time(), 4),
                            "inferred": bool(inferred_this_step),
                            "t_infer_ms": round(t_infer * 1000.0, 2),
                            "joint_pos": [float(v) for v in _jp[:7]],
                            "eef_pos6": [float(v) for v in _ep[:6]],
                            "state_vec": (
                                [float(v) for v in np.asarray(state_vec).reshape(-1)]
                                if state_vec is not None
                                else None
                            ),
                            "world_delta": [float(v) for v in np.asarray(world_delta).reshape(-1)],
                            "rot_delta": [float(v) for v in np.asarray(rot_delta).reshape(-1)],
                            "grip_raw": float(grip_raw),
                            "grip_cmd": float(gripper) if "gripper" in locals() else None,
                            "ik_ok": bool(locals().get("ik_ok", False))
                            if args.action_space == "eef_via_joint"
                            else None,
                            "rtc": last_rtc_info,
                            "dry_run": bool(args.dry_run),
                        }
                    )
                except Exception as _rec_exc:  # noqa: BLE001
                    node.get_logger().warning(f"record step failed: {_rec_exc}")

            t_loop = time.perf_counter() - t0
            t_infer_ms.append(t_infer * 1000.0)
            t_loop_ms.append(t_loop * 1000.0)
            if t_loop < sleep_sec:
                time.sleep(sleep_sec - t_loop)
    finally:
        if args.latency_log and t_infer_ms:
            arr_i = np.sort(np.asarray(t_infer_ms))
            arr_l = np.sort(np.asarray(t_loop_ms))
            budget = 1000.0 / max(args.control_hz, 0.5)
            hit = float(np.mean(arr_l <= budget + 1e-6))
            print(
                "[latency] n=%d T_infer_ms mean/p50/p95=%.1f/%.1f/%.1f | "
                "T_loop_ms mean/p50/p95=%.1f/%.1f/%.1f | budget=%.1fms hit_rate=%.2f"
                % (
                    len(arr_i),
                    float(arr_i.mean()),
                    float(arr_i[len(arr_i) // 2]),
                    float(arr_i[min(len(arr_i) - 1, int(len(arr_i) * 0.95))]),
                    float(arr_l.mean()),
                    float(arr_l[len(arr_l) // 2]),
                    float(arr_l[min(len(arr_l) - 1, int(len(arr_l) * 0.95))]),
                    budget,
                    hit,
                )
            )
        if recorder is not None:
            recorder.close()
        if pred_log is not None:
            pred_log.close()
        image_source.close()
        if hasattr(client, "close"):
            client.close()
        elif hasattr(client, "client"):
            client.client.close()
        # --- Stop interpolation publish thread ---
        if _interp_thread is not None:
            _interp_thread.stop()
            _interp_thread.join(timeout=2.0)
            print("[interp] publish thread stopped", flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
