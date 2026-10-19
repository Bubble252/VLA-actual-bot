"""Upper-layer safety for mode=4 (end-effector / SDK IK) commands.

The ARX SDK already uses KDL LMA damped IK, but wrist singularities can still
cause gripper spin. This module adds orientation limits, per-step clipping,
and joint-velocity spike holds before commands reach /arm_cmd.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .schemas import ArmCommand, ArmState


def _shortest_delta(target: float, current: float) -> float:
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


def _clip_step(current: float, target: float, max_step: float) -> float:
    delta = _shortest_delta(target, current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current + np.sign(delta) * max_step)


@dataclass
class Mode4GuardConfig:
    enabled: bool = True
    max_pitch_rad: float = 1.35
    min_pitch_rad: float = -1.35
    max_roll_rad: float = 3.2
    min_roll_rad: float = -3.2
    max_yaw_rad: float = 3.2
    min_yaw_rad: float = -3.2
    max_rpy_step_rad: float = 0.12
    max_xyz_step_m: float = 0.04
    max_joint_vel_rad_s: float = 2.5
    hold_on_joint_vel_spike: bool = True
    # After a spike, keep holding for this long so IK cannot chatter every frame.
    hold_sec: float = 0.2
    workspace_min_xyz: Optional[List[float]] = None
    workspace_max_xyz: Optional[List[float]] = None
    warn: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Mode4GuardConfig:
        if not data:
            return cls()
        cfg = cls()
        for key in (
            "enabled",
            "max_pitch_rad",
            "min_pitch_rad",
            "max_roll_rad",
            "min_roll_rad",
            "max_yaw_rad",
            "min_yaw_rad",
            "max_rpy_step_rad",
            "max_xyz_step_m",
            "max_joint_vel_rad_s",
            "hold_on_joint_vel_spike",
            "hold_sec",
            "warn",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        if "workspace_min_xyz" in data:
            cfg.workspace_min_xyz = [float(v) for v in data["workspace_min_xyz"]]
        if "workspace_max_xyz" in data:
            cfg.workspace_max_xyz = [float(v) for v in data["workspace_max_xyz"]]
        return cfg


@dataclass
class Mode4GuardResult:
    command: ArmCommand
    modified: bool = False
    held: bool = False
    reasons: List[str] = field(default_factory=list)


class Mode4Guard:
    def __init__(self, config: Optional[Mode4GuardConfig] = None) -> None:
        self.config = config or Mode4GuardConfig()
        self._warn_count = 0
        self._hold_until_mono = 0.0

    def update_config(self, data: Optional[Dict[str, Any]]) -> None:
        self.config = Mode4GuardConfig.from_dict(data)
        self._hold_until_mono = 0.0

    def apply(
        self,
        command: ArmCommand,
        state: Optional[ArmState],
    ) -> Mode4GuardResult:
        if not self.config.enabled or int(command.mode) != 4:
            return Mode4GuardResult(command=command)

        reasons: List[str] = []
        held = False
        now = time.monotonic()

        if state is not None and self.config.hold_on_joint_vel_spike:
            max_vel = float(np.max(np.abs(state.joint_vel[:6])))
            if max_vel > self.config.max_joint_vel_rad_s:
                held = True
                hold_sec = max(0.0, float(self.config.hold_sec))
                if hold_sec > 0.0:
                    self._hold_until_mono = max(self._hold_until_mono, now + hold_sec)
                reasons.append(f"joint_vel_spike({max_vel:.2f})")
            elif now < self._hold_until_mono:
                held = True
                reasons.append("hold_cooldown")

        current = np.asarray(state.end_pos[:6], dtype=float) if state is not None else None
        target = np.asarray(command.xyzrpy[:6], dtype=float).copy()

        if held and current is not None:
            safe_xyzrpy = current.tolist()
        else:
            safe_xyzrpy = self._sanitize_target(target, current, reasons)

        modified = (
            held
            or current is None
            or float(np.max(np.abs(np.asarray(safe_xyzrpy) - current))) > 1e-6
        )

        safe = ArmCommand(
            xyzrpy=[float(v) for v in safe_xyzrpy],
            gripper=float(command.gripper),
            mode=int(command.mode),
            joint_pos=command.joint_pos,
        )
        return Mode4GuardResult(command=safe, modified=modified, held=held, reasons=reasons)

    def _sanitize_target(
        self,
        target: np.ndarray,
        current: Optional[np.ndarray],
        reasons: List[str],
    ) -> List[float]:
        out = target.copy()

        if self.config.workspace_min_xyz is not None and self.config.workspace_max_xyz is not None:
            min_xyz = np.asarray(self.config.workspace_min_xyz, dtype=float)
            max_xyz = np.asarray(self.config.workspace_max_xyz, dtype=float)
            clipped = np.clip(out[:3], min_xyz, max_xyz)
            if not np.allclose(clipped, out[:3]):
                reasons.append("workspace_xyz")
            out[:3] = clipped

        rpy_bounds = (
            (self.config.min_roll_rad, self.config.max_roll_rad),
            (self.config.min_pitch_rad, self.config.max_pitch_rad),
            (self.config.min_yaw_rad, self.config.max_yaw_rad),
        )
        for idx, (lo, hi) in enumerate(rpy_bounds, start=3):
            clipped = float(np.clip(out[idx], lo, hi))
            if abs(clipped - out[idx]) > 1e-6:
                reasons.append(f"rpy_limit_{idx - 3}")
            out[idx] = clipped

        if current is not None:
            for i in range(3):
                stepped = _clip_step(float(current[i]), float(out[i]), self.config.max_xyz_step_m)
                if abs(stepped - out[i]) > 1e-6:
                    reasons.append("xyz_step")
                out[i] = stepped
            for i in range(3, 6):
                stepped = _clip_step(float(current[i]), float(out[i]), self.config.max_rpy_step_rad)
                if abs(stepped - out[i]) > 1e-6:
                    reasons.append("rpy_step")
                out[i] = stepped

        return out.tolist()

    def maybe_log(self, node: Any, result: Mode4GuardResult) -> None:
        if not self.config.warn or not result.modified:
            return
        self._warn_count += 1
        if self._warn_count > 20:
            return
        reason = ",".join(result.reasons) if result.reasons else "clipped"
        node.get_logger().warn(f"[mode4_guard] {'HOLD' if result.held else 'clip'}: {reason}")
