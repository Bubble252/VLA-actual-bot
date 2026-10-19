"""Map U-Arm master angles to ARX X5 follower joint commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class UArmMappingConfig:
    dof: int = 6
    # follower joint j reads master servo master_joint_indices[j] (Config2 wiring may differ)
    master_joint_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    scale: List[float] = field(default_factory=lambda: [1.0] * 6)
    offset_rad: List[float] = field(default_factory=lambda: [0.0] * 6)
    swap_joints_45: bool = True
    invert_joints: List[int] = field(default_factory=list)
    gripper_index: int = 6
    gripper_min_deg: float = 0.0
    gripper_max_deg: float = 30.0
    gripper_open_cmd: float = 2.0
    gripper_close_cmd: float = 0.0
    gripper_press_deg: float = 8.0
    gripper_mode: str = "relative"  # relative | press | linear (see README)
    gripper_min_cmd: float = 0.0
    gripper_max_cmd: float = 1.0
    gripper_zero_at_boot: bool = True
    gripper_invert: bool = False
    initial_gripper_cmd: float = 0.0
    gripper_open_hold_sec: float = 2.5
    max_joint_vel: List[float] = field(default_factory=lambda: [1.2, 1.2, 1.2, 2.0, 2.0, 2.0])
    max_gripper_vel: float = 2.0
    # Live teleop: ALWAYS off by default — must not alter recorded trajectories.
    # Singularity mitigation is replay/post-process only (mode4_guard, seed blend).
    singularity_avoidance: bool = False
    singularity_avoidance_gain: float = 0.12  # only used when avoidance enabled (soft bias)
    elbow_joint_index: int = 2
    wrist_joint_index: int = 4
    elbow_min_abs_rad: float = 0.12  # ~7°
    wrist_min_abs_rad: float = 0.12
    forearm_vertical_min_err_rad: float = 0.0  # disabled: was a major stutter source
    # Warn if master servo5 stuck; do NOT auto-flip swap (wrong mapping feels like lag).
    auto_disable_swap_if_master5_stuck: bool = False
    master_stuck_std_deg: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> UArmMappingConfig:
        cfg = cls()
        for key in (
            "dof",
            "master_joint_indices",
            "scale",
            "offset_rad",
            "swap_joints_45",
            "invert_joints",
            "gripper_index",
            "gripper_min_deg",
            "gripper_max_deg",
            "gripper_open_cmd",
            "gripper_close_cmd",
            "gripper_press_deg",
            "gripper_mode",
            "gripper_min_cmd",
            "gripper_max_cmd",
            "gripper_zero_at_boot",
            "gripper_invert",
            "initial_gripper_cmd",
            "gripper_open_hold_sec",
            "max_joint_vel",
            "max_gripper_vel",
            "singularity_avoidance",
            "singularity_avoidance_gain",
            "elbow_joint_index",
            "wrist_joint_index",
            "elbow_min_abs_rad",
            "wrist_min_abs_rad",
            "forearm_vertical_min_err_rad",
            "auto_disable_swap_if_master5_stuck",
            "master_stuck_std_deg",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        if "gripper_open_cmd" not in data and "gripper_min_cmd" in data:
            cfg.gripper_open_cmd = float(data["gripper_min_cmd"])
        if "gripper_close_cmd" not in data and "gripper_max_cmd" in data:
            cfg.gripper_close_cmd = float(data["gripper_max_cmd"])
        if "initial_gripper_cmd" not in data:
            cfg.initial_gripper_cmd = cfg.gripper_open_cmd
        # Allow *_deg aliases for singularity thresholds.
        if "elbow_min_abs_deg" in data and "elbow_min_abs_rad" not in data:
            cfg.elbow_min_abs_rad = float(np.deg2rad(float(data["elbow_min_abs_deg"])))
        if "wrist_min_abs_deg" in data and "wrist_min_abs_rad" not in data:
            cfg.wrist_min_abs_rad = float(np.deg2rad(float(data["wrist_min_abs_deg"])))
        if "forearm_vertical_min_err_deg" in data and "forearm_vertical_min_err_rad" not in data:
            cfg.forearm_vertical_min_err_rad = float(
                np.deg2rad(float(data["forearm_vertical_min_err_deg"]))
            )
        return cfg


@dataclass
class TeleopCommand:
    joint_pos_rad: np.ndarray
    gripper: float
    master_deg: np.ndarray


class UArmToX5Mapper:
    def __init__(self, config: UArmMappingConfig, control_dt: float = 0.02) -> None:
        self.config = config
        self.control_dt = control_dt
        self._inited = False
        self._last_joint = np.zeros(config.dof, dtype=np.float64)
        self._last_gripper = float(config.initial_gripper_cmd)
        self._gripper_rest_deg = 0.0
        self._runtime_offset_rad = np.zeros(config.dof, dtype=np.float64)

    def set_gripper_rest_deg(self, deg: float) -> None:
        """Treat current master gripper angle as open (elastic clip rest pose)."""
        self._gripper_rest_deg = float(deg)

    def reset_gripper_state(self, cmd: Optional[float] = None) -> None:
        """Sync internal gripper filter after startup open-hold."""
        self._last_gripper = float(
            cmd if cmd is not None else self.config.gripper_open_cmd
        )

    def _gripper_delta_deg(self, master: np.ndarray) -> float:
        grip_deg = float(master[self.config.gripper_index])
        if self.config.gripper_zero_at_boot:
            grip_deg -= self._gripper_rest_deg
        if self.config.gripper_invert:
            grip_deg = -grip_deg
        return grip_deg

    def _gripper_press_amount_deg(self, master: np.ndarray) -> float:
        return abs(self._gripper_delta_deg(master))

    def _map_gripper_cmd(self, master: np.ndarray) -> float:
        open_cmd = float(self.config.gripper_open_cmd)
        close_cmd = float(self.config.gripper_close_cmd)
        mode = str(self.config.gripper_mode).lower()

        if mode == "linear":
            grip_deg = float(master[self.config.gripper_index])
            if self.config.gripper_invert:
                grip_deg = -grip_deg
            span = max(1e-6, self.config.gripper_max_deg - self.config.gripper_min_deg)
            t = float(np.clip((grip_deg - self.config.gripper_min_deg) / span, 0.0, 1.0))
            return open_cmd + t * (close_cmd - open_cmd)

        delta = self._gripper_delta_deg(master)
        press_amount = abs(delta)

        if mode == "relative":
            press_deg = float(self.config.gripper_press_deg)
            if press_amount <= press_deg:
                return open_cmd
            span = max(1e-6, self.config.gripper_max_deg - press_deg)
            t = float(np.clip((press_amount - press_deg) / span, 0.0, 1.0))
            return open_cmd + t * (close_cmd - open_cmd)

        # press: dead zone then map remaining travel to close
        press_deg = float(self.config.gripper_press_deg)
        if press_amount <= press_deg:
            return open_cmd

        span = max(1e-6, self.config.gripper_max_deg - press_deg)
        t = float(np.clip((press_amount - press_deg) / span, 0.0, 1.0))
        return open_cmd + t * (close_cmd - open_cmd)

    def _mapped_joint_from_master(
        self,
        master: np.ndarray,
        *,
        include_runtime_offset: bool = True,
    ) -> np.ndarray:
        desired_joint = np.zeros(self.config.dof, dtype=np.float64)
        for j in range(self.config.dof):
            master_idx = self.config.master_joint_indices[j]
            offset = float(self.config.offset_rad[j])
            if include_runtime_offset:
                offset += float(self._runtime_offset_rad[j])
            desired_joint[j] = np.deg2rad(master[master_idx]) * self.config.scale[j] + offset

        if self.config.swap_joints_45 and self.config.dof >= 6:
            desired_joint[4], desired_joint[5] = -desired_joint[5], desired_joint[4]

        for idx in self.config.invert_joints:
            if 0 <= idx < self.config.dof:
                desired_joint[idx] = -desired_joint[idx]
        return desired_joint

    def _soft_push_abs_min(self, q: np.ndarray, index: int, min_abs: float, gain: float) -> None:
        """Gently bias |q[i]| toward min_abs (no hard snap — avoids teleop fight)."""
        if min_abs <= 0.0 or gain <= 0.0 or not (0 <= index < len(q)):
            return
        val = float(q[index])
        if abs(val) >= min_abs:
            return
        hint = float(self._last_joint[index]) if self._inited else 0.0
        sign = 1.0
        if abs(val) > 1e-9:
            sign = 1.0 if val >= 0.0 else -1.0
        elif abs(hint) > 1e-6:
            sign = 1.0 if hint >= 0.0 else -1.0
        target = sign * min_abs
        q[index] = val + float(gain) * (target - val)

    def _avoid_singularities(self, joint: np.ndarray) -> np.ndarray:
        """Optional soft bias away from elbow/wrist singularities (live teleop)."""
        if not self.config.singularity_avoidance:
            return joint
        q = np.asarray(joint, dtype=np.float64).copy()
        gain = float(np.clip(self.config.singularity_avoidance_gain, 0.0, 1.0))
        self._soft_push_abs_min(
            q, int(self.config.elbow_joint_index), float(self.config.elbow_min_abs_rad), gain
        )
        self._soft_push_abs_min(
            q, int(self.config.wrist_joint_index), float(self.config.wrist_min_abs_rad), gain
        )

        min_err = float(self.config.forearm_vertical_min_err_rad)
        if min_err > 0.0 and self.config.dof >= 3 and gain > 0.0:
            j1 = float(q[1])
            j2 = float(q[2])
            j23 = j1 + j2
            for target in (0.5 * np.pi, -0.5 * np.pi):
                err = j23 - target
                if abs(err) < min_err:
                    # Soft nudge only (fraction of the deficit) — hard push caused stutter.
                    push = gain * (min_err - abs(err)) * (1.0 if err >= 0.0 else -1.0)
                    q[1] = j1 + push
                    break
        return q

    def maybe_fix_swap_from_master_std(self, master_std_deg: Sequence[float]) -> bool:
        """Disable swap_joints_45 when master servo5 is stuck but servo4 moves.

        With Config2 swap, follower wrist j4 <- -master5. A dead master5 parks j4 at ~0
        (wrist singularity) for the whole episode.
        """
        if not self.config.auto_disable_swap_if_master5_stuck:
            return False
        if not self.config.swap_joints_45 or self.config.dof < 6:
            return False
        std = np.asarray(master_std_deg, dtype=np.float64).reshape(-1)
        if std.shape[0] < 6:
            return False
        thr = float(self.config.master_stuck_std_deg)
        if float(std[5]) < thr and float(std[4]) >= thr:
            self.config.swap_joints_45 = False
            return True
        return False

    def align_to_follower(
        self,
        follower_qpos7: Sequence[float],
        master_deg: Sequence[float],
    ) -> float:
        """Re-base mapping: current U-Arm pose -> current follower joints (soft sync)."""
        follower = np.asarray(follower_qpos7, dtype=np.float64)
        master = np.asarray(master_deg, dtype=np.float64)
        base = self._mapped_joint_from_master(master, include_runtime_offset=False)
        self._runtime_offset_rad[:] = follower[: self.config.dof] - base
        self._last_joint[:] = follower[: self.config.dof]
        self._last_gripper = float(follower[6])
        self._inited = True
        return float(np.max(np.abs(self._runtime_offset_rad)))

    def map_master_deg(self, master_angles_deg: Sequence[float]) -> TeleopCommand:
        master = np.asarray(master_angles_deg, dtype=np.float64)
        desired_joint = self._avoid_singularities(self._mapped_joint_from_master(master))
        desired_grip = self._map_gripper_cmd(master)

        if not self._inited:
            self._last_joint[:] = desired_joint
            self._last_gripper = float(self.config.gripper_open_cmd)
            self._inited = True

        max_step = np.asarray(self.config.max_joint_vel, dtype=np.float64) * self.control_dt
        delta = desired_joint - self._last_joint
        cmd_joint = self._last_joint + np.clip(delta, -max_step, max_step)

        open_cmd = float(self.config.gripper_open_cmd)
        grip_delta = desired_grip - self._last_gripper
        # Elastic clip released: snap to fully open (rate limit only slows reopen otherwise).
        if desired_grip >= open_cmd - 1e-6 and grip_delta > 0:
            cmd_grip = open_cmd
        else:
            grip_step = self.config.max_gripper_vel * self.control_dt
            cmd_grip = self._last_gripper + float(np.clip(grip_delta, -grip_step, grip_step))

        self._last_joint[:] = cmd_joint
        self._last_gripper = cmd_grip

        return TeleopCommand(
            joint_pos_rad=cmd_joint.copy(),
            gripper=cmd_grip,
            master_deg=master.copy(),
        )

    def to_qpos7(self, command: TeleopCommand) -> np.ndarray:
        qpos = np.zeros(7, dtype=np.float64)
        qpos[: self.config.dof] = command.joint_pos_rad
        qpos[6] = command.gripper
        return qpos
