#!/usr/bin/env python3
"""Quick checks for replay trajectory smoothing (no ROS required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from episode_replay import (  # noqa: E402
    interpolate_trajectory,
    limit_eef_orientation_steps,
    plan_eef_replay_frame_modes,
    prepare_replay_trajectory,
    smooth_eef_orientation_quat,
    smooth_trajectory,
)


def test_interpolate_rpy_is_continuous() -> None:
    n = 5
    stamps = np.arange(n, dtype=np.float64) * 0.05
    qpos = np.zeros((n, 7), dtype=np.float64)
    eef = np.zeros((n, 7), dtype=np.float64)
    eef[:, 3] = np.linspace(-3.0, 3.0, n)
    eef_i, qpos_i, _ = interpolate_trajectory(eef, qpos, stamps, out_hz=50.0)
    steps = np.abs(np.diff(eef_i[:, 3]))
    assert float(np.max(steps)) < 0.65, f"RPY steps too large: {np.max(steps)}"


def test_smooth_reduces_joint_jitter() -> None:
    n = 100
    qpos = np.zeros((n, 7), dtype=np.float64)
    qpos[:, 0] = np.sin(np.linspace(0, 6.0, n)) + 0.08 * np.random.default_rng(0).standard_normal(n)
    eef = qpos.copy()
    eef_i, qpos_i = smooth_trajectory(eef, qpos, window=7)
    raw_j = float(np.mean(np.abs(np.diff(qpos[:, 0]))))
    smooth_j = float(np.mean(np.abs(np.diff(qpos_i[:, 0]))))
    assert smooth_j < raw_j, f"smooth did not reduce jitter: {smooth_j} vs {raw_j}"


def test_eef_rpy_rate_limit() -> None:
    n = 20
    stamps = np.arange(n, dtype=np.float64) * 0.05
    qpos = np.zeros((n, 7), dtype=np.float64)
    eef = np.zeros((n, 7), dtype=np.float64)
    eef[:, 4] = np.linspace(0.0, 1.0, n)
    limited = limit_eef_orientation_steps(eef, max_rpy_step_rad=0.04)
    steps = np.abs(np.diff(limited[:, 4]))
    assert float(np.max(steps)) <= 0.0401, f"rate limit failed: {np.max(steps)}"


def test_prepare_eef_trajectory_notes() -> None:
    n = 30
    stamps = np.arange(n, dtype=np.float64) * 0.05
    qpos = np.zeros((n, 7), dtype=np.float64)
    eef = qpos.copy()
    eef[:, 4] = np.sin(np.linspace(0, 4.0, n)) + 0.1 * np.random.default_rng(1).standard_normal(n)
    _, _, _, note = prepare_replay_trajectory(
        eef,
        qpos,
        stamps,
        out_hz=50.0,
        smooth_window=7,
        control_mode="eef",
        eef_smooth_window=21,
        eef_max_rpy_step_rad=0.03,
        eef_ik_seed_smooth_window=15,
        eef_quat_smooth_window=7,
    )
    assert "eef_rpy_smooth=21" in note
    assert "ik_seed_smooth=15" in note
    assert "eef_quat_smooth=7" in note
    assert "rpy_step<=" in note


def test_quat_smooth_reduces_rpy_spikes() -> None:
    n = 100
    qpos = np.zeros((n, 7), dtype=np.float64)
    eef = np.zeros((n, 7), dtype=np.float64)
    eef[:, 4] = np.sin(np.linspace(0, 6.0, n))
    eef[50, 4] += 0.35
    raw = float(np.max(np.abs(np.diff(np.degrees(eef[:, 4])))))
    smoothed = smooth_eef_orientation_quat(eef, window=7)
    sm = float(np.max(np.abs(np.diff(np.degrees(smoothed[:, 4])))))
    assert sm < raw, f"quat smooth did not reduce spike: {sm} vs {raw}"


def test_eef_joint_fallback_plan_triggers_on_fast_motion() -> None:
    n = 200
    period = 0.02
    qpos = np.zeros((n, 7), dtype=np.float64)
    eef = np.zeros((n, 7), dtype=np.float64)
    qpos[:80, 0] = 0.0
    qpos[80:120, 0] = np.linspace(0.0, 0.6, 40)
    qpos[120:, 0] = 0.6
    eef[:, :6] = qpos[:, :6]
    modes, stats = plan_eef_replay_frame_modes(
        eef,
        qpos,
        period,
        lead_steps=10,
        fallback_enabled=True,
        step_thresh_rad=np.deg2rad(0.85),
        hold_sec=0.8,
        lookahead_sec=0.2,
    )
    assert stats["fallback_frames"] > 0
    assert modes[100] == "joint"
    assert modes[0] == "joint"
    assert any(m == "eef" for m in modes[150:])


if __name__ == "__main__":
    test_interpolate_rpy_is_continuous()
    test_smooth_reduces_joint_jitter()
    test_eef_rpy_rate_limit()
    test_prepare_eef_trajectory_notes()
    test_quat_smooth_reduces_rpy_spikes()
    test_eef_joint_fallback_plan_triggers_on_fast_motion()
    print("replay smooth tests passed")
