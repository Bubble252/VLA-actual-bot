#!/usr/bin/env python3
"""Tests for offline HDF5 singularity densify (no ROS)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SRC_DIR))

from optimize_hdf5_singularity import densify_near_danger, optimize_episode  # noqa: E402
from lewm_x5_grasp.singularity_adjuster import SingularityConfig  # noqa: E402


def test_densify_stretches_time_and_preserves_endpoints() -> None:
    n = 5
    eef = np.zeros((n, 7), dtype=np.float64)
    eef[:, 0] = np.linspace(0.0, 1.0, n)
    qpos = np.zeros((n, 7), dtype=np.float64)
    qpos[:, 4] = 0.5
    action = qpos.copy()
    stamps = np.arange(n, dtype=np.float64) * 0.1
    scales = np.ones(n, dtype=np.float64)
    scales[1:4] = 3.0
    out = densify_near_danger(
        eef=eef,
        qpos=qpos,
        action=action,
        stamps=stamps,
        scales=scales,
        extras={},
    )
    assert out["eef"][0, 0] == 0.0
    assert abs(out["eef"][-1, 0] - 1.0) < 1e-9
    assert out["inserted"] > 0
    # Danger intervals stretch: duration grows vs original 0.4s
    assert out["stamps"][-1] - out["stamps"][0] > 0.4 + 1e-9
    # Path monotonic in x
    assert np.all(np.diff(out["eef"][:, 0]) >= -1e-12)


def test_optimize_writes_new_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / "episode_0.hdf5"
        dst = td_path / "episode_0_opt.hdf5"
        n = 20
        qpos = np.zeros((n, 7), dtype=np.float32)
        qpos[:, 2] = 1.0
        qpos[:, 4] = 0.5
        qpos[8:12, 2] = 0.0  # elbow singular neighborhood
        eef = np.zeros((n, 7), dtype=np.float32)
        eef[:, 0] = np.linspace(0.2, 0.4, n)
        eef[:, 4] = 0.1
        stamps = np.arange(n, dtype=np.float64) / 30.0
        with h5py.File(src, "w") as f:
            f.attrs["task"] = "test"
            f.attrs["fps"] = 30.0
            f.attrs["camera_names"] = "[]"
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            obs.create_dataset("eef", data=eef)
            f.create_dataset("action", data=qpos.copy())
            f.create_dataset("action_eef", data=eef.copy())
            f.create_dataset("time", data=stamps)
            f.create_dataset("arm_stamp", data=stamps)

        cfg = SingularityConfig(
            elbow_thresh_rad=0.5,
            slowdown_enabled=True,
            slowdown_factor=3.0,
            slowdown_expand_frames=1,
            max_rpy_adjust_rad=np.deg2rad(3.0),
        )
        info = optimize_episode(
            src, dst, do_slowdown=True, do_adjust=True, cfg=cfg, skip_images=True
        )
        assert dst.is_file()
        assert info["frames_out"] >= info["frames_in"]
        with h5py.File(dst, "r") as f:
            assert bool(f.attrs["optimized"])
            assert f["observations/eef"].shape[0] == info["frames_out"]
            assert f["action"].shape[0] == info["frames_out"]


if __name__ == "__main__":
    test_densify_stretches_time_and_preserves_endpoints()
    test_optimize_writes_new_file()
    print("optimize_hdf5_singularity tests passed")
