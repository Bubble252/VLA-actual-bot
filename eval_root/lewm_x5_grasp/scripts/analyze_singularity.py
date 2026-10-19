#!/usr/bin/env python3
"""Offline singularity scan for teleop/teach HDF5 episodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_DIR / "src"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from episode_replay import load_episode_arrays, prepare_replay_trajectory  # noqa: E402
from lewm_x5_grasp.singularity_adjuster import SingularityAdjuster, SingularityConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze EEF replay singularity risk in HDF5 episodes.")
    p.add_argument("hdf5", type=Path, help="Path to episode_*.hdf5")
    p.add_argument("--replay-hz", type=float, default=50.0, help="Resample rate for analysis")
    p.add_argument("--seed-source", choices=("action", "state"), default="action")
    p.add_argument("--eef-source", choices=("auto", "action", "observations", "eef_fk"), default="auto")
    p.add_argument("--adjust", action="store_true", help="Run offline RPY micro-adjustment preview")
    p.add_argument("--no-adjust", action="store_true", help="Skip adjust preview even with --adjust off")
    p.add_argument("--min-run", type=int, default=3, help="Min consecutive dangerous frames to report")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = args.hdf5.resolve()
    if not hdf5_path.is_file():
        raise SystemExit(f"File not found: {hdf5_path}")

    eef, qpos, stamps = load_episode_arrays(
        hdf5_path,
        joint_source="action",
        control_mode="eef",
        eef_source=args.eef_source,  # type: ignore[arg-type]
        eef_ik_seed=args.seed_source,  # type: ignore[arg-type]
    )
    eef_i, qpos_i, period, note = prepare_replay_trajectory(
        eef,
        qpos,
        stamps,
        out_hz=args.replay_hz,
        control_mode="eef",
        singularity_adjust=False,
    )

    adjuster = SingularityAdjuster(SingularityConfig())
    flags = adjuster.detect(qpos_i)
    segments = adjuster.segment_report(qpos_i, min_run=max(1, args.min_run))
    danger_frames = int(np.sum(flags))

    print(f"file: {hdf5_path.name}")
    print(f"frames: raw={len(qpos)} resampled={len(qpos_i)} period={period:.4f}s")
    print(f"preprocess: {note}")
    print(f"danger frames: {danger_frames}/{len(qpos_i)} ({100.0 * danger_frames / max(len(qpos_i), 1):.1f}%)")
    if segments:
        print("danger segments (>= min_run):")
        for seg in segments:
            lo, hi = seg["start"], seg["end"]
            t0 = lo * period
            t1 = hi * period
            print(f"  [{lo:4d}, {hi:4d}] len={seg['length']:4d}  t={t0:6.2f}-{t1:6.2f}s")
    else:
        print("danger segments: none")

    if args.adjust and not args.no_adjust:
        eef_adj, qpos_adj, stats = adjuster.adjust(eef_i.copy(), qpos_i)
        print(f"adjust: {stats.summary()}")
        if stats.warnings:
            for w in stats.warnings:
                print(f"  warn: {w}")
        verify = adjuster.verify(eef_adj, qpos_adj)
        if verify["warnings"]:
            for warn in verify["warnings"]:
                print(f"  warn: {warn}")


if __name__ == "__main__":
    main()
