#!/usr/bin/env python3
"""Quick sanity check for a LeRobot v2.1 dataset directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _ffprobe_frames(video: Path) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(video),
            ],
            text=True,
        ).strip()
        return int(out) if out.isdigit() else None
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def verify_dataset(root: Path) -> int:
    root = root.expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print(f"LeRobot dataset: {root}")
    print("=" * 60)

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print("FAIL: missing meta/info.json")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    print(f"codebase: {info.get('codebase_version')}  robot: {info.get('robot_type')}")
    print(
        f"episodes: {info.get('total_episodes')}  frames: {info.get('total_frames')}  "
        f"fps: {info.get('fps')}"
    )

    modality_path = root / "meta" / "modality.json"
    if modality_path.exists():
        modality = json.loads(modality_path.read_text(encoding="utf-8"))
        video_keys = {
            v["original_key"] for v in modality.get("video", {}).values()
        }
    else:
        video_keys = {
            k for k in info.get("features", {}) if k.startswith("observation.images.")
        }
        warnings.append("meta/modality.json missing")

    print(f"cameras: {', '.join(sorted(video_keys)) or '(none)'}")

    try:
        import pandas as pd
    except ImportError:
        print("FAIL: pip install pandas pyarrow")
        return 1

    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        errors.append("missing meta/episodes.jsonl")
        episodes = []
    else:
        episodes = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    total_parquet_frames = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        length = int(ep["length"])
        parquet = root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not parquet.exists():
            errors.append(f"episode {ep_idx}: missing {parquet.name}")
            continue

        df = pd.read_parquet(parquet)
        n = len(df)
        total_parquet_frames += n
        if n != length:
            errors.append(f"episode {ep_idx}: episodes.jsonl length={length} but parquet has {n}")

        for col in ("observation.state", "action", "timestamp"):
            if col not in df.columns:
                errors.append(f"episode {ep_idx}: parquet missing column {col!r}")

        if "observation.state" in df.columns and "action" in df.columns:
            states = np.stack(df["observation.state"].to_numpy(), axis=0)
            actions = np.stack(df["action"].to_numpy(), axis=0)
            diff = float(np.max(np.abs(states - actions)))
            print(f"  ep {ep_idx}: parquet {n} rows, max|state-action|={diff:.4f}")
            if diff > 0.05:
                warnings.append(
                    f"episode {ep_idx}: state vs action differ a lot (max {diff:.3f}); "
                    "expected close during replay-record"
                )

        for video_key in sorted(video_keys):
            cam_name = video_key.split("observation.images.", 1)[-1]
            video = (
                root
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{ep_idx:06d}.mp4"
            )
            if not video.exists():
                errors.append(f"episode {ep_idx}: missing video {video_key}")
                continue
            size_kb = video.stat().st_size / 1024
            vf = _ffprobe_frames(video)
            vf_str = str(vf) if vf is not None else "?"
            print(f"         video {cam_name}: {size_kb:.0f} KB, ~{vf_str} frames")
            if vf is not None and abs(vf - n) > 2:
                warnings.append(
                    f"episode {ep_idx} {cam_name}: video frames ({vf}) != parquet rows ({n})"
                )
            if video.stat().st_size < 10_000:
                warnings.append(f"episode {ep_idx} {cam_name}: video file very small")

    if int(info.get("total_frames", -1)) != total_parquet_frames:
        errors.append(
            f"info.json total_frames={info.get('total_frames')} "
            f"but parquet sum={total_parquet_frames}"
        )

    tasks_path = root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        tasks = [json.loads(l) for l in tasks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"tasks: {[t.get('task') for t in tasks]}")

    print("=" * 60)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print("RESULT: FAIL")
        return 1

    print("RESULT: OK")
    print("\nAlso check manually:")
    print("  1. Play mp4 — motion matches the task?")
    print("  2. Wrist + desktop views both sensible? (if dual cam)")
    print("  3. Joint values in reasonable rad range?")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Verify LeRobot v2.1 dataset layout.")
    p.add_argument(
        "dataset_dir",
        nargs="?",
        default="data/arx_x5_real/runs/latest",
        help="Dataset root (default: data/arx_x5_real/runs/latest)",
    )
    args = p.parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.dataset_dir)
    if not root.is_absolute():
        root = (script_dir.parent / root).resolve()
    sys.exit(verify_dataset(root))


if __name__ == "__main__":
    main()
