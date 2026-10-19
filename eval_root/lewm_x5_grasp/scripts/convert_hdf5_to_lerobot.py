#!/usr/bin/env python3
"""Convert record_teach.py HDF5 episodes to LeRobot v2.1 for StarVLA Arx5_ljn.

Input (HDF5 from record_teach.py):
  observations/qpos[T,7]     6 joint rad + gripper
  observations/eef[T,7]      xyz(m) + rpy(rad) + gripper  (stored, not exported yet)
  observations/images/wrist  D405 RGB uint8
  action[T,7]                = qpos at same timestep (abs_qpos BC)

Output (LeRobot v2.1, matches examples/Arx5_ljn/modality.json):
  observation.state  -> 7D abs joint + gripper
  action             -> 7D abs joint + gripper
  observation.images.cam_wrist -> mp4 per episode

Coordinate note (TODO, compare with LIBERO later):
  Pass-through from /arm_status — no frame flip or axis remap in this script.
  Joint order = ROS2 joint_pos[0:7]. EEF rpy = ARX controller end_pos[3:6].
  Verify against LIBERO/Franka conventions before sim2real eval.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from lerobot_episode_io import ACTION_KEY, STATE_KEY, load_hdf5_episode

VIDEO_KEY_WRIST = "observation.images.cam_wrist"
VIDEO_KEY_HIGH = "observation.images.cam_high"


def _episode_sort_key(path: Path) -> tuple:
    m = re.search(r"episode_(\d+)$", path.stem)
    if m:
        return (0, int(m.group(1)), path.stem)
    ts_m = re.search(r"episode_(\d{4}-\d{2}-\d{2}_\d{6})", path.stem)
    if ts_m:
        return (1, ts_m.group(1), path.stem)
    return (2, path.stem, str(path.stat().st_mtime))


def _episode_has_movement(qpos: np.ndarray, tol: float = 0.05) -> bool:
    if len(qpos) < 2:
        return False
    return bool(np.any(np.max(np.abs(qpos - qpos[0]), axis=0) > tol))


def _write_video_ffmpeg(frames: np.ndarray, out_path: Path, fps: int) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, c = frames.shape
    if c != 3:
        raise ValueError(f"Expected RGB frames (H,W,3), got shape {frames.shape}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        step = max(1, n // 20)
        for i in range(n):
            proc.stdin.write(np.ascontiguousarray(frames[i], dtype=np.uint8).tobytes())
            if i % step == 0 or i == n - 1:
                print(f"    video {i + 1}/{n}", flush=True)
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode() if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err.strip() or out_path}")
    return h, w


def _write_video_imageio(frames: np.ndarray, out_path: Path, fps: int) -> Tuple[int, int]:
    import imageio.v3 as iio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, c = frames.shape
    step = max(1, n // 20)
    print(f"    encoding {n} frames via imageio...", flush=True)
    iio.imwrite(
        out_path,
        frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )
    for i in range(0, n, step):
        print(f"    video {min(i + step, n)}/{n}", flush=True)
    return h, w


def _write_video(frames: np.ndarray, out_path: Path, fps: int) -> Tuple[int, int]:
    if shutil.which("ffmpeg"):
        return _write_video_ffmpeg(frames, out_path, fps)
    try:
        return _write_video_imageio(frames, out_path, fps)
    except ImportError as e:
        raise SystemExit(
            "Video export needs ffmpeg in PATH or: pip install imageio imageio-ffmpeg"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Video export failed for {out_path}. "
            "Install ffmpeg or: pip install imageio imageio-ffmpeg"
        ) from e


def _series_stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def _scalar_stats(arr: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


def _build_info_json(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int,
    state_dim: int,
    video_shapes: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for video_key, (video_h, video_w) in video_shapes.items():
        features[video_key] = {
            "dtype": "video",
            "shape": [video_h, video_w, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": video_h,
                "video.width": video_w,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": {"motors": [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"]},
    }
    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": {"motors": [f"joint_{i}" for i in range(state_dim - 1)] + ["gripper"]},
    }
    num_cams = max(len(video_shapes), 1)
    return {
        "codebase_version": "v2.1",
        "robot_type": "arx_x5",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * num_cams,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def convert(
    input_dir: Path,
    output_dir: Path,
    fps: int,
    modality_template: Path,
    pattern: str,
    min_frames: int,
    skip_static: bool,
) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas required. pip install pandas pyarrow") from e

    episodes = sorted(input_dir.glob(pattern), key=_episode_sort_key)
    if not episodes:
        raise FileNotFoundError(f"No files matching {pattern} in {input_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    task_to_index: Dict[str, int] = {}
    episodes_meta: List[Dict[str, Any]] = []
    episodes_stats: List[Dict[str, Any]] = []

    all_states: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    global_index = 0
    out_ep_idx = 0
    video_shapes: Dict[str, Tuple[int, int]] = {}

    for ep_path in episodes:
        print(f"--- {ep_path.name} ---", flush=True)
        ep = load_hdf5_episode(ep_path)
        n = len(ep["qpos"])
        if n < min_frames:
            print(f"  skip: {n} frames < min_frames={min_frames}", flush=True)
            continue
        if skip_static and not _episode_has_movement(ep["qpos"]):
            print("  skip: no joint movement (static episode)", flush=True)
            continue

        task = ep["task"]
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
        task_index = task_to_index[task]

        images_by_cam = ep.get("images_by_cam") or {}
        if not images_by_cam and ep.get("images") is not None:
            images_by_cam = {"cam_wrist": ep["images"]}
        if not images_by_cam:
            raise ValueError(f"{ep_path}: missing observations/images")

        print(f"  -> episode_{out_ep_idx:06d}: {n} frames, task={task!r}, cams={list(images_by_cam)}", flush=True)
        for cam_name, frames in images_by_cam.items():
            video_key = f"observation.images.{cam_name}"
            video_dir = output_dir / "videos" / "chunk-000" / video_key
            video_dir.mkdir(parents=True, exist_ok=True)
            print(f"  writing video {video_key}...", flush=True)
            vh, vw = _write_video(
                frames,
                video_dir / f"episode_{out_ep_idx:06d}.mp4",
                fps=fps,
            )
            video_shapes[video_key] = (vh, vw)
        print("  writing parquet...", flush=True)

        timestamps = np.arange(n, dtype=np.float32) / float(fps)
        df = pd.DataFrame(
            {
                STATE_KEY: [row for row in ep["qpos"]],
                ACTION_KEY: [row for row in ep["action"]],
                "timestamp": timestamps,
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, out_ep_idx, dtype=np.int64),
                "index": np.arange(global_index, global_index + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
            }
        )
        df.to_parquet(data_dir / f"episode_{out_ep_idx:06d}.parquet", index=False)

        episodes_meta.append(
            {"episode_index": out_ep_idx, "tasks": [task], "length": n}
        )
        episodes_stats.append(
            {
                "episode_index": out_ep_idx,
                "stats": {
                    STATE_KEY: {**_series_stats(ep["qpos"]), "count": [n]},
                    ACTION_KEY: {**_series_stats(ep["action"]), "count": [n]},
                    "timestamp": _scalar_stats(timestamps),
                    "frame_index": _scalar_stats(np.arange(n)),
                    "episode_index": _scalar_stats(np.full(n, out_ep_idx)),
                    "index": _scalar_stats(np.arange(global_index, global_index + n)),
                    "task_index": _scalar_stats(np.full(n, task_index)),
                },
            }
        )

        all_states.append(ep["qpos"])
        all_actions.append(ep["action"])
        global_index += n
        out_ep_idx += 1

    if out_ep_idx == 0:
        raise RuntimeError("No valid episodes exported. Check min_frames / skip_static filters.")

    total_frames = global_index
    total_episodes = out_ep_idx
    state_dim = int(all_states[0].shape[1]) if all_states else 7

    info = _build_info_json(
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(task_to_index),
        fps=fps,
        state_dim=state_dim,
        video_shapes=video_shapes if video_shapes else {VIDEO_KEY_WRIST: (480, 480)},
    )
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, idx in sorted(task_to_index.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_stats:
            f.write(json.dumps(row) + "\n")

    states_cat = np.concatenate(all_states, axis=0)
    actions_cat = np.concatenate(all_actions, axis=0)
    stats_gr00t = {
        STATE_KEY: _series_stats(states_cat),
        ACTION_KEY: _series_stats(actions_cat),
        "timestamp": _scalar_stats(np.arange(total_frames, dtype=np.float32) / float(fps)),
        "frame_index": _scalar_stats(np.arange(total_frames)),
        "episode_index": _scalar_stats(
            np.concatenate(
                [np.full(m["length"], m["episode_index"]) for m in episodes_meta]
            )
        ),
        "index": _scalar_stats(np.arange(total_frames)),
        "task_index": _scalar_stats(
            np.concatenate(
                [
                    np.full(m["length"], task_to_index[m["tasks"][0]])
                    for m in episodes_meta
                ]
            )
        ),
    }
    (meta_dir / "stats_gr00t.json").write_text(
        json.dumps(stats_gr00t, indent=4), encoding="utf-8"
    )

    shutil.copyfile(modality_template, meta_dir / "modality.json")

    abs_out = output_dir.resolve()
    print("\n" + "=" * 60, flush=True)
    print(f"LeRobot dataset: {abs_out}", flush=True)
    print("=" * 60, flush=True)
    print(f"  episodes: {total_episodes}, frames: {total_frames}, tasks: {len(task_to_index)}")
    for vk, (vh, vw) in video_shapes.items():
        print(f"  video: {vk} ({vh}x{vw} @ {fps}fps)")
    print(f"  meta/   info.json, modality.json, tasks.jsonl, episodes.jsonl")
    print(f"  data/   chunk-000/episode_*.parquet")
    print(f"  videos/ chunk-000/observation.images.*/episode_*.mp4")
    print("\nServer sync:")
    print(f"  rsync -av {abs_out}/ SERVER:.../playground/Datasets/arx_x5_real/")
    print("  python examples/Arx5_ljn/train_files/prepare_arx5_dataset.py \\")
    print("    --data-root playground/Datasets")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_modality = script_dir.parent / "config" / "lerobot_modality_arx_x5.json"

    p = argparse.ArgumentParser(description="Convert teach HDF5 episodes to LeRobot v2.1.")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/teach_demos"),
        help="Directory with episode_*.hdf5",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/arx_x5_real/runs/latest"),
        help="LeRobot dataset root. run_convert_to_lerobot.sh sets a dated run dir by default.",
    )
    p.add_argument("--fps", type=int, default=30, help="FPS (match record --hz). Default: 30")
    p.add_argument("--pattern", default="episode_*.hdf5", help="HDF5 glob pattern")
    p.add_argument("--min-frames", type=int, default=30, help="Skip episodes shorter than this.")
    p.add_argument(
        "--keep-static",
        action="store_true",
        help="Export episodes with no joint movement (default: skip).",
    )
    p.add_argument(
        "--modality-template",
        type=Path,
        default=default_modality,
        help="modality.json for StarVLA arx_x5_real",
    )
    args = p.parse_args()

    convert(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        fps=args.fps,
        modality_template=args.modality_template.expanduser().resolve(),
        pattern=args.pattern,
        min_frames=args.min_frames,
        skip_static=not args.keep_static,
    )


if __name__ == "__main__":
    main()
