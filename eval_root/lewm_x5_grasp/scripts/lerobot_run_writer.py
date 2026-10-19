"""Incremental LeRobot v2.1 dataset writer (one dated run dir, many episodes)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from convert_hdf5_to_lerobot import (
    ACTION_KEY,
    STATE_KEY,
    _build_info_json,
    _scalar_stats,
    _series_stats,
    _write_video,
)
from lerobot_episode_io import write_episode_parquet

DEFAULT_MODALITY = Path(__file__).resolve().parent.parent / "config" / "lerobot_modality_arx_x5.json"


def _video_feature_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}"


def _normalize_image_buffers(
    images: Union[List[Any], Dict[str, List[Any]]],
    camera_names: List[str],
) -> Dict[str, List[Any]]:
    if isinstance(images, dict):
        return images
    if len(camera_names) == 1:
        return {camera_names[0]: images}
    raise ValueError("Multi-camera episode requires images dict keyed by camera name.")


class LeRobotRunWriter:
    def __init__(
        self,
        run_dir: Path,
        fps: float,
        camera_names: List[str],
        modality_template: Path = DEFAULT_MODALITY,
    ) -> None:
        if not camera_names:
            raise ValueError("camera_names must not be empty")
        self.run_dir = run_dir.resolve()
        self.fps = int(fps)
        self.camera_names = list(camera_names)
        self.modality_template = modality_template.resolve()
        self.episode_count = 0
        self.total_frames = 0
        self.task_to_index: Dict[str, int] = {}
        self.episodes_meta: List[Dict[str, Any]] = []
        self.episodes_stats: List[Dict[str, Any]] = []
        self.all_states: List[np.ndarray] = []
        self.all_actions: List[np.ndarray] = []
        self.video_shapes: Dict[str, Tuple[int, int]] = {
            _video_feature_key(name): (480, 480) for name in self.camera_names
        }

        meta_dir = self.run_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.modality_template, meta_dir / "modality.json")

    @classmethod
    def create_dated_run(
        cls,
        runs_root: Path,
        fps: float,
        camera_names: List[str],
    ) -> "LeRobotRunWriter":
        from datetime import datetime

        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = runs_root / run_id
        writer = cls(run_dir, fps=fps, camera_names=camera_names)
        runs_root.mkdir(parents=True, exist_ok=True)
        latest = runs_root / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir)
        return writer

    def append_episode(self, task: str, buffers: Dict[str, List[Any]]) -> Path:
        qpos = np.stack(buffers["qpos"], axis=0).astype(np.float32)
        action = np.stack(buffers["action"], axis=0).astype(np.float32)
        eef = (
            np.stack(buffers["eef"], axis=0).astype(np.float32)
            if buffers.get("eef")
            else None
        )
        images_by_cam = _normalize_image_buffers(buffers["images"], self.camera_names)
        n = len(qpos)
        if n == 0:
            raise ValueError("empty episode buffers")

        ep_idx = self.episode_count
        if task not in self.task_to_index:
            self.task_to_index[task] = len(self.task_to_index)
        task_index = self.task_to_index[task]

        parquet_path = self.run_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        for cam_name in self.camera_names:
            if cam_name not in images_by_cam or len(images_by_cam[cam_name]) != n:
                raise ValueError(
                    f"camera {cam_name!r}: expected {n} frames, "
                    f"got {len(images_by_cam.get(cam_name, []))}"
                )
            video_key = _video_feature_key(cam_name)
            video_path = (
                self.run_dir
                / "videos"
                / "chunk-000"
                / video_key
                / f"episode_{ep_idx:06d}.mp4"
            )
            frames = np.stack(images_by_cam[cam_name], axis=0).astype(np.uint8)
            print(f"  writing {cam_name} video ({n} frames)...", flush=True)
            h, w = _write_video(frames, video_path, fps=self.fps)
            self.video_shapes[video_key] = (h, w)

        print("  writing parquet...", flush=True)
        write_episode_parquet(
            qpos=qpos,
            action=action,
            out_path=parquet_path,
            fps=self.fps,
            episode_index=ep_idx,
            global_index=self.total_frames,
            task_index=task_index,
            eef=eef,
        )

        timestamps = np.arange(n, dtype=np.float32) / float(self.fps)
        self.episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n})
        self.episodes_stats.append(
            {
                "episode_index": ep_idx,
                "stats": {
                    STATE_KEY: {**_series_stats(qpos), "count": [n]},
                    ACTION_KEY: {**_series_stats(action), "count": [n]},
                    "timestamp": _scalar_stats(timestamps),
                    "frame_index": _scalar_stats(np.arange(n)),
                    "episode_index": _scalar_stats(np.full(n, ep_idx)),
                    "index": _scalar_stats(np.arange(self.total_frames, self.total_frames + n)),
                    "task_index": _scalar_stats(np.full(n, task_index)),
                },
            }
        )
        self.all_states.append(qpos)
        self.all_actions.append(action)
        self.total_frames += n
        self.episode_count += 1
        self._flush_meta()
        return parquet_path

    def delete_last_episode(self) -> bool:
        if self.episode_count == 0:
            return False
        prev = self.episode_count - 1
        paths = [f"data/chunk-000/episode_{prev:06d}.parquet"]
        for cam_name in self.camera_names:
            video_key = _video_feature_key(cam_name)
            paths.append(f"videos/chunk-000/{video_key}/episode_{prev:06d}.mp4")
        for rel in paths:
            path = self.run_dir / rel
            if path.exists():
                path.unlink()

        self.episode_count -= 1
        if self.episodes_meta:
            removed = self.episodes_meta.pop()
            self.total_frames -= int(removed["length"])
        if self.episodes_stats:
            self.episodes_stats.pop()
        if self.all_states:
            self.all_states.pop()
        if self.all_actions:
            self.all_actions.pop()
        self._flush_meta()
        return True

    def _flush_meta(self) -> None:
        meta_dir = self.run_dir / "meta"
        info = _build_info_json(
            total_episodes=self.episode_count,
            total_frames=self.total_frames,
            total_tasks=len(self.task_to_index),
            fps=self.fps,
            state_dim=7,
            video_shapes=self.video_shapes,
        )
        (meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

        with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as f:
            for task, idx in sorted(self.task_to_index.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

        with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in self.episodes_meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with (meta_dir / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
            for row in self.episodes_stats:
                f.write(json.dumps(row) + "\n")

        if self.all_states:
            states_cat = np.concatenate(self.all_states, axis=0)
            actions_cat = np.concatenate(self.all_actions, axis=0)
            stats_gr00t = {
                STATE_KEY: _series_stats(states_cat),
                ACTION_KEY: _series_stats(actions_cat),
                "timestamp": _scalar_stats(
                    np.arange(self.total_frames, dtype=np.float32) / float(self.fps)
                ),
                "frame_index": _scalar_stats(np.arange(self.total_frames)),
                "episode_index": _scalar_stats(
                    np.concatenate(
                        [np.full(m["length"], m["episode_index"]) for m in self.episodes_meta]
                    )
                ),
                "index": _scalar_stats(np.arange(self.total_frames)),
                "task_index": _scalar_stats(
                    np.concatenate(
                        [
                            np.full(m["length"], self.task_to_index[m["tasks"][0]])
                            for m in self.episodes_meta
                        ]
                    )
                ),
            }
            (meta_dir / "stats_gr00t.json").write_text(
                json.dumps(stats_gr00t, indent=4), encoding="utf-8"
            )
        elif (meta_dir / "stats_gr00t.json").exists():
            (meta_dir / "stats_gr00t.json").unlink()
