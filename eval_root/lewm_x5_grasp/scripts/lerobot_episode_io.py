"""Shared HDF5 <-> LeRobot episode I/O (parquet only, no video)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np

STATE_KEY = "observation.state"
ACTION_KEY = "action"
EEF_KEY = "observation.eef"


def load_hdf5_episode(path: Path, *, load_eef_fk: bool = False) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)
        eef = np.asarray(f["observations/eef"], dtype=np.float32)
        eef_fk = None
        if load_eef_fk and "observations/eef_fk" in f:
            eef_fk = np.asarray(f["observations/eef_fk"], dtype=np.float32)
        task = str(f.attrs.get("task", "manipulation task"))

        camera_names: list[str] = []
        if "camera_names" in f.attrs:
            camera_names = json.loads(str(f.attrs["camera_names"]))

        images: Dict[str, np.ndarray] = {}
        if "observations/images" in f:
            img_grp = f["observations/images"]
            keys = camera_names or list(img_grp.keys())
            for name in keys:
                if name in img_grp:
                    images[name] = np.asarray(img_grp[name], dtype=np.uint8)
            if not images and "wrist" in img_grp:
                images["cam_wrist"] = np.asarray(img_grp["wrist"], dtype=np.uint8)

        if "cam_wrist" in images:
            legacy_wrist = images["cam_wrist"]
        elif "wrist" in images:
            legacy_wrist = images["wrist"]
        elif images:
            legacy_wrist = next(iter(images.values()))
        else:
            legacy_wrist = None

        stamps = (
            np.asarray(f["arm_stamp"], dtype=np.float64)
            if "arm_stamp" in f
            else np.asarray(f["time"], dtype=np.float64)
        )
        fps = float(f.attrs["fps"]) if "fps" in f.attrs else None
    if qpos.shape != action.shape:
        raise ValueError(f"{path}: qpos {qpos.shape} != action {action.shape}")
    return {
        "qpos": qpos,
        "action": action,
        "eef": eef,
        "eef_fk": eef_fk,
        "images": legacy_wrist,
        "images_by_cam": images,
        "camera_names": list(images.keys()),
        "task": task,
        "stamps": stamps,
        "fps": fps,
    }


def write_episode_parquet(
    *,
    qpos: np.ndarray,
    action: np.ndarray,
    out_path: Path,
    fps: float,
    episode_index: int = 0,
    global_index: int = 0,
    task_index: int = 0,
    eef: np.ndarray | None = None,
) -> None:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas required. pip install pandas pyarrow") from e

    n = len(qpos)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(n, dtype=np.float32) / float(fps)
    df = pd.DataFrame(
        {
            STATE_KEY: [row for row in qpos],
            ACTION_KEY: [row for row in action],
            "timestamp": timestamps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, task_index, dtype=np.int64),
        }
    )
    if eef is not None:
        df[EEF_KEY] = [row for row in eef]
    df.to_parquet(out_path, index=False)


def hdf5_to_parquet(hdf5_path: Path, parquet_path: Path, fps: float = 20.0) -> Path:
    """Convert one teach HDF5 file to a LeRobot-style episode parquet."""
    ep = load_hdf5_episode(hdf5_path)
    write_episode_parquet(
        qpos=ep["qpos"],
        action=ep["action"],
        out_path=parquet_path,
        fps=fps,
    )
    return parquet_path


def load_parquet_episode(parquet_path: Path) -> Dict[str, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas required. pip install pandas pyarrow") from e

    df = pd.read_parquet(parquet_path)
    if ACTION_KEY not in df.columns:
        raise KeyError(f"{parquet_path} missing column {ACTION_KEY!r}")

    action = np.stack(df[ACTION_KEY].to_numpy(), axis=0).astype(np.float64)
    state = (
        np.stack(df[STATE_KEY].to_numpy(), axis=0).astype(np.float64)
        if STATE_KEY in df.columns
        else action.copy()
    )
    timestamps = (
        df["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in df.columns
        else np.arange(len(action), dtype=np.float64) / 10.0
    )
    out: Dict[str, np.ndarray] = {
        "action": action,
        "state": state,
        "timestamp": timestamps,
    }
    if EEF_KEY in df.columns:
        out["eef"] = np.stack(df[EEF_KEY].to_numpy(), axis=0).astype(np.float64)
    return out


def compare_hdf5_parquet(hdf5_path: Path, parquet_path: Path) -> Dict[str, float]:
    """Return max abs diff between HDF5 and parquet tensors (conversion sanity check)."""
    ep = load_hdf5_episode(hdf5_path)
    lr = load_parquet_episode(parquet_path)
    action_diff = float(np.max(np.abs(ep["action"] - lr["action"])))
    state_diff = float(np.max(np.abs(ep["qpos"] - lr["state"])))
    return {"action_max_abs_diff": action_diff, "state_max_abs_diff": state_diff}
