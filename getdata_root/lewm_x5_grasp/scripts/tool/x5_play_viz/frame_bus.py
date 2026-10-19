"""Atomic latest-frame bus between eval client and viz process."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .types import VizFrame

DEFAULT_BUS_DIR = Path(os.environ.get("X5_PLAY_VIZ_DIR", "/tmp/x5_play_viz"))
DEFAULT_BUS_PATH = DEFAULT_BUS_DIR / "latest.npz"


def ensure_bus_dir(path: Path = DEFAULT_BUS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def publish(
    frame: VizFrame,
    path: Path = DEFAULT_BUS_PATH,
) -> None:
    """Write frame atomically. Never raises to caller if wrapped by eval."""
    ensure_bus_dir(path)
    payload: dict[str, Any] = {"meta_json": np.asarray(json.dumps(frame.meta))}
    if frame.q_cmd is not None:
        payload["q_cmd"] = np.asarray(frame.q_cmd, dtype=np.float64)
    if frame.joint_pos is not None:
        payload["joint_pos"] = np.asarray(frame.joint_pos, dtype=np.float64)
    if frame.end_pos is not None:
        payload["end_pos"] = np.asarray(frame.end_pos, dtype=np.float64)
    if frame.cam_high is not None:
        payload["cam_high"] = np.asarray(frame.cam_high)
    if frame.cam_wrist is not None:
        payload["cam_wrist"] = np.asarray(frame.cam_wrist)

    fd, tmp_name = tempfile.mkstemp(prefix=".x5_viz_", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load(path: Path = DEFAULT_BUS_PATH) -> Optional[VizFrame]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            meta: dict[str, Any] = {}
            if "meta_json" in z.files:
                raw = z["meta_json"]
                text = raw.item() if raw.ndim == 0 else str(raw)
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                meta = json.loads(str(text))
            return VizFrame(
                q_cmd=z["q_cmd"] if "q_cmd" in z.files else None,
                joint_pos=z["joint_pos"] if "joint_pos" in z.files else None,
                end_pos=z["end_pos"] if "end_pos" in z.files else None,
                cam_high=z["cam_high"] if "cam_high" in z.files else None,
                cam_wrist=z["cam_wrist"] if "cam_wrist" in z.files else None,
                meta=meta,
            )
    except Exception:
        return None
