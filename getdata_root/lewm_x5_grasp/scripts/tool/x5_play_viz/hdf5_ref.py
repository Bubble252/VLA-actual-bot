"""Load a static HDF5 reference pose / images for P4 overlay."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Prefer recorded feedback joints, then action, then eef-only fallbacks.
_QPOS_KEYS: tuple[str, ...] = (
    "observations/qpos",
    "observations/joint_pos",
    "action",
    "observations/actions",
)
_EEF_KEYS: tuple[str, ...] = (
    "observations/eef",
    "action_eef",
)
_IMG_HIGH_KEYS: tuple[str, ...] = (
    "observations/images/cam_high",
    "observations/images/agent",
    "observations/images/cam_front",
)
_IMG_WRIST_KEYS: tuple[str, ...] = (
    "observations/images/cam_wrist",
    "observations/images/wrist",
    "observations/images/cam_left_wrist",
)


@dataclass
class Hdf5Ref:
    qpos: Optional[np.ndarray] = None  # (6,)
    eef: Optional[np.ndarray] = None  # (6,) xyzrpy
    cam_high: Optional[np.ndarray] = None
    cam_wrist: Optional[np.ndarray] = None
    frame_idx: int = 0
    path: Optional[Path] = None
    found: dict[str, str] = field(default_factory=dict)  # role -> key
    tried: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_qpos(self) -> bool:
        return self.qpos is not None

    @property
    def has_images(self) -> bool:
        return self.cam_high is not None or self.cam_wrist is not None


def _frame_slice(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        return a
    idx = int(np.clip(frame_idx, 0, a.shape[0] - 1))
    return a[idx]


def _first_dataset(f, keys: Sequence[str]) -> tuple[Optional[str], Optional[Any]]:
    for key in keys:
        if key in f:
            return key, f[key]
    return None, None


def _list_image_keys(f) -> list[str]:
    out: list[str] = []
    if "observations/images" not in f:
        return out
    grp = f["observations/images"]
    for name in grp.keys():
        out.append(f"observations/images/{name}")
    return out


def load_hdf5_ref(path: str | Path, frame_idx: int = 0) -> Hdf5Ref:
    """Probe common teleop / teach keys; print what was tried vs found."""
    try:
        import h5py
    except ImportError:
        print("[hdf5_ref] h5py missing; skip --hdf5", file=sys.stderr)
        return Hdf5Ref(frame_idx=frame_idx)

    p = Path(path).expanduser()
    ref = Hdf5Ref(frame_idx=frame_idx, path=p)
    ref.tried = {
        "qpos": list(_QPOS_KEYS),
        "eef": list(_EEF_KEYS),
        "cam_high": list(_IMG_HIGH_KEYS),
        "cam_wrist": list(_IMG_WRIST_KEYS),
    }

    if not p.is_file():
        print(f"[hdf5_ref] file not found: {p}", file=sys.stderr)
        return ref

    with h5py.File(p, "r") as f:
        top = list(f.keys())
        img_keys = _list_image_keys(f)
        print(
            f"[hdf5_ref] {p.name}  top={top}  "
            f"images={img_keys or '(none)'}  ref_frame={frame_idx}"
        )

        q_key, q_ds = _first_dataset(f, _QPOS_KEYS)
        if q_key is not None and q_ds is not None:
            q = np.asarray(_frame_slice(q_ds, frame_idx), dtype=np.float64).reshape(-1)
            ref.qpos = q[:6]
            ref.found["qpos"] = q_key
            print(
                f"[hdf5_ref] qpos <- {q_key}[{frame_idx}] "
                f"q[:6]={np.array2string(ref.qpos, precision=3)}"
            )
        else:
            print(
                "[hdf5_ref] no qpos-like dataset; tried: "
                + ", ".join(_QPOS_KEYS),
                file=sys.stderr,
            )

        e_key, e_ds = _first_dataset(f, _EEF_KEYS)
        if e_key is not None and e_ds is not None:
            e = np.asarray(_frame_slice(e_ds, frame_idx), dtype=np.float64).reshape(-1)
            if e.size >= 6:
                ref.eef = e[:6]
                ref.found["eef"] = e_key
                print(
                    f"[hdf5_ref] eef  <- {e_key}[{frame_idx}] "
                    f"xyzrpy={np.array2string(ref.eef, precision=3)}"
                )
        else:
            print(
                "[hdf5_ref] no eef dataset (optional); tried: "
                + ", ".join(_EEF_KEYS)
            )

        for role, candidates in (
            ("cam_high", _IMG_HIGH_KEYS),
            ("cam_wrist", _IMG_WRIST_KEYS),
        ):
            # also accept whatever exists under observations/images/
            keys = list(candidates)
            for ik in img_keys:
                if ik not in keys:
                    # heuristic: high vs wrist in name
                    low = ik.lower()
                    if role == "cam_high" and any(
                        s in low for s in ("high", "agent", "front", "desktop")
                    ):
                        keys.insert(0, ik)
                    if role == "cam_wrist" and "wrist" in low:
                        keys.insert(0, ik)
            k, ds = _first_dataset(f, keys)
            if k is None or ds is None:
                print(f"[hdf5_ref] no {role}; tried: {', '.join(candidates)}")
                continue
            frame = _frame_slice(ds, frame_idx)
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                if role == "cam_high":
                    ref.cam_high = arr[..., :3]
                else:
                    ref.cam_wrist = arr[..., :3]
                ref.found[role] = k
                print(f"[hdf5_ref] {role} <- {k}[{frame_idx}] shape={arr.shape}")
            else:
                print(
                    f"[hdf5_ref] {role} key {k} unexpected shape {arr.shape}",
                    file=sys.stderr,
                )

    if not ref.found:
        print("[hdf5_ref] nothing usable found in file", file=sys.stderr)
    else:
        print(f"[hdf5_ref] using fields: {ref.found}")
    return ref
