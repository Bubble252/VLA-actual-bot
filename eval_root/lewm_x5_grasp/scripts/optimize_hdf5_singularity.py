#!/usr/bin/env python3
"""Offline HDF5 singularity optimizer (no robot / no ROS required).

Rewrites a copy of an episode so EEF replay is safer near singularities:
  - slowdown: densify path near danger (same geometry, stretched time)
  - adjust:   small RPY micro-nudge on observations/eef (and action_eef)

Original file is never overwritten. Images are hold-duplicated when densifying.
Does NOT use FK-delta rewrite or joint-mask hacks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lewm_x5_grasp.singularity_adjuster import SingularityAdjuster, SingularityConfig  # noqa: E402


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * a + alpha * b


def _lerp_angle(a: float, b: float, alpha: float) -> float:
    d = float(np.arctan2(np.sin(b - a), np.cos(b - a)))
    return float(a + alpha * d)


def _lerp_vec_angles(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(a, dtype=np.float64)
    for i in range(len(a)):
        out[i] = _lerp_angle(float(a[i]), float(b[i]), alpha)
    return out


def _interp_joint_row(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = a.copy()
    n_arm = min(6, len(a), len(b))
    out[:n_arm] = _lerp_vec_angles(a[:n_arm], b[:n_arm], alpha)
    if len(a) > 6:
        out[6:] = _lerp(a[6:], b[6:], alpha)
    return out


def _interp_eef_row(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = a.copy()
    out[:3] = _lerp(a[:3], b[:3], alpha)
    if len(a) >= 6:
        out[3:6] = _lerp_vec_angles(a[3:6], b[3:6], alpha)
    if len(a) > 6:
        out[6:] = _lerp(a[6:], b[6:], alpha)
    return out


def densify_near_danger(
    *,
    eef: np.ndarray,
    qpos: np.ndarray,
    action: np.ndarray,
    stamps: np.ndarray,
    scales: np.ndarray,
    extras: dict[str, np.ndarray],
    image_index_src: np.ndarray | None = None,
) -> dict[str, Any]:
    """Insert interpolated samples on intervals adjacent to singular frames.

    Geometry follows the recorded path; wall-clock time of each original interval
    stretches by n_steps (= round(max(scale_i, scale_{i+1}))).
    """
    eef = np.asarray(eef, dtype=np.float64)
    qpos = np.asarray(qpos, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    stamps = np.asarray(stamps, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    n = len(qpos)
    if n == 0:
        raise ValueError("empty episode")
    if not (len(eef) == len(action) == len(stamps) == len(scales) == n):
        raise ValueError("eef/qpos/action/stamps/scales length mismatch")

    out_eef = [eef[0].copy()]
    out_qpos = [qpos[0].copy()]
    out_action = [action[0].copy()]
    out_stamps = [float(stamps[0])]
    out_img_idx = [int(image_index_src[0]) if image_index_src is not None else 0]
    out_extras: dict[str, list[np.ndarray]] = {k: [np.asarray(v[0]).copy()] for k, v in extras.items()}

    inserted = 0
    for i in range(n - 1):
        s = float(max(scales[i], scales[i + 1]))
        n_steps = max(1, int(round(s)))
        dt = float(stamps[i + 1] - stamps[i])
        if dt <= 1e-9:
            dt = 1e-3
        for step in range(1, n_steps + 1):
            alpha = step / float(n_steps)
            out_eef.append(_interp_eef_row(eef[i], eef[i + 1], alpha))
            out_qpos.append(_interp_joint_row(qpos[i], qpos[i + 1], alpha))
            out_action.append(_interp_joint_row(action[i], action[i + 1], alpha))
            out_stamps.append(out_stamps[-1] + dt)
            # Hold source image from the start of the interval (no pixel interp).
            src_i = i if step < n_steps else i + 1
            out_img_idx.append(int(image_index_src[src_i]) if image_index_src is not None else src_i)
            for k, arr in extras.items():
                a = np.asarray(arr[i], dtype=np.float64)
                b = np.asarray(arr[i + 1], dtype=np.float64)
                if a.ndim == 0:
                    out_extras[k].append(_lerp(a, b, alpha))
                elif k in {"master_deg"}:
                    out_extras[k].append(_interp_joint_row(a, b, alpha))
                else:
                    out_extras[k].append(_lerp(a, b, alpha))
            if n_steps > 1 and step < n_steps:
                inserted += 1

    result: dict[str, Any] = {
        "eef": np.asarray(out_eef, dtype=np.float64),
        "qpos": np.asarray(out_qpos, dtype=np.float64),
        "action": np.asarray(out_action, dtype=np.float64),
        "stamps": np.asarray(out_stamps, dtype=np.float64),
        "image_index": np.asarray(out_img_idx, dtype=np.int64),
        "inserted": inserted,
        "extras": {k: np.asarray(v, dtype=np.float64) for k, v in out_extras.items()},
    }
    return result


def _load_optional(f: h5py.File, key: str) -> np.ndarray | None:
    if key not in f:
        return None
    return np.asarray(f[key])


def optimize_episode(
    src: Path,
    dst: Path,
    *,
    do_slowdown: bool,
    do_adjust: bool,
    cfg: SingularityConfig,
    skip_images: bool = False,
) -> dict[str, Any]:
    src = src.resolve()
    dst = dst.resolve()
    if dst == src:
        raise ValueError("output path must differ from input (refuse overwrite)")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as f:
        qpos = np.asarray(f["observations/qpos"], dtype=np.float64)
        eef = np.asarray(f["observations/eef"], dtype=np.float64)
        action = np.asarray(f["action"], dtype=np.float64)
        stamps = (
            np.asarray(f["arm_stamp"], dtype=np.float64)
            if "arm_stamp" in f
            else np.asarray(f["time"], dtype=np.float64)
        )
        time_arr = _load_optional(f, "time")
        arm_stamp = _load_optional(f, "arm_stamp")
        action_eef = _load_optional(f, "action_eef")
        qvel = _load_optional(f, "observations/qvel")
        effort = _load_optional(f, "observations/effort")
        master_deg = _load_optional(f, "master_deg")
        eef_fk = _load_optional(f, "observations/eef_fk")
        fps = float(f.attrs["fps"]) if "fps" in f.attrs else 30.0
        task = str(f.attrs.get("task", "manipulation task"))
        camera_names: list[str] = []
        if "camera_names" in f.attrs:
            try:
                camera_names = json.loads(str(f.attrs["camera_names"]))
            except json.JSONDecodeError:
                camera_names = []

        images: dict[str, np.ndarray] = {}
        if not skip_images and "observations/images" in f:
            img_grp = f["observations/images"]
            keys = camera_names or list(img_grp.keys())
            for name in keys:
                if name in img_grp:
                    images[name] = np.asarray(img_grp[name])
            if not camera_names:
                camera_names = list(images.keys())

        depth: dict[str, np.ndarray] = {}
        if not skip_images and "observations/images_depth" in f:
            dgrp = f["observations/images_depth"]
            for name in dgrp.keys():
                depth[name] = np.asarray(dgrp[name])

        root_attrs = {k: f.attrs[k] for k in f.attrs.keys()}

    n0 = len(qpos)
    adjuster = SingularityAdjuster(cfg)
    danger0 = adjuster.detect(qpos)
    n_danger0 = int(np.sum(danger0))

    note_parts: list[str] = []
    scales = np.ones(n0, dtype=np.float64)
    if do_slowdown:
        scales = adjuster.period_scale(qpos)
        note_parts.append(
            f"slowdown×{cfg.slowdown_factor:g} expand={cfg.slowdown_expand_frames}"
        )

    eef_work = eef.copy()
    adj_stats = None
    if do_adjust:
        eef_work, adj_stats = adjuster.adjust(eef_work, qpos)
        note_parts.append(f"rpy_adjust:{adj_stats.summary()}")

    extras: dict[str, np.ndarray] = {}
    if qvel is not None:
        extras["qvel"] = np.asarray(qvel, dtype=np.float64)
    if effort is not None:
        extras["effort"] = np.asarray(effort, dtype=np.float64)
    if master_deg is not None:
        extras["master_deg"] = np.asarray(master_deg, dtype=np.float64)
    if eef_fk is not None:
        # Keep eef_fk time-aligned if present; do not invent FK-delta rewrite.
        extras["eef_fk"] = np.asarray(eef_fk, dtype=np.float64)
    if action_eef is not None:
        extras["action_eef"] = np.asarray(action_eef, dtype=np.float64)
        if do_adjust:
            extras["action_eef"] = eef_work.copy()

    image_index = np.arange(n0, dtype=np.int64)
    if do_slowdown and float(np.max(scales)) > 1.0 + 1e-9:
        dens = densify_near_danger(
            eef=eef_work,
            qpos=qpos,
            action=action,
            stamps=stamps,
            scales=scales,
            extras=extras,
            image_index_src=image_index,
        )
        eef_out = dens["eef"]
        qpos_out = dens["qpos"]
        action_out = dens["action"]
        stamps_out = dens["stamps"]
        img_idx = dens["image_index"]
        extras_out = dens["extras"]
        inserted = int(dens["inserted"])
        note_parts.append(f"densify+{inserted} frames")
    else:
        eef_out = eef_work
        qpos_out = qpos
        action_out = action
        stamps_out = stamps
        img_idx = image_index
        extras_out = extras
        inserted = 0
        if do_adjust and "action_eef" not in extras_out:
            extras_out = dict(extras_out)
            extras_out["action_eef"] = eef_work.copy()

    n1 = len(qpos_out)
    duration = float(stamps_out[-1] - stamps_out[0]) if n1 > 1 else 0.0
    mean_fps = (n1 - 1) / duration if duration > 1e-9 else fps

    # Build wall times: prefer stretch arm_stamp; keep time offset if both exist.
    arm_out = stamps_out.copy()
    if time_arr is not None and arm_stamp is not None and len(time_arr) == n0:
        # Map via same densify schedule already applied to stamps (arm preferred).
        time_out = stamps_out - stamps_out[0] + float(time_arr[0])
    elif time_arr is not None and len(time_arr) == n0 and not do_slowdown:
        time_out = np.asarray(time_arr, dtype=np.float64)
    else:
        time_out = stamps_out - stamps_out[0] + (
            float(time_arr[0]) if time_arr is not None and len(time_arr) else float(stamps[0])
        )

    with h5py.File(dst, "w") as root:
        for key, val in root_attrs.items():
            try:
                root.attrs[key] = val
            except TypeError:
                root.attrs[key] = str(val)
        root.attrs["task"] = task
        root.attrs["fps"] = float(mean_fps)
        root.attrs["filename"] = dst.name
        root.attrs["optimized"] = True
        root.attrs["optimizer"] = "optimize_hdf5_singularity"
        root.attrs["optimizer_note"] = "; ".join(note_parts) if note_parts else "noop"
        root.attrs["optimizer_src"] = src.name
        root.attrs["optimizer_frames_in"] = int(n0)
        root.attrs["optimizer_frames_out"] = int(n1)
        root.attrs["optimizer_danger_in"] = int(n_danger0)
        if adj_stats is not None:
            root.attrs["optimizer_adjust"] = adj_stats.summary()

        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos_out.astype(np.float32))
        obs.create_dataset("eef", data=eef_out.astype(np.float32))
        if "qvel" in extras_out:
            obs.create_dataset("qvel", data=extras_out["qvel"].astype(np.float32))
        if "effort" in extras_out:
            obs.create_dataset("effort", data=extras_out["effort"].astype(np.float32))
        if "eef_fk" in extras_out:
            obs.create_dataset("eef_fk", data=extras_out["eef_fk"].astype(np.float32))

        if images and not skip_images:
            img_grp = obs.create_group("images")
            names = camera_names or list(images.keys())
            root.attrs["camera_names"] = json.dumps(names)
            for name in names:
                if name not in images:
                    continue
                src_imgs = images[name]
                stacked = src_imgs[np.clip(img_idx, 0, len(src_imgs) - 1)]
                img_grp.create_dataset(
                    name, data=stacked.astype(np.uint8), compression="gzip", compression_opts=1
                )
        elif camera_names:
            root.attrs["camera_names"] = json.dumps(camera_names)

        if depth and not skip_images:
            dgrp = obs.create_group("images_depth")
            for name, arr in depth.items():
                stacked = arr[np.clip(img_idx, 0, len(arr) - 1)]
                dgrp.create_dataset(
                    name,
                    data=stacked.astype(arr.dtype),
                    compression="gzip",
                    compression_opts=4,
                )

        root.create_dataset("action", data=action_out.astype(np.float32))
        if "action_eef" in extras_out:
            root.create_dataset("action_eef", data=extras_out["action_eef"].astype(np.float32))
        else:
            root.create_dataset("action_eef", data=eef_out.astype(np.float32))
        if "master_deg" in extras_out:
            root.create_dataset("master_deg", data=extras_out["master_deg"].astype(np.float32))
        root.create_dataset("time", data=time_out.astype(np.float64))
        root.create_dataset("arm_stamp", data=arm_out.astype(np.float64))

    return {
        "src": str(src),
        "dst": str(dst),
        "frames_in": n0,
        "frames_out": n1,
        "inserted": inserted,
        "danger_in": n_danger0,
        "mean_fps": mean_fps,
        "note": "; ".join(note_parts),
        "adjust": adj_stats.summary() if adj_stats else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline HDF5 singularity optimizer (no robot required)."
    )
    p.add_argument("hdf5", type=Path, nargs="+", help="Input episode_*.hdf5 (one or more)")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (single input only). Default: <stem>_opt.hdf5",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for batch mode (default: beside each input)",
    )
    p.add_argument(
        "--mode",
        choices=("slowdown", "adjust", "both"),
        default="both",
        help="slowdown=densify near danger; adjust=RPY micro-nudge; both=default",
    )
    p.add_argument("--slowdown-factor", type=float, default=None)
    p.add_argument("--slowdown-expand-frames", type=int, default=None)
    p.add_argument("--max-rpy-adjust-deg", type=float, default=None)
    p.add_argument("--skip-images", action="store_true", help="Omit image datasets (smaller, replay-only)")
    p.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON with a 'singularity' object (e.g. teleop_uarm.example.json)",
    )
    return p.parse_args()


def _load_cfg(args: argparse.Namespace) -> SingularityConfig:
    data: dict[str, Any] = {}
    if args.config_json is not None:
        raw = json.loads(args.config_json.read_text(encoding="utf-8"))
        if isinstance(raw.get("replay"), dict) and isinstance(raw["replay"].get("singularity"), dict):
            data = dict(raw["replay"]["singularity"])
        elif isinstance(raw.get("singularity"), dict):
            data = dict(raw["singularity"])
    cfg = SingularityConfig.from_dict(data)
    if args.slowdown_factor is not None:
        cfg.slowdown_factor = float(args.slowdown_factor)
        cfg.slowdown_enabled = True
    if args.slowdown_expand_frames is not None:
        cfg.slowdown_expand_frames = int(args.slowdown_expand_frames)
        cfg.slowdown_enabled = True
    if args.max_rpy_adjust_deg is not None:
        cfg.max_rpy_adjust_rad = float(np.deg2rad(args.max_rpy_adjust_deg))
    return cfg


def main() -> None:
    args = parse_args()
    inputs = [p.resolve() for p in args.hdf5]
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"File not found: {p}")
    if args.output is not None and len(inputs) != 1:
        raise SystemExit("--output only valid with a single input file")

    cfg = _load_cfg(args)
    do_slowdown = args.mode in ("slowdown", "both")
    do_adjust = args.mode in ("adjust", "both")
    if do_slowdown:
        cfg.slowdown_enabled = True

    for src in inputs:
        if args.output is not None:
            dst = args.output.resolve()
        elif args.outdir is not None:
            args.outdir.mkdir(parents=True, exist_ok=True)
            dst = (args.outdir / f"{src.stem}_opt.hdf5").resolve()
        else:
            dst = src.with_name(f"{src.stem}_opt.hdf5")

        info = optimize_episode(
            src,
            dst,
            do_slowdown=do_slowdown,
            do_adjust=do_adjust,
            cfg=cfg,
            skip_images=bool(args.skip_images),
        )
        print(
            f"[ok] {src.name} -> {Path(info['dst']).name}  "
            f"frames {info['frames_in']}->{info['frames_out']}  "
            f"danger={info['danger_in']}  fps≈{info['mean_fps']:.1f}  "
            f"({info['note'] or 'noop'})",
            flush=True,
        )
        if info.get("adjust"):
            print(f"     adjust: {info['adjust']}", flush=True)


if __name__ == "__main__":
    main()
