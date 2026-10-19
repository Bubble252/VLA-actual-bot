#!/usr/bin/env python3
"""Batch bake observations/eef_fk and/or EEF (mode=4) replay for teleop HDF5 episodes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import rclpy

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from episode_replay import (  # noqa: E402
    ensure_eef_fk_baked,
    hdf5_has_eef_fk,
    replay_hdf5_file,
)
from lewm_x5_grasp.arm_ros2 import X5ArmBridge  # noqa: E402
from lewm_x5_grasp.singularity_adjuster import SingularityConfig  # noqa: E402

Phase = Literal["bake", "replay", "both"]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_episodes(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("episode_*.hdf5"))


def replay_settings_from_config(cfg: dict) -> Dict[str, Any]:
    replay_cfg = cfg.get("replay") or {}
    replay_hz = float(replay_cfg.get("hz", 50.0))
    replay_eef_ik_seed = str(replay_cfg.get("eef_ik_seed", "action"))
    replay_orientation_interp = str(replay_cfg.get("orientation_interp", "slerp"))
    replay_eef_source = str(replay_cfg.get("eef_source", "observations"))
    return {
        "replay_hz": replay_hz,
        "replay_common": {
            "replay_hz": replay_hz,
            "smooth_window": int(replay_cfg.get("smooth_window", 7)),
            "blend_sec": float(replay_cfg.get("blend_sec", 0.5)),
            "smooth_gripper": bool(replay_cfg.get("smooth_gripper", False)),
            "max_blend_sec": float(replay_cfg.get("max_blend_sec", 2.0)),
            "blend_skip_tol": float(replay_cfg.get("blend_skip_tol_rad", 0.05)),
            "blend_speed_rad_s": float(replay_cfg.get("blend_speed_rad_s", 0.8)),
            "post_hold_sec": float(replay_cfg.get("post_hold_sec", 1.0)),
            "return_to_start": bool(replay_cfg.get("return_to_start_after_replay", True)),
            "return_blend_sec": float(replay_cfg.get("return_to_start_blend_sec", 1.5)),
            "return_max_blend_sec": float(replay_cfg.get("return_to_start_max_blend_sec", 6.0)),
            "return_settle_sec": float(replay_cfg.get("return_to_start_settle_sec", 1.5)),
            "return_hold_sec": float(replay_cfg.get("return_to_start_hold_sec", 2.5)),
            "return_tol_rad": float(replay_cfg.get("return_to_start_tol_rad", 0.04)),
            "start_max_blend_sec": float(replay_cfg.get("start_max_blend_sec", 6.0)),
            "start_settle_sec": float(replay_cfg.get("start_settle_sec", 0.8)),
            "start_snap_sec": float(
                replay_cfg.get("start_snap_sec", replay_cfg.get("homing_lock_sec", 2.5))
            ),
        },
        "replay_eef": {
            "eef_smooth_window": int(replay_cfg.get("eef_smooth_window", 15)),
            "eef_max_rpy_step_rad": float(replay_cfg.get("eef_max_rpy_step_rad", 0.04)),
            "eef_ik_seed": replay_eef_ik_seed,  # type: ignore[typeddict-item]
            "orientation_interp": replay_orientation_interp,  # type: ignore[typeddict-item]
            "eef_source": replay_eef_source,  # type: ignore[typeddict-item]
            "eef_ik_seed_smooth_window": int(replay_cfg.get("eef_ik_seed_smooth_window", 15)),
            "eef_quat_smooth_window": int(replay_cfg.get("eef_quat_smooth_window", 7)),
            "eef_joint_lead_sec": float(replay_cfg.get("eef_joint_lead_sec", 0.0)),
            "eef_joint_fallback_enabled": bool(
                replay_cfg.get("eef_joint_fallback_enabled", False)
            ),
            "eef_joint_fallback_step_rad": float(
                replay_cfg.get("eef_joint_fallback_step_deg", 0.85)
            )
            * (np.pi / 180.0),
            "eef_joint_fallback_hold_sec": float(
                replay_cfg.get("eef_joint_fallback_hold_sec", 0.8)
            ),
            "eef_joint_fallback_lookahead_sec": float(
                replay_cfg.get("eef_joint_fallback_lookahead_sec", 0.2)
            ),
            "singularity_adjust": bool(replay_cfg.get("singularity_adjust", False)),
            "singularity_joint_fallback": bool(
                replay_cfg.get("singularity_joint_fallback", False)
            ),
            "eef_ik_seed_closed_loop": bool(replay_cfg.get("eef_ik_seed_closed_loop", True)),
            "eef_ik_seed_hint_blend": float(replay_cfg.get("eef_ik_seed_hint_blend", 0.0)),
            "singularity_config": SingularityConfig.from_dict(
                replay_cfg.get("singularity")
                if isinstance(replay_cfg.get("singularity"), dict)
                else None
            ),
        },
        "bake": {
            "replay_hz": float(replay_cfg.get("bake_hz", 100.0)),
            "smooth_window": int(replay_cfg.get("bake_smooth_window", 15)),
            "quat_smooth_window": int(replay_cfg.get("bake_quat_smooth_window", 7)),
            "blend_sec": float(replay_cfg.get("blend_sec", 0.5)),
            "max_blend_sec": float(replay_cfg.get("max_blend_sec", 2.0)),
            "return_to_start": bool(replay_cfg.get("return_to_start_after_replay", True)),
            "return_blend_sec": float(replay_cfg.get("return_to_start_blend_sec", 1.5)),
            "return_max_blend_sec": float(replay_cfg.get("return_to_start_max_blend_sec", 6.0)),
            "return_blend_speed_rad_s": float(replay_cfg.get("blend_speed_rad_s", 0.8)),
            "return_settle_sec": float(replay_cfg.get("return_to_start_settle_sec", 1.5)),
            "return_hold_sec": float(replay_cfg.get("return_to_start_hold_sec", 2.5)),
            "return_tol_rad": float(replay_cfg.get("return_to_start_tol_rad", 0.04)),
        },
        "auto_bake": bool(replay_cfg.get("auto_bake_before_eef_replay", False)),
        "force_rebake": bool(replay_cfg.get("force_rebake_eef_fk", False)),
        "eef_ik_seed": replay_eef_ik_seed,
        "orientation_interp": replay_orientation_interp,
        "eef_source": replay_eef_source,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch bake eef_fk and/or EEF replay for episode_*.hdf5 in a directory."
    )
    p.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("data/teleop_demos"),
        help="Directory containing episode_*.hdf5 (default: data/teleop_demos).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("config/teleop_uarm.example.json"),
        help="Teleop config JSON (replay section drives bake/replay params).",
    )
    p.add_argument(
        "--phase",
        choices=("both", "bake", "replay"),
        default="replay",
        help="replay: EEF replay only (default); bake: bake eef_fk only; both: bake then replay.",
    )
    p.add_argument(
        "--force-rebake",
        action="store_true",
        help="Re-bake eef_fk even when observations/eef_fk already exists.",
    )
    p.add_argument(
        "--skip-missing-fk",
        action="store_true",
        help="In replay-only mode, skip episodes without eef_fk instead of failing.",
    )
    p.add_argument(
        "--pause-sec",
        type=float,
        default=3.0,
        help="Pause between episodes (default: 3 s). Use 0 for no pause.",
    )
    p.add_argument(
        "--step",
        action="store_true",
        help="Wait for Enter before each episode (after bake/replay).",
    )
    p.add_argument(
        "--from-index",
        type=int,
        default=0,
        help="Start from this 0-based index in sorted episode list.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="Max number of episodes to process (default: all from --from-index).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List episodes and exit without moving the arm.",
    )
    return p.parse_args()


def _wait_between(
    idx: int,
    total: int,
    *,
    pause_sec: float,
    step: bool,
) -> bool:
    """Return False if user wants to abort the batch."""
    if idx >= total - 1:
        return True
    if step:
        try:
            ans = input(
                f"\n[{idx + 1}/{total}] 按 Enter 继续下一条，输入 q 退出: "
            ).strip().lower()
        except EOFError:
            return False
        if ans in {"q", "quit"}:
            return False
        return True
    if pause_sec > 0:
        print(f"\n>> 下一条 {idx + 2}/{total} 将在 {pause_sec:.1f}s 后开始…", flush=True)
        time.sleep(pause_sec)
    return True


def run_batch_bake(
    bridge: X5ArmBridge,
    episodes: List[Path],
    settings: Dict[str, Any],
    *,
    force_rebake: bool,
    pause_sec: float,
    step: bool,
) -> None:
    bake_kw = settings["bake"]
    total = len(episodes)
    print(f"\n{'=' * 60}", flush=True)
    print(f"批量 Bake: {total} 条 episode → observations/eef_fk", flush=True)
    print(f"  force_rebake={force_rebake}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for i, path in enumerate(episodes):
        print(f"\n--- [{i + 1}/{total}] bake {path.name} ---", flush=True)
        if hdf5_has_eef_fk(path) and not force_rebake:
            print("  已有 eef_fk，跳过（用 --force-rebake 覆盖）", flush=True)
        else:
            ensure_eef_fk_baked(
                bridge,
                path,
                auto_bake=settings["auto_bake"],
                eef_source=settings["eef_source"],  # type: ignore[arg-type]
                force_rebake=force_rebake,
                **bake_kw,
            )
        if not _wait_between(i, total, pause_sec=pause_sec, step=step):
            print(">> 用户中止批量 bake", flush=True)
            break


def run_batch_replay(
    bridge: X5ArmBridge,
    episodes: List[Path],
    settings: Dict[str, Any],
    *,
    skip_missing_fk: bool,
    pause_sec: float,
    step: bool,
) -> None:
    common = settings["replay_common"]
    eef_kw = settings["replay_eef"]
    replay_hz = settings["replay_hz"]
    total = len(episodes)
    print(f"\n{'=' * 60}", flush=True)
    print(
        f"批量末端回放: {total} 条 @ {replay_hz:.0f} Hz "
        f"(seed={settings['eef_ik_seed']}, orient={settings['orientation_interp']}, "
        f"eef={settings['eef_source']})",
        flush=True,
    )
    print(f"{'=' * 60}\n", flush=True)

    for i, path in enumerate(episodes):
        print(f"\n--- [{i + 1}/{total}] EEF replay {path.name} ---", flush=True)
        needs_fk = settings["eef_source"] == "eef_fk"
        if needs_fk and not hdf5_has_eef_fk(path):
            msg = f"  缺少 observations/eef_fk: {path.name}"
            if skip_missing_fk:
                print(msg + "，跳过", flush=True)
                continue
            raise SystemExit(
                msg + "\n  先运行: bash scripts/run_batch_eef_replay.sh --phase bake"
                " 或 config 设 eef_source=observations"
            )

        try:
            replay_hdf5_file(
                bridge,
                path,
                joint_source="action",
                control_mode="eef",
                **common,
                **eef_kw,
            )
        except KeyboardInterrupt:
            print("\n>> 批量回放已取消", flush=True)
            break

        if not _wait_between(i, total, pause_sec=pause_sec, step=step):
            print(">> 用户中止批量回放", flush=True)
            break


def main() -> None:
    args = parse_args()
    script_dir = _SCRIPT_DIR
    project_dir = script_dir.parent

    config_path = args.config
    if not config_path.is_absolute():
        config_path = (project_dir / config_path).resolve()
    cfg = load_config(config_path)
    settings = replay_settings_from_config(cfg)
    force_rebake = args.force_rebake or settings["force_rebake"]

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = (project_dir / input_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    episodes = discover_episodes(input_dir)
    if args.from_index > 0:
        episodes = episodes[args.from_index :]
    if args.count is not None:
        episodes = episodes[: max(0, args.count)]

    if not episodes:
        raise SystemExit(f"No episode_*.hdf5 under {input_dir}")

    print("=" * 60, flush=True)
    print(f"Input : {input_dir}")
    print(f"Config: {config_path}")
    print(f"Phase : {args.phase}")
    print(f"Episodes ({len(episodes)}):")
    for i, ep in enumerate(episodes):
        fk = "eef_fk OK" if hdf5_has_eef_fk(ep) else "no eef_fk"
        print(f"  [{i + 1}] {ep.name}  ({fk})")
    print("=" * 60, flush=True)

    if args.dry_run:
        return

    mode4_guard_cfg = dict(cfg.get("mode4_guard") or {"enabled": True})
    print("\n>> 批量任务：从臂将连续运动，请保持主臂/U-Arm 不动", flush=True)
    print(">> 3 秒后开始…", flush=True)
    time.sleep(3.0)

    rclpy.init()
    bridge = X5ArmBridge(mode4_guard=mode4_guard_cfg)
    try:
        bridge.wait_for_state(timeout_sec=30.0)
        if args.phase in {"both", "bake"}:
            run_batch_bake(
                bridge,
                episodes,
                settings,
                force_rebake=force_rebake,
                pause_sec=args.pause_sec if args.phase == "both" else args.pause_sec,
                step=args.step,
            )
        if args.phase in {"both", "replay"}:
            run_batch_replay(
                bridge,
                episodes,
                settings,
                skip_missing_fk=args.skip_missing_fk,
                pause_sec=args.pause_sec,
                step=args.step,
            )
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("\n>> 批量任务完成", flush=True)


if __name__ == "__main__":
    main()
