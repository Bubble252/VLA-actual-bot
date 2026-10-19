"""MuJoCo Play Viz entrypoint.

Examples:
  cd lewm_x5_grasp/scripts/tool
  python -m x5_play_viz --demo
  python -m x5_play_viz --bus
  python -m x5_play_viz --demo --hdf5 /path/to/episode.hdf5 --ref-frame 0
  python -m x5_play_viz --demo --headless --steps 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from . import frame_bus
from .axes import draw_axes, draw_base_axes, draw_ghost_arm, xyzrpy_to_T
from .eef_offset import apply_eef_xyz_offset
from .hdf5_ref import Hdf5Ref, load_hdf5_ref
from .image_panel import ImageSidePanel
from .mj_model import load_x5
from .sources import BusSource, DemoSource, RosStateSource
from .types import VizFrame


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="X5 Play MuJoCo visualizer")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="synthetic q_cmd (default)")
    mode.add_argument("--bus", action="store_true", help="read /tmp/x5_play_viz/latest.npz")
    mode.add_argument(
        "--ros",
        action="store_true",
        help="subscribe /arm_status (joint feedback → FK)",
    )
    p.add_argument("--urdf", type=str, default=None, help="override URDF/MJCF path")
    p.add_argument("--ee-body", type=str, default="link6", help="MuJoCo body for EE axes")
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--axis-len", type=float, default=0.10)
    p.add_argument("--show-sdk", action="store_true", help="draw SDK end_pos axes if present")
    p.add_argument(
        "--no-floor",
        action="store_true",
        help="disable checker floor / sky scene (plain URDF)",
    )
    p.add_argument(
        "--hdf5",
        type=str,
        default=None,
        help="HDF5 episode for ghost arm + optional capture images (P4)",
    )
    p.add_argument("--ref-frame", type=int, default=0)
    p.add_argument(
        "--no-ghost",
        action="store_true",
        help="with --hdf5, only draw ref EE axes (skip translucent arm)",
    )
    p.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OpenCV side window for cam_high/cam_wrist (default: on in GUI)",
    )
    p.add_argument(
        "--demo-images",
        type=str,
        default=None,
        help="dir with step0000_cam_high.jpg / cam_wrist (default: assets/demo_images)",
    )
    p.add_argument("--bus-path", type=str, default=str(frame_bus.DEFAULT_BUS_PATH))
    p.add_argument("--headless", action="store_true", help="no GUI; print EE for N steps")
    p.add_argument("--steps", type=int, default=0, help="with --headless, stop after N steps")
    p.add_argument("--print-every", type=int, default=30)
    return p.parse_args(argv)


def _apply_frame(xm, frame: VizFrame) -> None:
    q = frame.joints_for_fk()
    if q is not None:
        xm.set_arm_qpos(q)


def _pick_panel_images(
    frame: VizFrame,
    hdf5_ref: Optional[Hdf5Ref],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Prefer live/bus/demo images; fall back to HDF5 capture stills."""
    if frame.cam_high is not None or frame.cam_wrist is not None:
        src = str(frame.meta.get("source", "live"))
        return frame.cam_high, frame.cam_wrist, src
    if hdf5_ref is not None and hdf5_ref.has_images:
        return hdf5_ref.cam_high, hdf5_ref.cam_wrist, "hdf5"
    return None, None, "waiting"


def _draw_overlay(
    viewer,
    xm,
    frame: VizFrame,
    *,
    axis_len: float,
    show_sdk: bool,
    ref_T: Optional[np.ndarray],
    ghost_data: Optional[mujoco.MjData],
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    draw_base_axes(scn, length=axis_len * 1.4)
    if ghost_data is not None:
        draw_ghost_arm(scn, ghost_data, xm.link_body_ids)
        draw_axes(
            scn,
            xm.ee_T(ghost_data),
            length=axis_len * 0.75,
            width=0.007,
            alpha=0.55,
        )
    elif ref_T is not None:
        draw_axes(scn, ref_T, length=axis_len * 0.7, width=0.008, alpha=0.55)
    draw_axes(scn, xm.ee_T(), length=axis_len, width=0.012, alpha=1.0)
    if show_sdk and frame.end_pos is not None:
        # SDK triad: sdk_end_pos xyz + YAML eef_xyz_offset_m (scripts/x5_play_params.yaml).
        ep_draw = apply_eef_xyz_offset(frame.end_pos)
        draw_axes(
            scn,
            xyzrpy_to_T(ep_draw),
            length=axis_len * 1.55,
            width=0.005,
            alpha=0.85,
        )


def _setup_camera(viewer) -> None:
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -22.0
    viewer.cam.distance = 1.15
    viewer.cam.lookat[:] = [0.18, 0.0, 0.12]


def run(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    use_bus = bool(args.bus)
    use_ros = bool(args.ros)
    if not use_bus and not use_ros and not args.demo:
        args.demo = True

    xm = load_x5(args.urdf, ee_body=args.ee_body, with_floor=not args.no_floor)
    print(f"[viz] urdf={xm.urdf_path}  floor={'off' if args.no_floor else 'on'}")
    print(f"[viz] arm joints={xm.joint_names()}  ee_body_id={xm.ee_body_id}")

    hdf5_ref: Optional[Hdf5Ref] = None
    ref_T: Optional[np.ndarray] = None
    ghost_data: Optional[mujoco.MjData] = None
    if args.hdf5:
        hdf5_ref = load_hdf5_ref(args.hdf5, args.ref_frame)
        if hdf5_ref.has_qpos:
            if args.no_ghost:
                xm.set_arm_qpos(hdf5_ref.qpos)
                ref_T = xm.ee_T().copy()
                print(f"[viz] ref EE xyzrpy={xm.ee_xyzrpy()} (axes only)")
            else:
                ghost_data = xm.make_data()
                xm.set_arm_qpos(hdf5_ref.qpos, data=ghost_data)
                print(
                    f"[viz] ghost arm from hdf5 qpos; "
                    f"EE xyzrpy={xm.ee_xyzrpy(ghost_data)}"
                )
        elif hdf5_ref.eef is not None:
            ref_T = xyzrpy_to_T(hdf5_ref.eef)
            print("[viz] no qpos; drawing HDF5 eef axes only")

    source = None
    if use_ros:
        source = RosStateSource()
        if not args.show_sdk:
            args.show_sdk = True  # useful default when watching real arm
        print("[viz] mode=ros (/arm_status → joint FK; SDK axes on)")
    elif use_bus:
        source = BusSource(Path(args.bus_path))
        print(f"[viz] mode=bus path={args.bus_path}")
    else:
        source = DemoSource(
            hz=args.hz,
            n_joints=xm.n_arm,
            demo_images_dir=args.demo_images,
        )
        print("[viz] mode=demo (synthetic q_cmd → MuJoCo FK + demo images)")

    show_images = bool(args.images) and not args.headless
    panel = ImageSidePanel(enabled=show_images)
    if show_images:
        print("[viz] OpenCV camera side window enabled (cam_high | cam_wrist)")

    dt = 1.0 / max(args.hz, 1e-3)
    step = 0
    last_print = 0

    try:
        if args.headless:
            while True:
                frame = source.next()
                if frame is None:
                    frame = VizFrame(meta={"waiting": True, "source": "waiting"})
                else:
                    _apply_frame(xm, frame)
                step += 1
                if step - last_print >= args.print_every or step == 1:
                    ee = xm.ee_xyzrpy()
                    hi, wr, badge = _pick_panel_images(frame, hdf5_ref)
                    print(
                        f"[viz] step={frame.meta.get('step', step)} "
                        f"ee_xyz=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                        f"rpy=({ee[3]:.2f},{ee[4]:.2f},{ee[5]:.2f}) "
                        f"imgs={badge} "
                        f"high={'Y' if hi is not None else 'N'} "
                        f"wrist={'Y' if wr is not None else 'N'}"
                    )
                    last_print = step
                if args.steps and step >= args.steps:
                    return 0
                time.sleep(dt)

        import mujoco.viewer

        print("[viz] RGB axes = XYZ | EE = q_cmd/joint FK | cyan ghost = HDF5 ref")
        print("[viz] close the MuJoCo window to exit")
        with mujoco.viewer.launch_passive(xm.model, xm.data) as viewer:
            _setup_camera(viewer)
            while viewer.is_running():
                t0 = time.time()
                frame = source.next()
                if frame is None:
                    frame = VizFrame(meta={"waiting": True, "source": "waiting"})
                else:
                    _apply_frame(xm, frame)
                _draw_overlay(
                    viewer,
                    xm,
                    frame,
                    axis_len=args.axis_len,
                    show_sdk=args.show_sdk,
                    ref_T=ref_T,
                    ghost_data=ghost_data,
                )
                hi, wr, badge = _pick_panel_images(frame, hdf5_ref)
                panel.update(hi, wr, badge=badge)
                viewer.sync()
                step += 1
                if step == 1 or step % max(int(args.hz), 1) == 0:
                    ee = xm.ee_xyzrpy()
                    src = frame.meta.get(
                        "source",
                        "ros" if use_ros else ("bus" if use_bus else "demo"),
                    )
                    print(
                        f"\r[viz] {src} step={frame.meta.get('step', step)} "
                        f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) "
                        f"imgs={badge}   ",
                        end="",
                        flush=True,
                    )
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        print()
        return 0
    finally:
        panel.close()
        close = getattr(source, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def main(argv: Optional[list[str]] = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
