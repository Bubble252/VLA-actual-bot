"""Frame sources: live demo / bus / future HDF5 replay."""

from __future__ import annotations

import time
from typing import Optional, Protocol

import numpy as np

from . import frame_bus
from .types import VizFrame


class FrameSource(Protocol):
    def next(self) -> Optional[VizFrame]:
        ...


class DemoSource:
    """Synthetic q_cmd for MuJoCo-only smoke test; optional still images (P3)."""

    def __init__(
        self,
        *,
        hz: float = 30.0,
        n_joints: int = 6,
        cam_high: Optional[np.ndarray] = None,
        cam_wrist: Optional[np.ndarray] = None,
        demo_images_dir: Optional[str] = None,
    ):
        self.hz = hz
        self.n_joints = n_joints
        self._t0 = time.time()
        self._step = 0
        if cam_high is None and cam_wrist is None:
            try:
                from .image_panel import resolve_demo_image_pair

                cam_high, cam_wrist = resolve_demo_image_pair(demo_images_dir)
            except Exception as exc:
                print(f"[sources] demo images skipped: {exc}")
                cam_high, cam_wrist = None, None
        self.cam_high = cam_high
        self.cam_wrist = cam_wrist

    def next(self) -> VizFrame:
        t = time.time() - self._t0
        q = np.zeros(self.n_joints, dtype=np.float64)
        q[0] = 0.4 * np.sin(0.6 * t)
        q[1] = 0.35 * np.sin(0.45 * t + 0.3)
        q[2] = 0.25 * np.sin(0.5 * t + 1.0)
        q[3] = 0.3 * np.sin(0.7 * t + 0.5)
        q[4] = 0.4 * np.sin(0.55 * t + 0.8)
        q[5] = 0.5 * np.sin(0.65 * t)
        self._step += 1
        return VizFrame(
            q_cmd=q,
            cam_high=self.cam_high,
            cam_wrist=self.cam_wrist,
            meta={"step": self._step, "source": "demo", "t": t},
        )


class BusSource:
    """Read latest frame published by eval client."""

    def __init__(self, path=frame_bus.DEFAULT_BUS_PATH):
        self.path = path
        self._last_mtime: Optional[float] = None

    def next(self) -> Optional[VizFrame]:
        frame = frame_bus.load(self.path)
        return frame


def RosStateSource(*args, **kwargs):
    """Lazy import so --demo works without ROS sourced."""
    from .ros_state_source import RosStateSource as _Impl

    return _Impl(*args, **kwargs)


class Hdf5ReplaySource:
    """Reserved for teleop r/p/v HDF5 replay UI (not in v1)."""

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def next(self) -> Optional[VizFrame]:
        raise NotImplementedError(
            "Hdf5ReplaySource is an interface stub only; "
            "HDF5 r/p/v replay visualization is deferred. "
            "Use --hdf5 for static reference frame (P4), not full replay."
        )
