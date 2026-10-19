from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Prefer GTK over Qt for cv2.imshow (Qt backend often black-screens on Ubuntu/Wayland).
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "GTK,FFMPEG")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def list_realsense_devices(
    retries: int = 3,
    retry_delay_sec: float = 1.5,
) -> List[Tuple[str, str]]:
    """Return [(model_name, serial_number), ...] for connected RealSense devices."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    last_exc: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            devices: List[Tuple[str, str]] = []
            for dev in rs.context().query_devices():
                try:
                    name = dev.get_info(rs.camera_info.name)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    devices.append((name, serial))
                except RuntimeError as exc:
                    print(f"[RealSense] skip one device during enumerate: {exc}", flush=True)
            return devices
        except RuntimeError as exc:
            last_exc = exc
            if attempt + 1 < retries:
                print(
                    f"[RealSense] query_devices failed ({exc}); "
                    f"retry in {retry_delay_sec:.1f}s ({attempt + 2}/{retries})…",
                    flush=True,
                )
                time.sleep(retry_delay_sec)

    raise RuntimeError(
        "RealSense USB 通信失败（常见于上次异常退出后相机未释放）。"
        "请先: pkill -f teleop_record.py ; 等待 2s ; 拔插相机 USB；"
        "或在 config/camera 里设置 wrist_serial / desktop_serial 跳过重扫。"
    ) from last_exc


def autodetect_realsense_serial(model_hint: str) -> Optional[str]:
    hint = model_hint.upper()
    try:
        devices = list_realsense_devices()
    except RuntimeError:
        return None
    for name, serial in devices:
        if hint in name.upper():
            return serial
    return None


def resolve_realsense_serial(
    *,
    provided: Optional[str],
    model_hint: str,
    label: str,
) -> str:
    """Use explicit serial from config/CLI; fall back to auto-detect."""
    if provided:
        print(f"[{label}] using serial from config/CLI: {provided}", flush=True)
        return str(provided)
    serial = autodetect_realsense_serial(model_hint)
    if serial:
        print(f"[{label}] auto-detected serial: {serial}", flush=True)
        return serial
    raise RuntimeError(
        f"{label}: no {model_hint} found. "
        f"Set wrist_serial/desktop_serial in config or use --wrist-serial/--desktop-serial."
    )


def lerobot_video_key(camera_name: str) -> str:
    return f"observation.images.{camera_name}"


# RealSense rs.option.power_line_frequency: 0=off, 1=50Hz, 2=60Hz, 3=auto
DEFAULT_ANTI_FLICKER_HZ = 50
DEFAULT_CAMERA_WIDTH = 480
DEFAULT_CAMERA_HEIGHT = 480
ANTI_FLICKER_HZ_TO_RS = {0: 0, 50: 1, 60: 2, 3: 3}


def _resolve_realsense_stream_size(output_w: int, output_h: int) -> Tuple[int, int]:
    """Map desired output size to a RealSense-native color stream."""
    if output_w == 480 and output_h == 480:
        return 640, 480
    return output_w, output_h


def _list_supported_color_fps(serial: str, stream_w: int, stream_h: int) -> List[int]:
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []
    supported: set[int] = set()
    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) != serial:
            continue
        for sensor in dev.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != rs.stream.color:
                    continue
                vp = profile.as_video_stream_profile()
                if vp.width() == stream_w and vp.height() == stream_h:
                    supported.add(int(vp.fps()))
        break
    return sorted(supported)


def _resolve_stream_fps(serial: str, stream_w: int, stream_h: int, fps: int, *, label: str = "") -> int:
    """Snap requested fps to a RealSense-native mode (e.g. 20 -> 30 or 15)."""
    supported = _list_supported_color_fps(serial, stream_w, stream_h)
    if not supported:
        return fps
    if fps in supported:
        return fps
    # Prefer the smallest native fps still >= requested (minimal oversampling).
    higher = [f for f in supported if f >= fps]
    resolved = higher[0] if higher else supported[-1]
    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}stream fps {fps} unsupported at {stream_w}x{stream_h}; "
        f"using {resolved} (supported: {supported}). "
        f"Data record rate is controlled by record.hz, not camera fps.",
        flush=True,
    )
    return resolved


def _crop_frame_to_size(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    if w < target_w or h < target_h:
        raise ValueError(f"Cannot crop {w}x{h} to {target_w}x{target_h}")
    x0 = (w - target_w) // 2
    y0 = (h - target_h) // 2
    cropped = frame[y0 : y0 + target_h, x0 : x0 + target_w]
    return cropped.copy() if cropped.base is frame else cropped


def _apply_power_line_frequency(
    device: Any,
    rs: Any,
    anti_flicker_hz: int,
    *,
    label: str = "",
    quiet: bool = False,
) -> Dict[str, float]:
    """Apply anti-flicker on every sensor that supports it (D405 stereo + D435 RGB)."""
    rs_val = ANTI_FLICKER_HZ_TO_RS.get(int(anti_flicker_hz), int(anti_flicker_hz))
    hz_text = {0: "off", 50: "50Hz", 60: "60Hz", 3: "auto"}.get(int(anti_flicker_hz), f"{anti_flicker_hz}Hz")
    prefix = f"[{label}] " if label else ""
    applied: Dict[str, float] = {}
    for sensor in device.query_sensors():
        try:
            if not sensor.supports(rs.option.power_line_frequency):
                continue
            name = sensor.get_info(rs.camera_info.name)
            sensor.set_option(rs.option.power_line_frequency, float(rs_val))
            actual = float(sensor.get_option(rs.option.power_line_frequency))
            applied[name] = actual
            actual_text = {0.0: "off", 1.0: "50Hz", 2.0: "60Hz", 3.0: "auto"}.get(actual, str(actual))
            if not quiet:
                print(f"{prefix}anti-flicker {name}: {actual_text} (requested {hz_text})", flush=True)
        except Exception as exc:
            if not quiet:
                print(f"{prefix}anti-flicker set failed: {exc}", flush=True)
    return applied


class FileImageSource:
    def __init__(self, path: str):
        self.path = Path(path)

    def read(self) -> Optional[np.ndarray]:
        if not self.path.exists():
            return None
        if self.path.suffix == ".npy":
            return np.load(self.path)
        if self.path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if Image is None:
                raise RuntimeError("Pillow is required to read image files.")
            return np.asarray(Image.open(self.path).convert("RGB"))
        raise ValueError(f"Unsupported image format: {self.path}")

    def close(self) -> None:
        return None


class RealSenseCamera:
    """Single RealSense device (select by serial)."""

    def __init__(
        self,
        *,
        serial: str,
        label: str,
        width: int = DEFAULT_CAMERA_WIDTH,
        height: int = DEFAULT_CAMERA_HEIGHT,
        fps: int = 30,
        enable_depth: bool = False,
        save_path: Optional[str] = None,
        display: bool = False,
        align_depth: bool = True,
        anti_flicker_hz: Optional[int] = DEFAULT_ANTI_FLICKER_HZ,
        auto_exposure: Optional[bool] = False,
        exposure_us: Optional[float] = 10000.0,
        gain: Optional[float] = 64.0,
        auto_white_balance: Optional[bool] = False,
        white_balance: Optional[float] = 4600.0,
    ):
        self.serial = str(serial)
        self.label = str(label)
        self.width = int(width)
        self.height = int(height)
        self._stream_width, self._stream_height = _resolve_realsense_stream_size(
            self.width, self.height
        )
        self.fps = _resolve_stream_fps(
            self.serial,
            self._stream_width,
            self._stream_height,
            int(fps),
            label=self.label,
        )
        self.enable_depth = bool(enable_depth)
        self.save_path = Path(save_path).expanduser() if save_path else None
        self.display = bool(display)
        self.align_depth = bool(align_depth) and self.enable_depth
        self.anti_flicker_hz = anti_flicker_hz
        self.auto_exposure = auto_exposure
        self.exposure_us = exposure_us
        self.gain = gain
        self.auto_white_balance = auto_white_balance
        self.white_balance = white_balance
        self._pipeline = None
        self._align = None
        self._cv2 = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_display_bgr: Optional[np.ndarray] = None
        self._last_depth_vis: Optional[np.ndarray] = None
        self._camera_info: Dict[str, Any] = {}
        self._imaging_configured = False
        self._start()

    @property
    def last_depth(self) -> Optional[np.ndarray]:
        return self._last_depth

    @property
    def camera_info(self) -> Dict[str, Any]:
        return dict(self._camera_info)

    def _start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for RealSense cameras. "
                "Install librealsense/pyrealsense2."
            ) from exc

        self._rs = rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self._stream_width, self._stream_height, rs.format.bgr8, self.fps
        )
        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth, self._stream_width, self._stream_height, rs.format.z16, self.fps
            )
        profile = pipeline.start(config)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.align_depth else None

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        info: Dict[str, Any] = {
            "label": self.label,
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "stream_width": self._stream_width,
            "stream_height": self._stream_height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "width_requested": self.width,
            "height_requested": self.height,
            "fps_requested": self.fps,
            "anti_flicker_hz": self.anti_flicker_hz,
        }
        if self.enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            info["depth_scale"] = depth_sensor.get_depth_scale()
            info["depth_units"] = "meter"
        self._camera_info = info

        if self.display:
            self._cv2 = self._try_init_display()

    def configure_imaging(self) -> None:
        """Apply exposure / anti-flicker after the stream is running (avoids USB timeouts)."""
        if self._imaging_configured or self._pipeline is None:
            return
        device = self._pipeline.get_active_profile().get_device()
        info = self._camera_info
        color_sensor = self._find_color_sensor(device, self._rs)
        if color_sensor is None:
            info["color_options"] = "unsupported"
            print(f"[{self.label}] color options: unsupported", flush=True)
            self._imaging_configured = True
            return
        try:
            info["color_sensor"] = color_sensor.get_info(self._rs.camera_info.name)
        except Exception:
            info["color_sensor"] = "unknown"
        self._apply_anti_flicker(device, info)
        self._configure_exposure_and_wb(color_sensor, info)
        self._apply_anti_flicker(device, info)
        if not self.auto_exposure:
            self._apply_manual_exposure(color_sensor, info)
        self._imaging_configured = True
        self._camera_info = info
        if not self.auto_exposure:
            print(
                f"[{self.label}] manual 50Hz lock: exp={info.get('exposure_us')}us "
                f"gain={info.get('gain')} fps={self.fps} plf={info.get('anti_flicker_sensors')}",
                flush=True,
            )

    def _set_sensor_option(
        self,
        sensor,
        option,
        value: float,
        info_key: str,
        label: str,
        info: Dict[str, Any],
        *,
        quiet: bool = False,
    ) -> None:
        try:
            if not sensor.supports(option):
                info[info_key] = None
                if not quiet:
                    print(f"[{self.label}] {label}: unsupported", flush=True)
                return
            option_range = sensor.get_option_range(option)
            clamped = min(max(float(value), float(option_range.min)), float(option_range.max))
            sensor.set_option(option, clamped)
            actual = float(sensor.get_option(option))
            info[info_key] = actual
            if abs(actual - round(actual)) < 1e-6:
                actual_text = str(int(round(actual)))
            else:
                actual_text = f"{actual:.3f}"
            if not quiet:
                if abs(clamped - float(value)) > 1e-6:
                    print(f"[{self.label}] {label}: {actual_text} (requested {value}, clamped)", flush=True)
                else:
                    print(f"[{self.label}] {label}: {actual_text}", flush=True)
        except Exception as exc:
            info[info_key] = "set_failed"
            if not quiet:
                print(f"[{self.label}] {label} set failed: {exc}", flush=True)

    def _find_color_sensor(self, device, rs):
        """Prefer RGB (D435) or Stereo Module (D405); fall back to any color-capable sensor."""
        rgb_sensor = None
        stereo_sensor = None
        fallback_sensor = None
        wanted_options = (
            rs.option.power_line_frequency,
            rs.option.enable_auto_exposure,
            rs.option.exposure,
            rs.option.gain,
            rs.option.enable_auto_white_balance,
            rs.option.white_balance,
        )
        for sensor in device.query_sensors():
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                name = ""
            if "RGB" in name.upper():
                rgb_sensor = sensor
                break
            try:
                if "STEREO" in name.upper() and (
                    sensor.supports(rs.option.enable_auto_exposure)
                    or sensor.supports(rs.option.power_line_frequency)
                ):
                    stereo_sensor = sensor
                if (
                    sensor.supports(rs.option.white_balance)
                    or sensor.supports(rs.option.enable_auto_white_balance)
                ):
                    if fallback_sensor is None:
                        fallback_sensor = sensor
                elif fallback_sensor is None and any(sensor.supports(option) for option in wanted_options):
                    fallback_sensor = sensor
            except Exception:
                continue
        return rgb_sensor or stereo_sensor or fallback_sensor

    def _apply_anti_flicker(self, device, info: Dict[str, Any], *, quiet: bool = False) -> None:
        if self.anti_flicker_hz is None:
            info["anti_flicker_hz"] = "default"
            if not quiet:
                print(f"[{self.label}] anti-flicker: driver default", flush=True)
            return
        applied = _apply_power_line_frequency(
            device,
            self._rs,
            int(self.anti_flicker_hz),
            label=self.label,
            quiet=quiet,
        )
        info["anti_flicker_hz"] = int(self.anti_flicker_hz)
        info["anti_flicker_sensors"] = applied
        if not applied and not quiet:
            print(f"[{self.label}] anti-flicker: no supported sensor", flush=True)

    def _configure_exposure_and_wb(self, color_sensor, info: Dict[str, Any]) -> None:
        """Configure auto/manual exposure and white balance on the color sensor."""
        rs = self._rs
        use_auto_exposure = bool(self.auto_exposure)
        if self.auto_exposure is not None:
            self._set_sensor_option(
                color_sensor,
                rs.option.enable_auto_exposure,
                1.0 if use_auto_exposure else 0.0,
                "auto_exposure",
                "auto exposure",
                info,
            )
        if not use_auto_exposure:
            self._apply_manual_exposure(color_sensor, info)

        auto_wb = self.auto_white_balance
        if auto_wb is None and use_auto_exposure:
            auto_wb = True
        if auto_wb is not None:
            self._set_sensor_option(
                color_sensor,
                rs.option.enable_auto_white_balance,
                1.0 if auto_wb else 0.0,
                "auto_white_balance",
                "auto white balance",
                info,
            )
        if auto_wb is False and self.white_balance is not None:
            self._set_sensor_option(
                color_sensor,
                rs.option.white_balance,
                float(self.white_balance),
                "white_balance",
                "white balance",
                info,
            )

    def _apply_manual_exposure(self, color_sensor, info: Dict[str, Any], *, quiet: bool = False) -> None:
        rs = self._rs
        if self.exposure_us is not None:
            self._set_sensor_option(
                color_sensor,
                rs.option.exposure,
                float(self.exposure_us),
                "exposure_us",
                "exposure us",
                info,
                quiet=quiet,
            )
        if self.gain is not None:
            self._set_sensor_option(
                color_sensor,
                rs.option.gain,
                float(self.gain),
                "gain",
                "gain",
                info,
                quiet=quiet,
            )

    def stabilize_imaging(self, *, quiet: bool = False) -> None:
        """Re-lock anti-flicker / manual exposure after stream warmup (D405 50Hz)."""
        if self._pipeline is None:
            return
        if not self._imaging_configured:
            self.configure_imaging()
            return
        device = self._pipeline.get_active_profile().get_device()
        patch: Dict[str, Any] = {}
        self._apply_anti_flicker(device, patch, quiet=quiet)
        color_sensor = self._find_color_sensor(device, self._rs)
        if color_sensor is not None and not self.auto_exposure:
            self._apply_manual_exposure(color_sensor, patch, quiet=quiet)
        self._camera_info.update(patch)

    def _configure_color_sensor(self, profile, info: Dict[str, Any]) -> None:
        """Deprecated: kept for compatibility; use _configure_exposure_and_wb."""
        device = profile.get_device()
        color_sensor = self._find_color_sensor(device, self._rs)
        if color_sensor is None:
            info["color_options"] = "unsupported"
            print(f"[{self.label}] color options: unsupported", flush=True)
            return
        self._configure_exposure_and_wb(color_sensor, info)

    def _try_init_display(self):
        try:
            import cv2
        except ImportError:
            print(f"[{self.label}] cv2 not installed; preview disabled.")
            self.display = False
            return None
        try:
            cv2.namedWindow(f"{self.label} color", cv2.WINDOW_NORMAL)
            if self.enable_depth:
                cv2.namedWindow(f"{self.label} depth", cv2.WINDOW_NORMAL)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
        except cv2.error as exc:
            print(f"[{self.label}] OpenCV has no GUI; preview disabled. ({exc})")
            self.display = False
            return None
        return cv2

    def read(self, timeout_ms: int = 200, *, pump_display: bool = True) -> Optional[np.ndarray]:
        if self._pipeline is None:
            return None

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            return None
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        if not color_frame:
            return None

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1].copy()
        raw_depth = None if not depth_frame else np.asanyarray(depth_frame.get_data()).copy()
        self._last_depth = (
            None
            if raw_depth is None
            else _crop_frame_to_size(raw_depth, self.width, self.height)
        )

        color_rgb = _crop_frame_to_size(color_rgb, self.width, self.height)
        color_bgr = color_rgb[:, :, ::-1]
        self._last_display_bgr = color_bgr

        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.save_path, color_rgb)

        if self._last_depth is not None and self.display and self._cv2 is not None:
            depth_vis = self._cv2.convertScaleAbs(self._last_depth, alpha=0.03)
            self._last_depth_vis = self._cv2.applyColorMap(depth_vis, self._cv2.COLORMAP_JET)
        else:
            self._last_depth_vis = None

        if pump_display:
            self.show_preview()
        return color_rgb

    def show_preview(self) -> None:
        """Update OpenCV windows (call from main thread only)."""
        if not self.display or self._cv2 is None:
            return
        if self._last_display_bgr is not None:
            self._cv2.imshow(f"{self.label} color", self._last_display_bgr)
        if self._last_depth_vis is not None:
            self._cv2.imshow(f"{self.label} depth", self._last_depth_vis)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow(f"{self.label} color")
                if self.enable_depth:
                    self._cv2.destroyWindow(f"{self.label} depth")
            except Exception:
                pass
            self._cv2 = None


class MultiRealSenseSource:
    """Read RGB from multiple RealSense cameras (e.g. D405 wrist + D435i desktop)."""

    def __init__(self, cameras: List[RealSenseCamera]):
        if not cameras:
            raise ValueError("At least one RealSense camera is required.")
        self._cameras = cameras
        self.camera_names = [c.label for c in cameras]

    def read(self, timeout_ms: int = 200, *, pump_display: bool = False) -> Dict[str, Optional[np.ndarray]]:
        out: Dict[str, Optional[np.ndarray]] = {}
        for cam in self._cameras:
            out[cam.label] = cam.read(timeout_ms=timeout_ms, pump_display=False)
        if pump_display:
            self.show_previews()
        return out

    def show_previews(self) -> int:
        """Refresh all preview windows; call from main thread. Returns cv2.waitKey code."""
        cv2_mod = None
        for cam in self._cameras:
            cam.show_preview()
            if cam._cv2 is not None:
                cv2_mod = cam._cv2
        if cv2_mod is not None:
            raw = cv2_mod.waitKey(1)
            if raw == -1:
                return -1
            return int(raw) & 0xFF
        return -1

    def read_all_ready(self, timeout_ms: int = 200) -> Optional[Dict[str, np.ndarray]]:
        frames = self.read(timeout_ms=timeout_ms)
        if any(v is None for v in frames.values()):
            return None
        return {k: v for k, v in frames.items() if v is not None}

    @property
    def camera_info(self) -> Dict[str, Dict[str, Any]]:
        return {c.label: c.camera_info for c in self._cameras}

    def close(self) -> None:
        for cam in self._cameras:
            cam.close()

    def stabilize_imaging(self) -> None:
        for cam in self._cameras:
            cam.stabilize_imaging()


class CameraPoller:
    """Background thread: read RealSense frames; preview on main thread via show_previews()."""

    def __init__(self, cameras: MultiRealSenseSource):
        self._cameras = cameras
        self._lock = threading.Lock()
        self._latest: Dict[str, Optional[np.ndarray]] = {n: None for n in cameras.camera_names}
        self._wrist_depth: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._last_stabilize = time.monotonic()
        self._stabilize_interval_sec = 15.0
        self._thread = threading.Thread(target=self._run, name="camera-poller", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            frames = self._cameras.read(timeout_ms=200, pump_display=False)
            wrist = self._cameras._cameras[0] if self._cameras._cameras else None
            with self._lock:
                for name, frame in frames.items():
                    if frame is not None:
                        self._latest[name] = frame.copy()
                if wrist is not None and wrist.last_depth is not None:
                    self._wrist_depth = wrist.last_depth.copy()
            now = time.monotonic()
            if now - self._last_stabilize >= self._stabilize_interval_sec:
                for cam in self._cameras._cameras:
                    if cam.auto_exposure is False:
                        cam.stabilize_imaging(quiet=True)
                self._last_stabilize = now
            time.sleep(0.002)

    def snapshot(self) -> Tuple[Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        with self._lock:
            frames = {n: (f.copy() if f is not None else None) for n, f in self._latest.items()}
            depth = None if self._wrist_depth is None else self._wrist_depth.copy()
        return frames, depth

    def show_previews(self) -> int:
        return self._cameras.show_previews()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class RealSenseD405ImageSource:
    """Backward-compatible single D405 source (returns RGB ndarray)."""

    def __init__(
        self,
        width: int = DEFAULT_CAMERA_WIDTH,
        height: int = DEFAULT_CAMERA_HEIGHT,
        fps: int = 30,
        save_path: Optional[str] = None,
        display: bool = False,
        align_depth: bool = True,
        serial: Optional[str] = None,
        anti_flicker_hz: Optional[int] = DEFAULT_ANTI_FLICKER_HZ,
        auto_exposure: Optional[bool] = None,
        exposure_us: Optional[float] = None,
        gain: Optional[float] = None,
        auto_white_balance: Optional[bool] = None,
        white_balance: Optional[float] = None,
    ):
        resolved_serial = serial or autodetect_realsense_serial("D405")
        if not resolved_serial:
            raise RuntimeError("No Intel RealSense D405 found. Check USB connection.")
        cam_kwargs: Dict[str, Any] = dict(
            serial=resolved_serial,
            label="wrist",
            width=width,
            height=height,
            fps=fps,
            enable_depth=True,
            save_path=save_path,
            display=display,
            align_depth=align_depth,
            anti_flicker_hz=anti_flicker_hz,
        )
        if auto_exposure is not None:
            cam_kwargs["auto_exposure"] = auto_exposure
        if exposure_us is not None:
            cam_kwargs["exposure_us"] = exposure_us
        if gain is not None:
            cam_kwargs["gain"] = gain
        if auto_white_balance is not None:
            cam_kwargs["auto_white_balance"] = auto_white_balance
        if white_balance is not None:
            cam_kwargs["white_balance"] = white_balance
        self._cam = RealSenseCamera(**cam_kwargs)

    @property
    def last_depth(self) -> Optional[np.ndarray]:
        return self._cam.last_depth

    @property
    def camera_info(self) -> Dict[str, Any]:
        return self._cam.camera_info

    def read(self, timeout_ms: int = 200) -> Optional[np.ndarray]:
        return self._cam.read(timeout_ms=timeout_ms)

    def close(self) -> None:
        self._cam.close()


def build_record_cameras(
    *,
    wrist_serial: Optional[str],
    desktop_serial: Optional[str],
    enable_desktop: bool,
    width: int,
    height: int,
    fps: int,
    display: bool,
    anti_flicker_hz: int = DEFAULT_ANTI_FLICKER_HZ,
    auto_exposure: Optional[bool] = True,
    wrist_enable_depth: bool = False,
    wrist_overrides: Optional[Dict[str, Any]] = None,
    desktop_overrides: Optional[Dict[str, Any]] = None,
) -> MultiRealSenseSource:
    wrist = resolve_realsense_serial(
        provided=wrist_serial,
        model_hint="D405",
        label="cam_wrist",
    )
    if not wrist:
        raise RuntimeError("D405 wrist camera not found.")

    cam_common: Dict[str, Any] = dict(
        width=width,
        height=height,
        fps=fps,
        display=display,
        anti_flicker_hz=anti_flicker_hz,
    )
    if auto_exposure is not None:
        cam_common["auto_exposure"] = auto_exposure
        if auto_exposure:
            cam_common["exposure_us"] = None
            cam_common["gain"] = None
            cam_common["auto_white_balance"] = True
            cam_common["white_balance"] = None

    def _merge_camera_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not overrides:
            return dict(base)
        merged = dict(base)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        if merged.get("auto_exposure"):
            merged.setdefault("exposure_us", None)
            merged.setdefault("gain", None)
            merged.setdefault("auto_white_balance", True)
            merged.setdefault("white_balance", None)
        return merged

    cameras: List[RealSenseCamera] = [
        RealSenseCamera(
            serial=wrist,
            label="cam_wrist",
            enable_depth=wrist_enable_depth,
            align_depth=wrist_enable_depth,
            **_merge_camera_overrides(cam_common, wrist_overrides),
        )
    ]

    desktop = resolve_realsense_serial(
        provided=desktop_serial,
        model_hint="D435",
        label="cam_high",
    ) if enable_desktop else None
    if enable_desktop:
        if not desktop:
            raise RuntimeError(
                "Desktop camera enabled but no D435/D435i found. "
                "Use --no-desktop-camera to record wrist only."
            )
        time.sleep(1.0)
        cameras.append(
            RealSenseCamera(
                serial=desktop,
                label="cam_high",
                enable_depth=False,
                align_depth=False,
                **_merge_camera_overrides(cam_common, desktop_overrides),
            )
        )

    print("RealSense cameras:", flush=True)
    for cam in cameras:
        info = cam.camera_info
        print(
            f"  {cam.label}: serial={cam.serial}, "
            f"{cam.width}x{cam.height}@{cam.fps}fps, stream={cam._stream_width}x{cam._stream_height}, "
            f"anti_flicker={info.get('anti_flicker_hz')}Hz {info.get('anti_flicker_sensors', {})}, "
            f"auto_exp={info.get('auto_exposure')}, exp={info.get('exposure_us')}, gain={info.get('gain')}",
            flush=True,
        )
    return MultiRealSenseSource(cameras)


def build_teleop_cameras_from_config(
    camera_cfg: dict,
    *,
    enable_desktop: bool = True,
    wrist_serial: Optional[str] = None,
    desktop_serial: Optional[str] = None,
    wrist_enable_depth: bool = False,
) -> MultiRealSenseSource:
    """D405 wrist (cam_wrist) + optional D435 desktop (cam_high) for teleop record."""
    auto_exp = camera_cfg.get("auto_exposure", True)
    return build_record_cameras(
        wrist_serial=wrist_serial or camera_cfg.get("wrist_serial"),
        desktop_serial=desktop_serial or camera_cfg.get("desktop_serial"),
        enable_desktop=bool(camera_cfg.get("enable_desktop", enable_desktop)),
        width=int(camera_cfg.get("width", DEFAULT_CAMERA_WIDTH)),
        height=int(camera_cfg.get("height", DEFAULT_CAMERA_HEIGHT)),
        fps=int(camera_cfg.get("fps", 30)),
        display=bool(camera_cfg.get("display", True)),
        anti_flicker_hz=int(camera_cfg.get("anti_flicker_hz", DEFAULT_ANTI_FLICKER_HZ)),
        auto_exposure=auto_exp if auto_exp is not None else True,
        wrist_enable_depth=wrist_enable_depth,
        wrist_overrides=camera_cfg.get("wrist"),
        desktop_overrides=camera_cfg.get("desktop"),
    )


def warmup_cameras(cameras: MultiRealSenseSource, n_frames: int = 30) -> Dict[str, Optional[np.ndarray]]:
    """Warm up all RealSense devices; return last frame per camera."""
    last: Dict[str, Optional[np.ndarray]] = {name: None for name in cameras.camera_names}
    n = max(1, n_frames)
    configure_after = min(8, max(3, n // 4))
    print(f"  warmup: {n} frames, configure imaging after frame {configure_after}…", flush=True)
    for i in range(n):
        frames = cameras.read(timeout_ms=300, pump_display=False)
        if i + 1 == configure_after:
            for cam in cameras._cameras:
                cam.configure_imaging()
        for name, frame in frames.items():
            if frame is not None:
                last[name] = frame
        if (i + 1) % 10 == 0 or i + 1 == configure_after or i == 0:
            stats = " ".join(
                (
                    f"{name}=mean{float(last[name].mean()):.0f}"
                    if last[name] is not None
                    else f"{name}=waiting"
                )
                for name in cameras.camera_names
            )
            print(f"  warmup {i + 1}/{n}: {stats}", flush=True)
    for cam in cameras._cameras:
        if not cam._imaging_configured:
            cam.configure_imaging()
    cameras.stabilize_imaging()
    for _ in range(5):
        frames = cameras.read(timeout_ms=300, pump_display=False)
        for name, frame in frames.items():
            if frame is not None:
                last[name] = frame
    return last


def build_d405_from_config(camera_cfg: dict) -> RealSenseD405ImageSource:
    """Construct D405 source from teleop/collect JSON camera section."""
    kwargs: Dict[str, Any] = {
        "width": int(camera_cfg.get("width", DEFAULT_CAMERA_WIDTH)),
        "height": int(camera_cfg.get("height", DEFAULT_CAMERA_HEIGHT)),
        "fps": int(camera_cfg.get("fps", 30)),
        "display": bool(camera_cfg.get("display", True)),
        "align_depth": bool(camera_cfg.get("align_depth", True)),
    }
    if camera_cfg.get("serial"):
        kwargs["serial"] = str(camera_cfg["serial"])
    for key in (
        "anti_flicker_hz",
        "auto_exposure",
        "exposure_us",
        "gain",
        "auto_white_balance",
        "white_balance",
    ):
        if key in camera_cfg:
            kwargs[key] = camera_cfg[key]
    return RealSenseD405ImageSource(**kwargs)


def warmup_camera(camera: RealSenseD405ImageSource, n_frames: int = 30) -> Optional[np.ndarray]:
    """Discard initial frames (auto-exposure settle) and return last RGB."""
    last: Optional[np.ndarray] = None
    for _ in range(max(1, n_frames)):
        last = camera.read(timeout_ms=500)
    return last


def build_image_source(config: Dict[str, Any]):
    image_cfg = config.get("image", {})
    image_type = image_cfg.get("type", "file")
    if image_type == "file":
        return FileImageSource(image_cfg["path"])
    if image_type in {"realsense_d405", "d405"}:
        return RealSenseD405ImageSource(
            width=image_cfg.get("width", DEFAULT_CAMERA_WIDTH),
            height=image_cfg.get("height", DEFAULT_CAMERA_HEIGHT),
            fps=image_cfg.get("fps", 30),
            save_path=image_cfg.get("save_path"),
            display=image_cfg.get("display", False),
            align_depth=image_cfg.get("align_depth", True),
            serial=image_cfg.get("serial"),
        )
    raise ValueError(f"Unsupported image source type: {image_type}")
