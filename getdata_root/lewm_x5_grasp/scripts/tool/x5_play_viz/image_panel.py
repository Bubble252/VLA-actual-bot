"""OpenCV side window for cam_high / cam_wrist (P3).

Does not open RealSense; only displays frames from the bus / demo / HDF5.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_TOOL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_IMAGES_DIR = _TOOL_ROOT / "assets" / "demo_images"
WINDOW_NAME = "x5_play_viz cameras"


def _try_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError:
        return None


def ensure_demo_images(out_dir: Path = DEFAULT_DEMO_IMAGES_DIR) -> Tuple[Path, Path]:
    """Create labeled placeholder JPEGs if missing (for --demo 灌图)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    high_p = out_dir / "step0000_cam_high.jpg"
    wrist_p = out_dir / "step0000_cam_wrist.jpg"
    if high_p.is_file() and wrist_p.is_file():
        return high_p, wrist_p

    cv2 = _try_cv2()
    for path, label, color in (
        (high_p, "cam_high (demo)", (40, 90, 180)),
        (wrist_p, "cam_wrist (demo)", (40, 160, 90)),
    ):
        if path.is_file():
            continue
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:] = color
        # checker hint
        img[::20, :, :] = np.clip(img[::20, :, :] + 30, 0, 255)
        img[:, ::20, :] = np.clip(img[:, ::20, :] + 20, 0, 255)
        if cv2 is not None:
            cv2.putText(
                img,
                label,
                (24, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(path), img)
        else:
            # Minimal PPM fallback then skip jpeg — write raw via numpy only as .npy
            # Prefer writing a tiny ppm renamed for loaders that accept jpg via PIL later.
            try:
                from PIL import Image, ImageDraw

                pil = Image.fromarray(img[:, :, ::-1] if False else img)
                draw = ImageDraw.Draw(pil)
                draw.text((24, 100), label, fill=(255, 255, 255))
                pil.save(path, format="JPEG", quality=85)
            except ImportError:
                # last resort: save npz sibling; loaders will generate solid color
                np.save(path.with_suffix(".npy"), img)
                print(
                    f"[image_panel] wrote {path.with_suffix('.npy')} "
                    "(install opencv-python or Pillow for JPEG demo)",
                    file=sys.stderr,
                )
    return high_p, wrist_p


def load_rgb(path: Path) -> Optional[np.ndarray]:
    """Load HxWx3 RGB uint8 from jpg/png/npy."""
    if not path.is_file():
        npy = path.with_suffix(".npy")
        if npy.is_file():
            arr = np.load(npy)
            return np.asarray(arr, dtype=np.uint8)
        return None
    cv2 = _try_cv2()
    if cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def resolve_demo_image_pair(
    images_dir: Optional[str | Path] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find cam_high / cam_wrist stills for demo mode."""
    root = Path(images_dir).expanduser() if images_dir else DEFAULT_DEMO_IMAGES_DIR
    if images_dir is None:
        ensure_demo_images(root)

    def _pick(patterns: tuple[str, ...]) -> Optional[Path]:
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
        return None

    if not root.is_dir():
        print(f"[image_panel] demo images dir missing: {root}", file=sys.stderr)
        return None, None

    high_p = _pick(
        (
            "step0000_cam_high.jpg",
            "*cam_high*.jpg",
            "*cam_high*.png",
            "*agent*.jpg",
        )
    )
    wrist_p = _pick(
        (
            "step0000_cam_wrist.jpg",
            "*cam_wrist*.jpg",
            "*cam_wrist*.png",
            "*wrist*.jpg",
        )
    )
    high = load_rgb(high_p) if high_p else None
    wrist = load_rgb(wrist_p) if wrist_p else None
    if high is not None:
        print(f"[image_panel] demo cam_high <- {high_p}")
    if wrist is not None:
        print(f"[image_panel] demo cam_wrist <- {wrist_p}")
    return high, wrist


def _placeholder(h: int, w: int, title: str) -> np.ndarray:
    img = np.full((h, w, 3), 36, dtype=np.uint8)
    img[8:h - 8, 8:w - 8] = 48
    cv2 = _try_cv2()
    if cv2 is not None:
        cv2.putText(
            img,
            title,
            (16, h // 2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "waiting for frames",
            (16, h // 2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
    return img


def _to_bgr_panel(
    rgb: Optional[np.ndarray],
    *,
    title: str,
    target_h: int = 240,
) -> np.ndarray:
    cv2 = _try_cv2()
    if rgb is None or rgb.size == 0:
        ph = _placeholder(target_h, int(target_h * 4 / 3), title)
        return cv2.cvtColor(ph, cv2.COLOR_RGB2BGR) if cv2 is not None else ph

    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    h, w = arr.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    if cv2 is not None:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            bgr,
            title,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return bgr

    # no cv2: return RGB resized crudely
    ys = (np.linspace(0, h - 1, target_h)).astype(int)
    xs = (np.linspace(0, w - 1, new_w)).astype(int)
    return arr[ys][:, xs]


class ImageSidePanel:
    """Non-blocking OpenCV window: left=cam_high, right=cam_wrist."""

    def __init__(self, *, enabled: bool = True, window_name: str = WINDOW_NAME):
        self.enabled = enabled
        self.window_name = window_name
        self._cv2 = _try_cv2() if enabled else None
        self._opened = False
        if enabled and self._cv2 is None:
            print(
                "[image_panel] opencv-python not installed; skip camera side window. "
                "pip3 install opencv-python-headless",
                file=sys.stderr,
            )
            self.enabled = False

    def update(
        self,
        cam_high: Optional[np.ndarray],
        cam_wrist: Optional[np.ndarray],
        *,
        badge: str = "",
    ) -> None:
        if not self.enabled or self._cv2 is None:
            return
        cv2 = self._cv2
        left = _to_bgr_panel(cam_high, title=f"cam_high {badge}".strip())
        right = _to_bgr_panel(cam_wrist, title=f"cam_wrist {badge}".strip())
        # match heights
        h = max(left.shape[0], right.shape[0])
        if left.shape[0] != h:
            left = cv2.resize(left, (left.shape[1], h))
        if right.shape[0] != h:
            right = cv2.resize(right, (right.shape[1], h))
        gap = np.full((h, 6, 3), 20, dtype=np.uint8)
        canvas = np.concatenate([left, gap, right], axis=1)
        if not self._opened:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, canvas.shape[1], canvas.shape[0])
            self._opened = True
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._opened and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._opened = False
