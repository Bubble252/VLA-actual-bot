from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .schemas import Observation, PolicyOutput
from .transforms import transform_point, xyzrpy_to_matrix


class BasePolicy:
    def predict(self, observation: Observation) -> PolicyOutput:
        raise NotImplementedError


class ScriptedPickPolicy(BasePolicy):
    def __init__(self, config: Dict[str, Any]):
        task = config["task"]
        self.pregrasp = task["pregrasp_xyzrpy"]
        self.grasp = task["grasp_xyzrpy"]
        self.lift = task["lift_xyzrpy"]
        self.open_gripper = float(task["open_gripper"])
        self.close_gripper = float(task["close_gripper"])

    def predict(self, observation: Observation) -> PolicyOutput:
        phase = observation.phase
        if phase == "approach":
            return PolicyOutput(target_xyzrpy=list(self.pregrasp), gripper=self.open_gripper)
        if phase == "descend":
            return PolicyOutput(target_xyzrpy=list(self.grasp), gripper=self.open_gripper)
        if phase == "close":
            return PolicyOutput(target_xyzrpy=list(self.grasp), gripper=self.close_gripper)
        if phase == "lift":
            return PolicyOutput(target_xyzrpy=list(self.lift), gripper=self.close_gripper)
        return PolicyOutput(target_xyzrpy=list(self.lift), gripper=self.close_gripper, phase="done")


class ImportedPolicyAdapter(BasePolicy):
    def __init__(self, config: Dict[str, Any]):
        policy_cfg = config["policy"]
        module_path = Path(policy_cfg["module_path"]).expanduser().resolve()
        class_name = policy_cfg["class_name"]
        init_kwargs = policy_cfg.get("init", {})

        spec = importlib.util.spec_from_file_location("user_lewm_policy", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        self.impl = cls(init_kwargs)

    def predict(self, observation: Observation) -> PolicyOutput:
        raw = self.impl.predict(observation.to_dict())
        target = raw["target_xyzrpy"]
        if len(target) != 6:
            raise ValueError("Policy output target_xyzrpy must have length 6.")
        return PolicyOutput(
            target_xyzrpy=[float(v) for v in target],
            gripper=float(raw.get("gripper", 0.0)),
            phase=raw.get("phase"),
        )


class ClickDepthGraspPolicy(BasePolicy):
    def __init__(self, config: Dict[str, Any]):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for click_depth_grasp.") from exc

        task = config["task"]
        grasp_cfg = config.get("visual_grasp", {})

        self.cv2 = cv2
        self.window_name = grasp_cfg.get("window_name", "D405 grasp target")
        self.open_gripper = float(task["open_gripper"])
        self.close_gripper = float(task["close_gripper"])
        self.camera_in_ee_xyzrpy = grasp_cfg.get("camera_in_ee_xyzrpy", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.grasp_rpy = grasp_cfg.get("grasp_rpy", task["grasp_xyzrpy"][3:6])
        self.depth_patch = int(grasp_cfg.get("depth_patch", 7))
        self.min_depth_m = float(grasp_cfg.get("min_depth_m", 0.05))
        self.max_depth_m = float(grasp_cfg.get("max_depth_m", 0.8))
        self.pregrasp_height = float(grasp_cfg.get("pregrasp_height", 0.08))
        self.grasp_z_offset = float(grasp_cfg.get("grasp_z_offset", 0.0))
        self.lift_height = float(grasp_cfg.get("lift_height", 0.12))
        self._clicked_uv: Optional[Tuple[int, int]] = None
        self._target_xyz: Optional[np.ndarray] = None
        self._target_pixel: Optional[Tuple[int, int]] = None
        self._image_width: Optional[int] = None
        self._last_message = "Click target on RGB or depth, press r to reset"

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == self.cv2.EVENT_LBUTTONDOWN:
            width = self._image_width or x + 1
            self._clicked_uv = (int(x % width), int(y))

    def _show(self, observation: Observation) -> None:
        if observation.image is None:
            return

        image_bgr = observation.image[:, :, ::-1].copy()
        self._image_width = image_bgr.shape[1]
        depth_bgr = np.zeros_like(image_bgr)
        if observation.depth_image is not None:
            depth_bgr = self.cv2.convertScaleAbs(observation.depth_image, alpha=0.03)
            depth_bgr = self.cv2.applyColorMap(depth_bgr, self.cv2.COLORMAP_JET)

        if self._target_pixel is not None:
            for panel in (image_bgr, depth_bgr):
                self.cv2.drawMarker(
                    panel,
                    self._target_pixel,
                    (0, 255, 0),
                    markerType=self.cv2.MARKER_CROSS,
                    markerSize=24,
                    thickness=2,
                )

        self.cv2.putText(
            image_bgr,
            "RGB",
            (12, image_bgr.shape[0] - 14),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            depth_bgr,
            "DEPTH",
            (12, depth_bgr.shape[0] - 14),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        combined = np.hstack((image_bgr, depth_bgr))
        self.cv2.putText(
            combined,
            self._last_message,
            (12, 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.imshow(self.window_name, combined)
        key = self.cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            self._target_xyz = None
            self._target_pixel = None
            self._last_message = "Target reset, click target on RGB or depth"

    def _sample_depth_m(self, depth_image: np.ndarray, u: int, v: int, depth_scale: float) -> Optional[float]:
        radius = max(1, self.depth_patch // 2)
        y0 = max(0, v - radius)
        y1 = min(depth_image.shape[0], v + radius + 1)
        x0 = max(0, u - radius)
        x1 = min(depth_image.shape[1], u + radius + 1)
        patch = depth_image[y0:y1, x0:x1].astype(float)
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return None
        depth_m = float(np.median(valid) * depth_scale)
        if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
            return None
        return depth_m

    def _try_set_target(self, observation: Observation) -> None:
        if self._clicked_uv is None:
            return
        if observation.depth_image is None or observation.camera_info is None:
            self._last_message = "No depth/camera info"
            self._clicked_uv = None
            return

        u, v = self._clicked_uv
        self._clicked_uv = None
        camera_info = observation.camera_info
        depth_scale = float(camera_info.get("depth_scale", 0.001))
        depth_m = self._sample_depth_m(observation.depth_image, u, v, depth_scale)
        if depth_m is None:
            self._last_message = "Invalid depth, click again"
            return

        fx = float(camera_info["fx"])
        fy = float(camera_info["fy"])
        ppx = float(camera_info["ppx"])
        ppy = float(camera_info["ppy"])
        point_camera = np.array(
            [
                (float(u) - ppx) * depth_m / fx,
                (float(v) - ppy) * depth_m / fy,
                depth_m,
            ],
            dtype=float,
        )

        base_from_ee = xyzrpy_to_matrix(observation.arm_state.end_pos[:6])
        ee_from_camera = xyzrpy_to_matrix(self.camera_in_ee_xyzrpy)
        point_base = transform_point(base_from_ee @ ee_from_camera, point_camera)

        self._target_xyz = point_base
        self._target_pixel = (u, v)
        self._last_message = f"Target xyz {point_base[0]:.3f} {point_base[1]:.3f} {point_base[2]:.3f}"
        print(
            "[click_depth_grasp] target",
            f"uv=({u},{v})",
            f"depth={depth_m:.3f}m",
            f"camera=({point_camera[0]:.3f},{point_camera[1]:.3f},{point_camera[2]:.3f})",
            f"base=({point_base[0]:.3f},{point_base[1]:.3f},{point_base[2]:.3f})",
            flush=True,
        )

    def _target_pose(self, xyz: np.ndarray) -> List[float]:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2]), *[float(v) for v in self.grasp_rpy]]

    def predict(self, observation: Observation) -> PolicyOutput:
        self._show(observation)
        self._try_set_target(observation)

        current = observation.arm_state.end_pos[:6].astype(float).tolist()
        if self._target_xyz is None:
            return PolicyOutput(target_xyzrpy=current, gripper=self.open_gripper, phase="wait")

        grasp_xyz = self._target_xyz.copy()
        grasp_xyz[2] += self.grasp_z_offset
        pregrasp_xyz = grasp_xyz.copy()
        pregrasp_xyz[2] += self.pregrasp_height
        lift_xyz = grasp_xyz.copy()
        lift_xyz[2] += self.lift_height

        phase = observation.phase
        if phase == "wait":
            return PolicyOutput(target_xyzrpy=self._target_pose(pregrasp_xyz), gripper=self.open_gripper, phase="approach")
        if phase == "approach":
            return PolicyOutput(target_xyzrpy=self._target_pose(pregrasp_xyz), gripper=self.open_gripper)
        if phase == "descend":
            return PolicyOutput(target_xyzrpy=self._target_pose(grasp_xyz), gripper=self.open_gripper)
        if phase == "close":
            return PolicyOutput(target_xyzrpy=self._target_pose(grasp_xyz), gripper=self.close_gripper)
        if phase == "lift":
            return PolicyOutput(target_xyzrpy=self._target_pose(lift_xyz), gripper=self.close_gripper)
        return PolicyOutput(target_xyzrpy=self._target_pose(lift_xyz), gripper=self.close_gripper, phase="done")


def build_policy(config: Dict[str, Any]) -> BasePolicy:
    policy_type = config["policy"]["type"]
    if policy_type == "scripted_pick":
        return ScriptedPickPolicy(config)
    if policy_type == "click_depth_grasp":
        return ClickDepthGraspPolicy(config)
    if policy_type == "imported":
        return ImportedPolicyAdapter(config)
    raise ValueError(f"Unsupported policy type: {policy_type}")


def clip_xyzrpy(xyzrpy: List[float], workspace: Dict[str, Any]) -> List[float]:
    result = np.asarray(xyzrpy, dtype=float).copy()
    min_xyz = np.asarray(workspace["min_xyz"], dtype=float)
    max_xyz = np.asarray(workspace["max_xyz"], dtype=float)
    result[:3] = np.clip(result[:3], min_xyz, max_xyz)
    return result.tolist()
