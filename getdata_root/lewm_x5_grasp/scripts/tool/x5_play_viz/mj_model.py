"""Load X5 URDF into MuJoCo and apply arm joint targets.

Default: wrap robot in a lit scene with checker floor (brighter UI).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from .axes import T_to_xyzrpy, body_pose_T

_TOOL_ROOT = Path(__file__).resolve().parents[1]
_LEWM_ROOT = Path(__file__).resolve().parents[3]  # lewm_x5_grasp/

# Bright scene: sky gradient + checker floor + key lights.
_SCENE_XML = """
<mujoco model="x5_play_scene">
  <visual>
    <headlight diffuse="0.9 0.9 0.9" ambient="0.5 0.5 0.5" specular="0.4 0.4 0.4"/>
    <rgba haze="0.18 0.28 0.38 1"/>
    <global azimuth="140" elevation="-22"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.58 0.72 0.92" rgb2="0.14 0.20 0.30"
             width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" mark="edge"
             rgb1="0.28 0.32 0.36" rgb2="0.62 0.66 0.70" markrgb="0.4 0.44 0.48"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.2 0.1 3.0" dir="0 0 -1"
           diffuse="1 1 1" ambient="0.3 0.3 0.3" specular="0.45 0.45 0.45"/>
    <light name="fill" pos="1.6 -1.2 2.4" dir="-0.45 0.35 -1" diffuse="0.8 0.8 0.75"/>
    <light name="rim" pos="-1.3 1.1 2.0" dir="0.35 -0.25 -1" diffuse="0.55 0.6 0.65"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.05" material="grid"/>
    <frame name="robot_frame" pos="0 0 0"/>
  </worldbody>
</mujoco>
"""


def default_urdf_candidates() -> list[Path]:
    env = os.environ.get("X5_URDF_PATH", "").strip()
    out: list[Path] = []
    if env:
        out.append(Path(env).expanduser())
    out.extend(
        [
            _LEWM_ROOT
            / "vendor/LeRobot-Anything-U-Arm/src/simulation/mani_skill/assets/robots/ARX-X5/X5A.urdf",
            _LEWM_ROOT.parent
            / "ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/x5.urdf",
            Path.home()
            / "project/ARX_X5_clean(7.12eval)/ROS2/X5_ws/src/arx_x5_ros2/arx_x5_controller/x5.urdf",
            Path.home()
            / "project/ARX_X5_clean(7.12eval)/py/arx_x5_python/bimanual/script/x5.urdf",
            _TOOL_ROOT / "assets/x5_kin.xml",
        ]
    )
    return out


def resolve_urdf(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--urdf not found: {p}")
        return p
    tried: list[str] = []
    for c in default_urdf_candidates():
        tried.append(str(c))
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        "No X5 URDF found. Set X5_URDF_PATH or pass --urdf.\nTried:\n  - "
        + "\n  - ".join(tried)
    )


def _name_variants(name: str) -> list[str]:
    name = name.lstrip("/")
    return [name, f"/{name}", f"x5/{name}", f"robot/{name}"]


def _find_named(model: mujoco.MjModel, obj_type: int, names: Sequence[str]) -> int:
    for name in names:
        for variant in _name_variants(name):
            bid = mujoco.mj_name2id(model, obj_type, variant)
            if bid >= 0:
                return bid
    # suffix match (attach may add odd prefixes)
    wanted = {n.lstrip("/") for n in names}
    nobj = model.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else model.njnt
    for i in range(nobj):
        raw = mujoco.mj_id2name(model, obj_type, i) or ""
        base = raw.rsplit("/", 1)[-1]
        if base in wanted:
            return i
    return -1


@dataclass
class X5MjModel:
    model: mujoco.MjModel
    data: mujoco.MjData
    urdf_path: Path
    arm_joint_ids: list[int]
    arm_qpos_addrs: list[int]
    ee_body_id: int
    base_body_id: int
    link_body_ids: list[int]

    @property
    def n_arm(self) -> int:
        return len(self.arm_joint_ids)

    def joint_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
            for j in self.arm_joint_ids
        ]

    def set_arm_qpos(self, q: Sequence[float], data: Optional[mujoco.MjData] = None) -> None:
        target = self.data if data is None else data
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        n = min(len(q_arr), self.n_arm)
        for i in range(n):
            target.qpos[self.arm_qpos_addrs[i]] = float(q_arr[i])
        mujoco.mj_forward(self.model, target)

    def make_data(self) -> mujoco.MjData:
        """Extra MjData sharing this model (e.g. HDF5 ghost pose)."""
        return mujoco.MjData(self.model)

    def ee_T(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        d = self.data if data is None else data
        return body_pose_T(d, self.ee_body_id)

    def ee_xyzrpy(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        return T_to_xyzrpy(self.ee_T(data))

    def base_T(self) -> np.ndarray:
        if self.base_body_id < 0:
            return np.eye(4, dtype=np.float64)
        return body_pose_T(self.data, self.base_body_id)


def _arm_joints(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for i in range(1, 7):
        jid = _find_named(model, mujoco.mjtObj.mjOBJ_JOINT, [f"joint{i}"])
        if jid < 0:
            ids = []
            break
        ids.append(jid)
    if ids:
        return ids
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            ids.append(j)
        if len(ids) >= 6:
            break
    if len(ids) < 6:
        raise RuntimeError(f"need >=6 hinge joints, found {len(ids)}")
    return ids[:6]


def _compile_with_scene(urdf_path: Path) -> mujoco.MjModel:
    """Attach robot URDF into a lit floor scene."""
    scene = mujoco.MjSpec.from_string(_SCENE_XML)
    robot = mujoco.MjSpec.from_file(str(urdf_path))
    frame = scene.worldbody.first_frame()
    if frame is None:
        raise RuntimeError("scene missing robot_frame")
    scene.attach(robot, frame=frame)
    model = scene.compile()
    # Extra headlight boost (viewer also uses this).
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    model.vis.headlight.specular[:] = [0.35, 0.35, 0.35]
    model.vis.rgba.haze[:] = [0.18, 0.28, 0.38, 1.0]
    return model


def _compile_plain(urdf_path: Path) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    return model


def load_x5(
    urdf: Optional[str | Path] = None,
    *,
    ee_body: str = "link6",
    with_floor: bool = True,
) -> X5MjModel:
    path = resolve_urdf(urdf)
    if with_floor:
        try:
            model = _compile_with_scene(path)
        except Exception as exc:
            print(f"[mj_model] scene+floor failed ({exc}); falling back to plain URDF")
            model = _compile_plain(path)
    else:
        model = _compile_plain(path)

    data = mujoco.MjData(model)
    arm_ids = _arm_joints(model)
    addrs = [int(model.jnt_qposadr[j]) for j in arm_ids]
    ee_id = _find_named(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        [ee_body, "link6", "ee_link", "tool0", "end_effector"],
    )
    if ee_id < 0:
        ee_id = max(1, model.nbody - 1)
    base_id = _find_named(model, mujoco.mjtObj.mjOBJ_BODY, ["base_link", "base", "link0"])
    link_ids: list[int] = []
    if base_id >= 0:
        link_ids.append(base_id)
    for i in range(1, 7):
        lid = _find_named(model, mujoco.mjtObj.mjOBJ_BODY, [f"link{i}"])
        if lid >= 0:
            link_ids.append(lid)
    if ee_id >= 0 and ee_id not in link_ids:
        link_ids.append(ee_id)
    xm = X5MjModel(
        model=model,
        data=data,
        urdf_path=path,
        arm_joint_ids=arm_ids,
        arm_qpos_addrs=addrs,
        ee_body_id=ee_id,
        base_body_id=base_id,
        link_body_ids=link_ids,
    )
    mujoco.mj_forward(model, data)
    return xm
