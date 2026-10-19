"""Load shared play/display params (EEF xyz offset, etc.) from YAML.

Canonical file:
  lewm_x5_grasp/scripts/x5_play_params.yaml
Override:
  export X5_PLAY_PARAMS=/path/to/x5_play_params.yaml

eef_via_joint Cartesian base (eval client):
  current = sdk_xyz + eef_xyz_offset_m
  target  = current + policy_Δ
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

# scripts/tool/x5_play_viz/this_file.py → parents[2] == scripts/
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _SCRIPTS_DIR / "x5_play_params.yaml"


def resolve_params_yaml() -> Path:
    env = os.environ.get("X5_PLAY_PARAMS", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            _DEFAULT_YAML,
            Path.home()
            / "project/ARX_X5_clean_server46(7.12get-data)/lewm_x5_grasp/scripts/x5_play_params.yaml",
            Path.home()
            / "project/ARX_X5_clean(7.12eval)/lewm_x5_grasp/scripts/x5_play_params.yaml",
        ]
    )
    tried: list[str] = []
    for p in candidates:
        tried.append(str(p))
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "x5_play_params.yaml not found. Set X5_PLAY_PARAMS or place the file under "
        "lewm_x5_grasp/scripts/.\nTried:\n  - " + "\n  - ".join(tried)
    )


def _parse_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"{path} root must be a mapping")
        return data
    except ImportError:
        # Minimal fallback: only eef_xyz_offset_m: [a, b, c]
        import ast
        import re

        m = re.search(
            r"eef_xyz_offset_m\s*:\s*(\[[^\]]+\])",
            text,
        )
        if not m:
            raise ImportError(
                "PyYAML not installed and fallback parse failed; "
                "pip install pyyaml"
            ) from None
        return {"eef_xyz_offset_m": list(ast.literal_eval(m.group(1)))}


@lru_cache(maxsize=1)
def load_play_params() -> dict[str, Any]:
    path = resolve_params_yaml()
    data = _parse_yaml(path)
    data["_yaml_path"] = str(path)
    return data


def reload_play_params() -> dict[str, Any]:
    load_play_params.cache_clear()
    return load_play_params()


def get_eef_xyz_offset() -> np.ndarray:
    """Meters; sdk_xyz + offset → display current_xyz."""
    data = load_play_params()
    raw = data.get("eef_xyz_offset_m", [0.0, 0.0, 0.0])
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"eef_xyz_offset_m needs 3 values, got {raw}")
    return arr[:3].copy()


# Eager default for importers that expect a module-level array.
# Re-read via get_eef_xyz_offset() / reload_play_params() after editing YAML.
EEF_XYZ_OFFSET = get_eef_xyz_offset()


def apply_eef_xyz_offset(xyzrpy: np.ndarray) -> np.ndarray:
    """Copy of xyzrpy with xyz += YAML eef_xyz_offset_m (rpy unchanged)."""
    # Always re-fetch so YAML edits apply after reload_play_params() / new process.
    offset = get_eef_xyz_offset()
    out = np.asarray(xyzrpy, dtype=np.float64).reshape(-1).copy()
    if out.size < 3:
        raise ValueError("expected at least xyz")
    out[:3] = out[:3] + offset
    return out


def current_xyz_with_offset(sdk_xyz: np.ndarray) -> np.ndarray:
    """SDK current position + YAML offset → display/tool current_xyz."""
    return np.asarray(sdk_xyz, dtype=np.float64).reshape(-1)[:3] + get_eef_xyz_offset()


def get_workspace_xyz_limits(
    *,
    default_min: tuple[float, float, float] = (-0.3, 0.11, 0.0),
    default_max: tuple[float, float, float] = (0.3, 0.55, 0.7),
) -> tuple[np.ndarray, np.ndarray]:
    """SDK/base_link flange AABB for eval workspace clip (meters).

    Keys: ``workspace_min_xyz`` / ``workspace_max_xyz`` in x5_play_params.yaml.
    """
    data = load_play_params()
    lo = data.get("workspace_min_xyz", default_min)
    hi = data.get("workspace_max_xyz", default_max)
    lo_a = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi_a = np.asarray(hi, dtype=np.float64).reshape(-1)
    if lo_a.size < 3 or hi_a.size < 3:
        raise ValueError(
            f"workspace_min/max_xyz need 3 values, got min={lo} max={hi}"
        )
    return lo_a[:3].copy(), hi_a[:3].copy()


def dxyz_xy_remap_enabled() -> bool:
    """Whether YAML dxyz_xy_remap_enabled is on."""
    return bool(load_play_params().get("dxyz_xy_remap_enabled", False))


def apply_dxyz_xy_remap(dxyz: np.ndarray) -> np.ndarray:
    """Remap cartesian Δxyz: (x,y,z) -> (-y, x, z) when enabled in YAML.

    Applied after workspace clip. IK uses remapped Δ; logs show pre and post.
    """
    v = np.asarray(dxyz, dtype=np.float64).reshape(-1)[:3].copy()
    if not dxyz_xy_remap_enabled():
        return v
    # new_x = -old_y ; new_y = old_x ; new_z = old_z  (see x5_play_params.yaml)
    return np.array([-float(v[1]), float(v[0]), float(v[2])], dtype=np.float64)


def get_diff_ik_control_params(
    *,
    default_ik_max_dq: float = 0.12,
    default_max_joint_delta_rad: float = 0.08,
    default_joint_ema: float = 0.55,
    default_ik_damp: float = 0.02,
) -> dict[str, Any]:
    """Shared replay-``v`` / eval ``eef_via_joint`` IK post-process knobs.

    Keys in ``x5_play_params.yaml``:
      ``ik_max_dq``, ``max_joint_delta_rad``, ``joint_ema``, ``ik_damp``

    Process overrides (CLI / Qt) via ``set_diff_ik_control_overrides``.
    """
    data = load_play_params()

    def _f(key: str, default: float) -> float:
        raw = data.get(key, default)
        return float(raw)

    out: dict[str, Any] = {
        "ik_max_dq": _f("ik_max_dq", default_ik_max_dq),
        "max_joint_delta_rad": _f("max_joint_delta_rad", default_max_joint_delta_rad),
        "joint_ema": _f("joint_ema", default_joint_ema),
        "ik_damp": _f("ik_damp", default_ik_damp),
        "yaml_path": str(data.get("_yaml_path", "")),
    }
    for k, v in _DIFF_IK_OVERRIDES.items():
        if v is not None and k in out:
            out[k] = float(v)
    return out


_DIFF_IK_OVERRIDES: dict[str, Optional[float]] = {}


def set_diff_ik_control_overrides(
    *,
    ik_max_dq: Optional[float] = None,
    max_joint_delta_rad: Optional[float] = None,
    joint_ema: Optional[float] = None,
    ik_damp: Optional[float] = None,
    clear: bool = False,
) -> None:
    """Override YAML knobs for this process (CLI / Qt). Pass clear=True to reset."""
    global _DIFF_IK_OVERRIDES
    if clear:
        _DIFF_IK_OVERRIDES = {}
        return
    for key, val in (
        ("ik_max_dq", ik_max_dq),
        ("max_joint_delta_rad", max_joint_delta_rad),
        ("joint_ema", joint_ema),
        ("ik_damp", ik_damp),
    ):
        if val is not None:
            _DIFF_IK_OVERRIDES[key] = float(val)

