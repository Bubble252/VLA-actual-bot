"""Build mode4_guard config, optionally inheriting workspace limits from task json."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .mode4_guard import Mode4GuardConfig


def mode4_guard_from_task_config(config: Dict[str, Any]) -> Dict[str, Any]:
    guard = dict(config.get("mode4_guard") or {})
    guard.setdefault("enabled", True)

    workspace = config.get("workspace")
    if workspace and "workspace_min_xyz" not in guard:
        guard["workspace_min_xyz"] = workspace.get("min_xyz")
    if workspace and "workspace_max_xyz" not in guard:
        guard["workspace_max_xyz"] = workspace.get("max_xyz")
    return guard


def resolve_mode4_guard_config(
    explicit: Optional[Dict[str, Any]] = None,
    task_config: Optional[Dict[str, Any]] = None,
) -> Mode4GuardConfig:
    if explicit is not None:
        return Mode4GuardConfig.from_dict(explicit)
    if task_config is not None:
        return Mode4GuardConfig.from_dict(mode4_guard_from_task_config(task_config))
    return Mode4GuardConfig()
