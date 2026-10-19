from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from .policy import clip_xyzrpy
from .schemas import ArmCommand, ArmState, PolicyOutput


class GraspStateMachine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase = "approach"
        self._phase_since = time.monotonic()

    def _task(self) -> Dict[str, Any]:
        return self.config["task"]

    def _position_close(self, arm_state: ArmState, target_xyzrpy: list[float]) -> bool:
        tolerance = float(self._task()["position_tolerance"])
        target = np.asarray(target_xyzrpy[:3], dtype=float)
        current = np.asarray(arm_state.end_pos[:3], dtype=float)
        return float(np.linalg.norm(target - current)) <= tolerance

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._phase_since = time.monotonic()

    def update(self, arm_state: ArmState, proposal: PolicyOutput) -> ArmCommand:
        if proposal.phase:
            self._set_phase(proposal.phase)

        if self.phase == "wait":
            return ArmCommand(xyzrpy=proposal.target_xyzrpy, gripper=proposal.gripper, mode=4)

        workspace = self.config["workspace"]
        target = clip_xyzrpy(proposal.target_xyzrpy, workspace)
        command = ArmCommand(xyzrpy=target, gripper=proposal.gripper, mode=4)

        if self.phase == "approach" and self._position_close(arm_state, target):
            self._set_phase("descend")
        elif self.phase == "descend" and self._position_close(arm_state, target):
            self._set_phase("close")
        elif self.phase == "close":
            hold = float(self._task()["close_hold_seconds"])
            if time.monotonic() - self._phase_since >= hold:
                self._set_phase("lift")
        elif self.phase == "lift" and self._position_close(arm_state, target):
            self._set_phase("done")

        return command

    def is_done(self) -> bool:
        return self.phase == "done"
