"""100Hz interpolation publish thread.

Sits between the 20Hz policy inference loop and the 500Hz ARX joint controller.
Linearly interpolates joint targets at a configurable publish rate (default 100Hz)
to eliminate the "move-stop-move-stop" jitter caused by sparse 20Hz commands.

Usage (from eval_starvla_libero_x5.py):

    target = InterpTarget()
    pub = InterpPublishThread(node, target, publish_hz=100.0,
                              interp_duration=1.0/control_hz)
    pub.start()
    # ... main loop: target.update(q6, gripper, target_eef) instead of publish_command
    pub.stop(); pub.join(timeout=2.0)

Design decisions (confirmed):
  - Interpolation space: joint space (IK already done in main thread)
  - New target arrival: immediately switch from current published position
  - Interpolation method: linear (no min-jerk)
  - Gripper: immediate switch (no interpolation, binary)
  - IK seed in main thread: last_cmd_q (not synced to publish thread)
  - EMA: unchanged (joint_ema stays at 0.9)
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import numpy as np

ARM_JOINT_DIMS = 6


class InterpTarget:
    """Thread-safe shared target: main thread writes, publish thread reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target_q: Optional[np.ndarray] = None
        self._target_gripper: Optional[float] = None
        self._target_eef: Optional[np.ndarray] = None
        self._seq: int = 0

    def update(self, q6: Sequence[float], gripper: float, eef: Sequence[float]) -> None:
        """Main thread: write new joint target + gripper + eef (for logging)."""
        with self._lock:
            self._target_q = np.asarray(q6, dtype=np.float64).reshape(-1)[:ARM_JOINT_DIMS].copy()
            self._target_gripper = float(gripper)
            self._target_eef = np.asarray(eef, dtype=np.float64).reshape(-1)[:6].copy()
            self._seq += 1

    def read(self) -> tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], int]:
        """Publish thread: read current target + sequence number."""
        with self._lock:
            if self._target_q is None:
                return None, None, None, 0
            return (
                self._target_q.copy(),
                self._target_gripper,
                self._target_eef.copy() if self._target_eef is not None else None,
                self._seq,
            )


def _shortest_delta(target: float, current: float) -> float:
    """Shortest angular difference (handles wrap-around)."""
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


def _interp_joints(start: np.ndarray, target: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation in joint space (shortest angular path)."""
    t = float(np.clip(t, 0.0, 1.0))
    out = np.zeros(ARM_JOINT_DIMS, dtype=np.float64)
    for j in range(ARM_JOINT_DIMS):
        d = _shortest_delta(float(target[j]), float(start[j]))
        out[j] = start[j] + t * d
    return out


def _safety_rate_limit(cur: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    """Conservative per-tick joint rate limit (safety net in publish thread)."""
    if max_delta <= 0:
        return target.copy()
    out = cur.copy()
    for j in range(ARM_JOINT_DIMS):
        d = _shortest_delta(float(target[j]), float(cur[j]))
        d = float(np.clip(d, -max_delta, max_delta))
        out[j] = cur[j] + d
    return out


class InterpPublishThread(threading.Thread):
    """100Hz publish thread: linearly interpolates toward latest target.

    Args:
        node: X5ArmBridge (or compatible) with publish_command(ArmCommand).
        target_holder: InterpTarget instance shared with main thread.
        publish_hz: publish frequency (default 100).
        interp_duration: time to reach each new target (default = 1/main_hz = 50ms).
        safety_rate_limit_rad: per-tick joint cap in publish thread (default 0.04).
    """

    def __init__(
        self,
        node,
        target_holder: InterpTarget,
        *,
        publish_hz: float = 100.0,
        interp_duration: float = 0.05,
        safety_rate_limit_rad: float = 0.04,
    ) -> None:
        super().__init__(daemon=True, name="interp-publish")
        self.node = node
        self.target_holder = target_holder
        self.publish_hz = float(publish_hz)
        self.period = 1.0 / max(self.publish_hz, 1e-3)
        self.interp_duration = max(float(interp_duration), self.period)
        self.safety_rate_limit_rad = float(safety_rate_limit_rad)
        self._stop_event = threading.Event()

        self._current_q: Optional[np.ndarray] = None
        self._interp_start: Optional[np.ndarray] = None
        self._interp_target: Optional[np.ndarray] = None
        self._interp_t0: float = 0.0
        self._last_seq: int = -1
        self._current_gripper: Optional[float] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        print(
            f"[interp-publish] thread started @ {self.publish_hz:.0f}Hz "
            f"interp_duration={self.interp_duration*1000:.0f}ms "
            f"safety_limit={self.safety_rate_limit_rad:.3f}rad/tick",
            flush=True,
        )
        try:
            self._loop()
        except Exception as exc:
            print(f"[interp-publish] FATAL: {exc}", flush=True)
        finally:
            print("[interp-publish] thread exited", flush=True)

    def _loop(self) -> None:
        from lewm_x5_grasp.schemas import ArmCommand

        while not self._stop_event.is_set():
            t_loop = time.monotonic()

            tgt_q, tgt_grip, tgt_eef, seq = self.target_holder.read()

            if tgt_q is not None:
                # Detect new target → immediately switch from current position
                if seq != self._last_seq:
                    if self._current_q is None:
                        # First target: snap directly
                        self._current_q = tgt_q.copy()
                        self._interp_start = tgt_q.copy()
                        self._interp_target = tgt_q.copy()
                    else:
                        self._interp_start = self._current_q.copy()
                        self._interp_target = tgt_q.copy()
                    self._interp_t0 = t_loop
                    self._last_seq = seq
                    self._current_gripper = tgt_grip

                # Linear interpolation
                elapsed = time.monotonic() - self._interp_t0
                t_frac = elapsed / self.interp_duration
                q_interp = _interp_joints(self._interp_start, self._interp_target, t_frac)

                # Safety rate limit (conservative net)
                if self._current_q is not None:
                    q_interp = _safety_rate_limit(
                        self._current_q, q_interp, self.safety_rate_limit_rad
                    )

                # Publish
                xyzrpy = (
                    [float(v) for v in tgt_eef[:6]]
                    if tgt_eef is not None
                    else [0.0] * 6
                )
                cmd = ArmCommand(
                    xyzrpy=xyzrpy,
                    gripper=float(self._current_gripper) if self._current_gripper is not None else 0.0,
                    joint_pos=[float(v) for v in q_interp[:ARM_JOINT_DIMS]],
                    mode=5,
                )
                self.node.publish_command(cmd)
                self._current_q = q_interp.copy()

            # Sleep to maintain publish_hz
            dt = time.monotonic() - t_loop
            if dt < self.period:
                time.sleep(self.period - dt)
