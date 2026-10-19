"""ΔEEF policy client with optional RTC track-B leftover / delay fields.

``rtc_mode=off``: sync refill when queue empty (compatible with StarVLA ModelClient).
``rtc_mode=track_b``: prefetch next chunk while executing leftover; send physical
prefix + inferred delay to a P3-patched StarVLA policy server.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .chunk_queue import PhysicalChunkQueue

logger = logging.getLogger(__name__)


def _to_raw_action(action7: np.ndarray) -> Dict[str, np.ndarray]:
    a = np.asarray(action7, dtype=np.float32).reshape(-1)
    return {
        "world_vector": a[:3].copy(),
        "rotation_delta": a[3:6].copy(),
        "open_gripper": a[6:7].copy(),
    }


class RTCDeltaEefClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        unnorm_key: str = "franka",
        rtc_mode: str = "off",
        execution_horizon: int = 4,
        prefetch_remaining: int = 4,
        control_hz: float = 5.0,
        starvla_root: Optional[str] = None,
    ) -> None:
        mode = str(rtc_mode).lower().strip()
        if mode not in ("off", "track_b"):
            raise ValueError(f"rtc_mode must be off|track_b, got {rtc_mode!r}")
        self.rtc_mode = mode
        self.unnorm_key = unnorm_key
        self.execution_horizon = int(execution_horizon)
        self.prefetch_remaining = max(1, int(prefetch_remaining))
        self.control_hz = float(control_hz)

        if starvla_root:
            import os
            import sys

            root = os.path.abspath(starvla_root)
            if root not in sys.path:
                sys.path.insert(0, root)

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self._ws = WebsocketClientPolicy(host, port)
        self._meta = self._ws.get_server_metadata()
        self.action_chunk_size = int(self._meta.get("action_chunk_size", 8))

        self.queue = PhysicalChunkQueue()
        self.task_description: Optional[str] = None

        self._lock = threading.Lock()
        # websockets sync client is NOT multi-recv safe — serialize all predict/recv.
        self._ws_lock = threading.Lock()
        self._infer_thread: Optional[threading.Thread] = None
        self._pending: Optional[Dict[str, Any]] = None  # filled by worker
        self._infer_error: Optional[BaseException] = None
        self._send_global_step = 0
        self._last_delay = 0
        self._last_rtc_meta: Dict[str, Any] = {}

        logger.info(
            "RTCDeltaEefClient ready: rtc_mode=%s chunk=%d prefetch_at=%d horizon=%d meta=%s",
            self.rtc_mode,
            self.action_chunk_size,
            self.prefetch_remaining,
            self.execution_horizon,
            self._meta,
        )

    def _cap_delay(self, delay: int) -> int:
        """Keep delay inside one chunk; cold-start RTT must not become delay=300+."""
        return int(max(0, min(int(delay), max(0, self.action_chunk_size - 1))))

    def close(self) -> None:
        self._join_infer(timeout=5.0)
        self._ws.close()

    def reset(self, task_description: str) -> None:
        self._join_infer(timeout=5.0)
        self.task_description = task_description
        self.queue.clear()
        with self._lock:
            self._pending = None
            self._infer_error = None
            self._last_rtc_meta = {}

    def step(self, example: dict, step: int = 0) -> dict:
        if example.get("lang") is not None and example.get("lang") != self.task_description:
            self.reset(str(example["lang"]))

        self._absorb_pending()

        if self.rtc_mode == "off":
            if self.queue.remaining() == 0:
                self._join_infer(timeout=120.0)
                self._infer_sync(example, rtc_enabled=False)
        else:
            # track_b: ensure we have something to execute; prefetch in background
            if self.queue.remaining() == 0:
                # Wait for in-flight prefetch first (avoids dual recv on one websocket).
                self._join_infer(timeout=120.0)
                if self.queue.remaining() == 0:
                    # blocking fill (no leftover → server falls back / no guidance)
                    self._infer_sync(example, rtc_enabled=False)
            self._maybe_prefetch(example)

        action = self.queue.pop()
        if action is None:
            self._join_infer(timeout=120.0)
            if self.queue.remaining() == 0:
                self._infer_sync(example, rtc_enabled=self.rtc_mode == "track_b")
            action = self.queue.pop()
        if action is None:
            raise RuntimeError("Policy returned empty action chunk")

        # After pop, still try prefetch for next steps
        if self.rtc_mode == "track_b":
            self._maybe_prefetch(example)

        out = {
            "raw_action": _to_raw_action(action),
            "rtc": {
                "mode": self.rtc_mode,
                "queue_remaining": self.queue.remaining(),
                "last_delay": self._last_delay,
                "server": dict(self._last_rtc_meta),
            },
        }
        return out

    # ------------------------------------------------------------------ sync / async
    def _predict(self, example: dict, *, rtc_enabled: bool, prefix: Optional[np.ndarray], delay: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "examples": [example],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
        }
        if rtc_enabled and prefix is not None and prefix.shape[0] > 0:
            payload.update(
                {
                    "rtc_enabled": True,
                    "rtc_prefix": np.asarray(prefix, dtype=np.float32),
                    "rtc_prefix_space": "physical",
                    "inference_delay": int(max(0, delay)),
                    "execution_horizon": int(self.execution_horizon),
                }
            )
        t0 = time.monotonic()
        with self._ws_lock:
            response = self._ws.predict_action(payload)
        rtt = time.monotonic() - t0
        data = response.get("data", response)
        if "actions" not in data:
            raise KeyError(f"Server response missing actions: keys={list(data.keys())}")
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        rtc_meta = data.get("rtc", {}) if isinstance(data, dict) else {}
        return {"actions": actions, "rtc": rtc_meta or {}, "rtt": rtt, "response": response}

    def _infer_sync(self, example: dict, *, rtc_enabled: bool) -> None:
        leftover = self.queue.leftover()
        delay_est = self._cap_delay(self._last_delay)
        result = self._predict(
            example,
            rtc_enabled=rtc_enabled and leftover.shape[0] > 0,
            prefix=leftover if leftover.shape[0] > 0 else None,
            delay=delay_est,
        )
        # sync path: queue was empty or we block-fill; skip=0
        skip = 0
        if rtc_enabled and leftover.shape[0] > 0:
            # if somehow called with leftover still present, replace from delay
            skip = min(delay_est, result["actions"].shape[0])
        self.queue.replace(result["actions"], skip=skip)
        self._last_rtc_meta = dict(result.get("rtc") or {})
        # update delay estimate from RTT (capped — cold start must not poison RTC)
        dt = 1.0 / max(self.control_hz, 1e-3)
        self._last_delay = self._cap_delay(int(max(0, round(result["rtt"] / dt))))

    def _maybe_prefetch(self, example: dict) -> None:
        with self._lock:
            busy = self._infer_thread is not None and self._infer_thread.is_alive()
            has_pending = self._pending is not None
        if busy or has_pending:
            return
        if self.queue.remaining() > self.prefetch_remaining:
            return

        leftover = self.queue.leftover()
        if leftover.shape[0] == 0:
            return

        send_step = self.queue.global_step
        delay_est = self._cap_delay(self._last_delay)
        # snapshot example images/lang for the worker
        example_snap = {
            "image": example.get("image"),
            "lang": example.get("lang"),
        }
        if "state" in example:
            example_snap["state"] = example["state"]

        def worker() -> None:
            try:
                result = self._predict(
                    example_snap,
                    rtc_enabled=True,
                    prefix=leftover,
                    delay=delay_est,
                )
                with self._lock:
                    self._pending = {
                        "result": result,
                        "send_global_step": send_step,
                        "prefix_len": int(leftover.shape[0]),
                    }
            except BaseException as exc:  # noqa: BLE001
                logger.exception("RTC prefetch failed")
                with self._lock:
                    self._infer_error = exc
                    self._pending = None

        t = threading.Thread(target=worker, name="rtc-prefetch", daemon=True)
        with self._lock:
            self._infer_thread = t
            self._send_global_step = send_step
        t.start()

    def _absorb_pending(self) -> None:
        with self._lock:
            err = self._infer_error
            self._infer_error = None
            pending = self._pending
            self._pending = None
            thread = self._infer_thread
        if err is not None:
            logger.warning("Ignoring RTC prefetch error: %s", err)
        if thread is not None and not thread.is_alive():
            with self._lock:
                if self._infer_thread is thread:
                    self._infer_thread = None
        if pending is None:
            return

        result = pending["result"]
        send_step = int(pending["send_global_step"])
        actual_delay = self._cap_delay(max(0, self.queue.global_step - send_step))
        self._last_delay = actual_delay
        # Replace remaining queue with guided chunk, skipping actions that time already passed
        self.queue.replace(result["actions"], skip=actual_delay)
        self._last_rtc_meta = dict(result.get("rtc") or {})
        self._last_rtc_meta["client_actual_delay"] = actual_delay
        self._last_rtc_meta["client_prefix_len_at_send"] = pending.get("prefix_len", 0)
        dt = 1.0 / max(self.control_hz, 1e-3)
        # blend RTT-based estimate for next send (capped)
        rtt_delay = self._cap_delay(int(max(0, round(float(result.get("rtt", 0.0)) / dt))))
        self._last_delay = self._cap_delay(max(actual_delay, rtt_delay))

    def _join_infer(self, timeout: float) -> None:
        with self._lock:
            t = self._infer_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._absorb_pending()
