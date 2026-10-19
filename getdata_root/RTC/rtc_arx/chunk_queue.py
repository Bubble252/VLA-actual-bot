"""Physical ΔEEF action chunk queue (pre pos_scale / rot_scale)."""

from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np


class PhysicalChunkQueue:
    """FIFO of physical actions with shape (T, 7) = ΔEEF(6) + gripper(1)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._actions: Optional[np.ndarray] = None  # (T, 7)
        self._index = 0
        self.global_step = 0  # number of successful pops

    def clear(self) -> None:
        with self._lock:
            self._actions = None
            self._index = 0

    def replace(self, actions: np.ndarray, *, skip: int = 0) -> None:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] < 6:
            raise ValueError(f"Expected (T, >=6); got {arr.shape}")
        if arr.shape[-1] == 6:
            arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=-1)
        elif arr.shape[-1] > 7:
            arr = arr[:, :7]
        skip = max(0, min(int(skip), arr.shape[0]))
        with self._lock:
            self._actions = arr[skip:].copy()
            self._index = 0

    def leftover(self) -> np.ndarray:
        """Unconsumed physical actions, shape (L, 7); may be empty (0, 7)."""
        with self._lock:
            if self._actions is None or self._index >= len(self._actions):
                return np.zeros((0, 7), dtype=np.float32)
            return self._actions[self._index :].copy()

    def remaining(self) -> int:
        with self._lock:
            if self._actions is None:
                return 0
            return max(0, len(self._actions) - self._index)

    def pop(self) -> Optional[np.ndarray]:
        """Return next action (7,) or None if empty."""
        with self._lock:
            if self._actions is None or self._index >= len(self._actions):
                return None
            action = self._actions[self._index].copy()
            self._index += 1
            self.global_step += 1
            return action
