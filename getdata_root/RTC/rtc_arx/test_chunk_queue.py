#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from chunk_queue import PhysicalChunkQueue


def main() -> None:
    q = PhysicalChunkQueue()
    chunk = np.arange(56, dtype=np.float32).reshape(8, 7) / 100.0
    q.replace(chunk)
    assert q.remaining() == 8
    a0 = q.pop()
    assert a0 is not None and np.allclose(a0, chunk[0])
    left = q.leftover()
    assert left.shape == (7, 7)
    assert np.allclose(left[0], chunk[1])
    q.replace(chunk, skip=2)
    assert q.remaining() == 6
    assert np.allclose(q.pop(), chunk[2])
    # (T,6) pad gripper
    q.replace(np.ones((3, 6), dtype=np.float32))
    assert q.pop().shape == (7,)
    print("test_chunk_queue OK")


if __name__ == "__main__":
    main()
