#!/usr/bin/env python3
"""Quick checks for singularity replay slowdown (no ROS required)."""

from __future__ import annotations

import numpy as np

from lewm_x5_grasp.singularity_adjuster import SingularityAdjuster, SingularityConfig


def test_period_scale_dilates_and_slows() -> None:
    cfg = SingularityConfig(
        elbow_thresh_rad=0.5,
        wrist_thresh_rad=0.15,
        slowdown_enabled=True,
        slowdown_factor=3.0,
        slowdown_expand_frames=2,
    )
    adj = SingularityAdjuster(cfg)
    # Safe baseline: elbow folded, wrist off singular; frame 5 elbow → 0.
    qpos = np.zeros((11, 7), dtype=np.float64)
    qpos[:, 2] = 1.0
    qpos[:, 4] = 0.5
    qpos[5, 2] = 0.0
    scales = adj.period_scale(qpos)
    assert scales[5] == 3.0
    assert scales[3] == 3.0 and scales[7] == 3.0
    assert scales[0] == 1.0 and scales[10] == 1.0


def test_period_scale_disabled() -> None:
    cfg = SingularityConfig(slowdown_enabled=False, slowdown_factor=5.0)
    qpos = np.zeros((5, 7), dtype=np.float64)
    scales = SingularityAdjuster(cfg).period_scale(qpos)
    assert np.allclose(scales, 1.0)


if __name__ == "__main__":
    test_period_scale_dilates_and_slows()
    test_period_scale_disabled()
    print("singularity_slowdown tests passed")
