"""Offline singularity detection and EEF RPY micro-adjustment for mode=4 replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .x5_fk import SingularityMetrics, j1_axis_distance, wrist_center


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


@dataclass
class SingularityConfig:
    elbow_thresh_rad: float = 0.15
    forearm_vertical_thresh_rad: float = 0.2
    shoulder_dist_thresh_m: float = 0.05
    wrist_thresh_rad: float = 0.15
    max_rpy_adjust_rad: float = np.deg2rad(3.0)
    blend_frames: int = 10
    max_rpy_step_rad: float = 0.04
    max_joint_step_rad: float = 0.05
    # Replay-only: slow through singular neighborhoods without changing EEF poses.
    slowdown_enabled: bool = True
    slowdown_factor: float = 3.0
    slowdown_expand_frames: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SingularityConfig":
        if not data:
            return cls()
        kwargs: dict[str, Any] = {}
        for key in (
            "elbow_thresh_rad",
            "forearm_vertical_thresh_rad",
            "shoulder_dist_thresh_m",
            "wrist_thresh_rad",
            "max_rpy_adjust_rad",
            "blend_frames",
            "max_rpy_step_rad",
            "max_joint_step_rad",
            "slowdown_enabled",
            "slowdown_factor",
            "slowdown_expand_frames",
        ):
            if key in data:
                kwargs[key] = data[key]
        deg_map = {
            "elbow_thresh_deg": "elbow_thresh_rad",
            "forearm_vertical_thresh_deg": "forearm_vertical_thresh_rad",
            "wrist_thresh_deg": "wrist_thresh_rad",
            "max_rpy_adjust_deg": "max_rpy_adjust_rad",
            "max_rpy_step_deg": "max_rpy_step_rad",
            "max_joint_step_deg": "max_joint_step_rad",
        }
        for deg_key, rad_key in deg_map.items():
            if deg_key in data and rad_key not in kwargs:
                kwargs[rad_key] = float(np.deg2rad(float(data[deg_key])))
        return cls(**kwargs)


@dataclass
class SingularityFrameReport:
    index: int
    metrics: SingularityMetrics
    reasons: list[str] = field(default_factory=list)
    rpy_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SingularityAdjustStats:
    total_frames: int = 0
    danger_frames: int = 0
    adjusted_frames: int = 0
    max_rpy_delta_deg: float = 0.0
    remaining_danger_frames: int = 0
    verify_ok: bool = True
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"danger={self.danger_frames}/{self.total_frames}, "
            f"adjusted={self.adjusted_frames}, "
            f"max|Δrpy|={self.max_rpy_delta_deg:.2f}°, "
            f"remaining={self.remaining_danger_frames}, "
            f"verify={'ok' if self.verify_ok else 'warn'}"
        )


class SingularityAdjuster:
    def __init__(self, config: SingularityConfig | None = None) -> None:
        self.config = config or SingularityConfig()

    def metrics_for_joints(self, joints: np.ndarray) -> SingularityMetrics:
        q = np.asarray(joints[:6], dtype=np.float64)
        cfg = self.config

        elbow = float(abs(q[2]))
        elbow_prox = max(0.0, 1.0 - elbow / cfg.elbow_thresh_rad) if elbow < cfg.elbow_thresh_rad else 0.0

        j23 = float(q[1] + q[2])
        forearm_err = min(abs(j23 - np.pi / 2.0), abs(j23 + np.pi / 2.0))
        forearm_prox = (
            max(0.0, 1.0 - forearm_err / cfg.forearm_vertical_thresh_rad)
            if forearm_err < cfg.forearm_vertical_thresh_rad
            else 0.0
        )

        dist = j1_axis_distance(q)
        shoulder_prox = (
            max(0.0, 1.0 - dist / cfg.shoulder_dist_thresh_m)
            if dist < cfg.shoulder_dist_thresh_m
            else 0.0
        )

        wrist = float(abs(q[4]))
        wrist_prox = (
            max(0.0, 1.0 - wrist / cfg.wrist_thresh_rad) if wrist < cfg.wrist_thresh_rad else 0.0
        )

        return SingularityMetrics(
            elbow_proximity=elbow_prox,
            forearm_proximity=forearm_prox,
            shoulder_proximity=shoulder_prox,
            wrist_proximity=wrist_prox,
        )

    def detect(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64)
        flags = np.zeros(len(qpos), dtype=bool)
        for i in range(len(qpos)):
            flags[i] = self.metrics_for_joints(qpos[i]).is_dangerous
        return flags

    def period_scale(self, qpos: np.ndarray) -> np.ndarray:
        """Per-frame replay period multiplier near singularities (path unchanged)."""
        qpos = np.asarray(qpos, dtype=np.float64)
        n = len(qpos)
        scales = np.ones(n, dtype=np.float64)
        if n == 0 or not self.config.slowdown_enabled:
            return scales
        factor = max(1.0, float(self.config.slowdown_factor))
        if factor <= 1.0 + 1e-9:
            return scales
        danger = self.detect(qpos)
        expand = max(0, int(self.config.slowdown_expand_frames))
        if expand > 0 and np.any(danger):
            dilated = danger.copy()
            for i in np.flatnonzero(danger):
                lo = max(0, int(i) - expand)
                hi = min(n, int(i) + expand + 1)
                dilated[lo:hi] = True
            danger = dilated
        scales[danger] = factor
        return scales

    def detect_detailed(self, qpos: np.ndarray) -> list[SingularityFrameReport]:
        qpos = np.asarray(qpos, dtype=np.float64)
        reports: list[SingularityFrameReport] = []
        for i in range(len(qpos)):
            metrics = self.metrics_for_joints(qpos[i])
            reasons: list[str] = []
            if metrics.elbow_proximity > 0.0:
                reasons.append("elbow")
            if metrics.forearm_proximity > 0.0:
                reasons.append("forearm_vertical")
            if metrics.shoulder_proximity > 0.0:
                reasons.append("shoulder")
            if metrics.wrist_proximity > 0.0:
                reasons.append("wrist")
            reports.append(SingularityFrameReport(index=i, metrics=metrics, reasons=reasons))
        return reports

    def _rpy_delta_for_frame(self, qpos_row: np.ndarray, metrics: SingularityMetrics) -> np.ndarray:
        if not metrics.is_dangerous:
            return np.zeros(3, dtype=np.float64)

        q = np.asarray(qpos_row[:6], dtype=np.float64)
        cfg = self.config
        delta = np.zeros(3, dtype=np.float64)
        amp = cfg.max_rpy_adjust_rad

        if metrics.elbow_proximity > 0.0:
            prox = metrics.elbow_proximity
            sign = 1.0 if q[2] >= 0.0 else -1.0
            delta[1] += sign * amp * _smoothstep(prox)

        if metrics.forearm_proximity > 0.0:
            prox = metrics.forearm_proximity
            j23 = float(q[1] + q[2])
            sign = 1.0 if j23 >= 0.0 else -1.0
            delta[0] += sign * amp * _smoothstep(prox) * 0.6
            delta[1] += sign * amp * _smoothstep(prox) * 0.4

        if metrics.shoulder_proximity > 0.0:
            prox = metrics.shoulder_proximity
            wc = wrist_center(q)
            yaw_sign = 1.0 if wc[1] >= 0.0 else -1.0
            delta[2] += yaw_sign * amp * _smoothstep(prox)
            delta[1] += amp * _smoothstep(prox) * 0.35

        if metrics.wrist_proximity > 0.0:
            prox = metrics.wrist_proximity
            sign = 1.0 if q[4] >= 0.0 else -1.0
            delta[0] += sign * amp * _smoothstep(prox) * 0.5

        norm = float(np.linalg.norm(delta))
        if norm > amp and norm > 1e-9:
            delta *= amp / norm
        return delta

    def _blend_delta_profile(self, raw: np.ndarray) -> np.ndarray:
        """Smooth delta_rpy across frame boundaries."""
        n = len(raw)
        if n == 0:
            return raw
        out = raw.copy()
        active = np.any(np.abs(raw) > 1e-9, axis=1)
        if not np.any(active):
            return out

        blend = max(1, int(self.config.blend_frames))
        segments: list[tuple[int, int]] = []
        start = None
        for i in range(n):
            if active[i] and start is None:
                start = i
            elif not active[i] and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, n - 1))

        smoothed = np.zeros_like(raw)
        for lo, hi in segments:
            seg_len = hi - lo + 1
            for i in range(lo, hi + 1):
                t = (i - lo) / max(seg_len - 1, 1)
                edge = min(t, 1.0 - t) * 2.0
                scale = _smoothstep(min(1.0, edge * blend))
                smoothed[i] = raw[i] * scale
        return smoothed

    def adjust(self, eef: np.ndarray, qpos: np.ndarray) -> tuple[np.ndarray, SingularityAdjustStats]:
        eef = np.asarray(eef, dtype=np.float64).copy()
        qpos = np.asarray(qpos, dtype=np.float64)
        if len(eef) != len(qpos):
            raise ValueError("eef and qpos must have the same length")

        stats = SingularityAdjustStats(total_frames=len(qpos))
        if len(qpos) == 0 or eef.shape[1] < 6:
            return eef, stats

        reports = self.detect_detailed(qpos)
        raw_delta = np.zeros((len(qpos), 3), dtype=np.float64)
        for rep in reports:
            if rep.metrics.is_dangerous:
                stats.danger_frames += 1
                raw_delta[rep.index] = self._rpy_delta_for_frame(qpos[rep.index], rep.metrics)

        delta_profile = self._blend_delta_profile(raw_delta)
        for i in range(len(qpos)):
            d = delta_profile[i]
            if float(np.linalg.norm(d)) < 1e-9:
                continue
            stats.adjusted_frames += 1
            eef[i, 3:6] = eef[i, 3:6] + d

        stats.max_rpy_delta_deg = float(
            np.max(np.degrees(np.linalg.norm(delta_profile, axis=1))) if len(delta_profile) else 0.0
        )
        verify = self.verify(eef, qpos)
        stats.remaining_danger_frames = verify["remaining_danger_frames"]
        stats.verify_ok = bool(verify["ok"])
        stats.warnings = list(verify.get("warnings", []))
        return eef, stats

    def verify(self, eef: np.ndarray, qpos: np.ndarray) -> dict[str, Any]:
        eef = np.asarray(eef, dtype=np.float64)
        qpos = np.asarray(qpos, dtype=np.float64)
        cfg = self.config
        warnings: list[str] = []

        danger = self.detect(qpos)
        remaining = int(np.sum(danger))

        if eef.shape[1] >= 6 and len(eef) > 1:
            rpy_steps = np.linalg.norm(np.diff(eef[:, 3:6], axis=0), axis=1)
            max_step = float(np.max(rpy_steps)) if len(rpy_steps) else 0.0
            if max_step > cfg.max_rpy_step_rad:
                warnings.append(
                    f"max RPY step {np.degrees(max_step):.2f}° > "
                    f"{np.degrees(cfg.max_rpy_step_rad):.2f}°"
                )

        if len(qpos) > 1:
            q_steps = np.max(
                np.abs(
                    np.arctan2(
                        np.sin(np.diff(qpos[:, :6], axis=0)),
                        np.cos(np.diff(qpos[:, :6], axis=0)),
                    )
                ),
                axis=1,
            )
            max_q = float(np.max(q_steps)) if len(q_steps) else 0.0
            if max_q > cfg.max_joint_step_rad:
                warnings.append(
                    f"recorded joint step {np.degrees(max_q):.2f}° > "
                    f"{np.degrees(cfg.max_joint_step_rad):.2f}°"
                )

        if remaining > 0:
            warnings.append(
                f"{remaining} frames still near singularity in recorded joints "
                "(RPY nudge cannot fully remove joint-space risk without IK verify)"
            )

        ok = len(warnings) == 0
        return {
            "ok": ok,
            "remaining_danger_frames": remaining,
            "warnings": warnings,
        }

    def segment_report(self, qpos: np.ndarray, *, min_run: int = 3) -> list[dict[str, Any]]:
        """Return contiguous dangerous segments for CLI reporting."""
        flags = self.detect(qpos)
        segments: list[dict[str, Any]] = []
        start = None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            elif not flag and start is not None:
                if i - start >= min_run:
                    segments.append({"start": start, "end": i - 1, "length": i - start})
                start = None
        if start is not None and len(flags) - start >= min_run:
            segments.append({"start": start, "end": len(flags) - 1, "length": len(flags) - start})
        return segments
