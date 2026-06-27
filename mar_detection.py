from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SQAD_P = 0.6827


@dataclass
class DynamicMARStatus:
    """Runtime status for adaptive MAR/yawn detection."""

    raw: float
    mu: float
    threshold: float
    is_open: bool
    calibrated: bool
    progress: float
    baseline: float
    sigma: float


class DynamicMAR:
    """Dynamic MAR logic kept consistent with the user's MAR files.

    This is the realtime class used by evaluate_mar_videos.py:
        DynamicMAR(window_size=150, factor=0.7, floor=0.15)

    Formula:
        mu_t = EWMA(MAR_t)
        sigma_t = SQAD-like quantile of |MAR_i - mu_t| in sliding window
        threshold_t = max(floor, mu_t + factor * sigma_t)
        is_open = MAR_t > threshold_t

    No personalized baseline/MAD/min_gap algorithm is added here.
    """

    def __init__(self, window_size: int = 150, factor: float = 0.7, floor: float = 0.15, alpha: float = 0.10) -> None:
        self.window_size = max(10, int(window_size))
        self.factor = float(factor)
        self.floor = float(floor)
        self.alpha = float(alpha)
        self._buf: list[float] = []
        self._mu: float | None = None
        self._sigma: float = 0.0
        self._threshold: float = self.floor

    def update(self, mar: float) -> DynamicMARStatus:
        raw = float(mar) if np.isfinite(mar) else 0.0
        if self._mu is None:
            self._mu = raw
        else:
            self._mu = self.alpha * raw + (1.0 - self.alpha) * self._mu

        self._buf.append(raw)
        if len(self._buf) > self.window_size:
            self._buf.pop(0)

        if len(self._buf) >= 10:
            devs = sorted(abs(v - self._mu) for v in self._buf)
            idx = min(len(devs) - 1, int(SQAD_P * len(devs)))
            self._sigma = float(devs[idx]) if devs else 0.0
        else:
            self._sigma = 0.0

        self._threshold = max(self.floor, float(self._mu) + self.factor * self._sigma)
        is_open = bool(raw > self._threshold)
        progress = min(1.0, len(self._buf) / self.window_size)

        return DynamicMARStatus(
            raw=raw,
            mu=float(self._mu),
            threshold=float(self._threshold),
            is_open=is_open,
            calibrated=len(self._buf) >= 10,
            progress=progress,
            baseline=float(self._mu),
            sigma=float(self._sigma),
        )

    @property
    def threshold(self) -> float:
        return float(self._threshold)

    @property
    def calibrated(self) -> bool:
        return len(self._buf) >= 10

    def reset(self) -> None:
        self._buf.clear()
        self._mu = None
        self._sigma = 0.0
        self._threshold = self.floor
