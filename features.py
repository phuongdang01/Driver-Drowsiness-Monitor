from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ema_filter import EMAFilter
from fsm import DrowsinessSignals
from perclos import PERCLOSCalculator
from runtime.config import RuntimeConfig
from runtime.mar_detection import DynamicMAR
from runtime.perception import RawPerception


@dataclass
class FeatureState:
    calibrated: bool = False
    ear_threshold: float = 0.23
    mar_threshold: float = 0.60
    calibration_count: int = 0
    ear_baseline: float = 0.23
    mar_baseline: float = 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class SignalFeaturePipeline:
    """Turn per-frame raw perception values into temporal DMS signals.

    Key behavior:
    - EAR threshold is personalized during calibration, then locked until Reset calibration.
    - MAR/yawn threshold is adaptive via DynamicMAR.
    - Head nod is guarded by motion + drowsiness context to reduce false triggers on road bumps.
    - No gaze features are used.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.state = FeatureState(
            ear_threshold=config.thresholds.ear_default,
            mar_threshold=getattr(config.thresholds, "mar", 0.60),
        )

        self.ear_ema = EMAFilter(alpha=0.3)
        self.mar_ema = EMAFilter(alpha=0.3)
        self.pitch_ema = EMAFilter(alpha=0.3)

        self.dynamic_mar = DynamicMAR(
            calibration_frames=config.runtime.calibration_frames,
            alpha=0.3,
            factor=float(getattr(config.thresholds, "mar_factor", 4.0)),
            floor=float(getattr(config.thresholds, "mar_min", 0.05)),
            ceiling=float(getattr(config.thresholds, "mar_max", 0.80)),
            min_gap=float(getattr(config.thresholds, "mar_gap", 0.04)),
            adapt_alpha=float(getattr(config.thresholds, "mar_adapt_alpha", 0.01)),
        )

        self.perclos_long = PERCLOSCalculator(
            window_seconds=config.windows.perclos_seconds,
            fps=config.runtime.fps,
        )
        self.perclos_short = PERCLOSCalculator(
            window_seconds=config.windows.perclos_short_seconds,
            fps=config.runtime.fps,
        )

        self._pitch_samples: list[float] = []
        self._yaw_samples: list[float] = []
        self._ear_samples: list[float] = []
        self._mar_samples: list[float] = []
        self._base_pitch = 0.0
        self._base_yaw = 0.0

        self._prev_rel_pitch: float | None = None
        self._eyes_closed_previous = False
        self._eyes_closed_consecutive = 0

        self._blink_timestamps: deque[float] = deque(maxlen=1000)
        self._yawn_timestamps: deque[float] = deque(maxlen=1000)

        self._current_yawn_frames = 0
        self._head_nod_counter = 0
        self._head_motion_recent = 0

    def _finish_calibration(self) -> None:
        ear_arr = np.asarray(self._ear_samples, dtype=float)
        ear_arr = ear_arr[np.isfinite(ear_arr)]
        if ear_arr.size:
            baseline_ear = float(np.quantile(ear_arr, 0.80))
        else:
            baseline_ear = self.config.thresholds.ear_default

        self.state.ear_baseline = baseline_ear
        self.state.ear_threshold = _clamp(
            baseline_ear * self.config.thresholds.ear_calibration_factor,
            self.config.thresholds.ear_min,
            self.config.thresholds.ear_max,
        )

        mar_arr = np.asarray(self._mar_samples, dtype=float)
        mar_arr = mar_arr[np.isfinite(mar_arr)]
        self.state.mar_baseline = float(np.quantile(mar_arr, 0.50)) if mar_arr.size else 0.0
        self.state.mar_threshold = self.dynamic_mar.threshold

        self._base_pitch = float(np.mean(self._pitch_samples)) if self._pitch_samples else 0.0
        self._base_yaw = float(np.mean(self._yaw_samples)) if self._yaw_samples else 0.0
        self.state.calibrated = True

    def update(self, raw: RawPerception, now: float) -> tuple[DrowsinessSignals, dict[str, Any]]:
        if raw.face_detected and not self.state.calibrated:
            self._ear_samples.append(raw.ear)
            self._mar_samples.append(raw.mar)
            self._pitch_samples.append(raw.pitch)
            self._yaw_samples.append(raw.yaw)
            self.state.calibration_count += 1
            mar_status = self.dynamic_mar.update(raw.mar)
            self.state.mar_threshold = mar_status.threshold
            if self.state.calibration_count >= self.config.runtime.calibration_frames:
                self._finish_calibration()

        if raw.face_detected:
            rel_pitch = raw.pitch - self._base_pitch
            rel_yaw = raw.yaw - self._base_yaw
            ear_raw = float(raw.ear)
            mar_raw = float(raw.mar)
        else:
            rel_pitch = 0.0
            rel_yaw = 0.0
            ear_raw = self.state.ear_threshold + 0.02
            mar_raw = 0.0

        # Smooth values for display/decision stability.
        ear_smooth = self.ear_ema.update(ear_raw)
        mar_smooth = self.mar_ema.update(mar_raw)
        pitch_smooth = self.pitch_ema.update(rel_pitch)

        if self._prev_rel_pitch is None:
            pitch_velocity = 0.0
        else:
            pitch_velocity = pitch_smooth - self._prev_rel_pitch
        self._prev_rel_pitch = pitch_smooth

        mar_status = self.dynamic_mar.update(mar_raw) if raw.face_detected else self.dynamic_mar.update(0.0)
        self.state.mar_threshold = mar_status.threshold

        eye_closed = ear_raw < self.state.ear_threshold
        if eye_closed:
            self._eyes_closed_consecutive += 1
        else:
            if self._eyes_closed_previous and self._eyes_closed_consecutive >= 2:
                self._blink_timestamps.append(now)
            self._eyes_closed_consecutive = 0
        self._eyes_closed_previous = eye_closed

        mouth_open = bool(mar_status.is_open)
        if mouth_open:
            self._current_yawn_frames += 1
        else:
            if self._current_yawn_frames >= self.config.thresholds.yawn_frames:
                self._yawn_timestamps.append(now)
            self._current_yawn_frames = 0

        self._trim_timestamps(self._blink_timestamps, now, self.config.windows.blink_window_seconds)
        self._trim_timestamps(self._yawn_timestamps, now, self.config.windows.yawn_window_seconds)

        blink_frequency = len(self._blink_timestamps)
        yawn_frequency = len(self._yawn_timestamps)

        perclos = self.perclos_long.update(eye_closed)
        perclos_short = self.perclos_short.update(eye_closed)

        # Robust head-nod logic:
        # Old logic used only sustained pitch angle, so bumpy road or camera shake could be counted as nodding.
        # New logic requires: (1) head angle condition, (2) recent head motion, and (3) drowsiness context.
        pitch_velocity_abs = abs(float(pitch_velocity))
        velocity_gate = 0.55  # degrees/frame after EMA; avoids slow natural head pose changes
        motion_hold_frames = max(8, int(self.config.runtime.fps * 0.45))
        if pitch_velocity_abs >= velocity_gate:
            self._head_motion_recent = motion_hold_frames
        else:
            self._head_motion_recent = max(0, self._head_motion_recent - 1)

        head_angle_candidate = (
            rel_pitch > self.config.thresholds.pitch
            and abs(rel_yaw) < min(self.config.thresholds.head_yaw, 22.0)
            and not mouth_open
        )
        # Require real drowsiness context, not just one closed-eye frame.
        # This reduces false head-nod detection when the car hits a bump while the driver is awake.
        drowsiness_context = (
            self._eyes_closed_consecutive >= max(4, int(self.config.runtime.fps * 0.20))
            or perclos_short >= 0.35
        )
        head_nod_candidate = head_angle_candidate and self._head_motion_recent > 0 and drowsiness_context

        if head_nod_candidate:
            self._head_nod_counter += 1
        else:
            # Decay faster to avoid road-bump false positives accumulating over time.
            self._head_nod_counter = max(0, self._head_nod_counter - 3)

        # Do not require a full second here; the candidate is already gated by motion + eye/PERCLOS context.
        head_nod_required_frames = max(8, int(self.config.runtime.fps * 0.35))
        head_nod_detected = self._head_nod_counter >= head_nod_required_frames

        signals = DrowsinessSignals(
            ear=float(ear_smooth),
            mar=float(mar_smooth),
            pitch=float(pitch_smooth),
            pitch_velocity=float(pitch_velocity),
            perclos=float(perclos),
            perclos_short=float(perclos_short),
            yawn_frequency=int(yawn_frequency),
            blink_frequency=int(blink_frequency),
            head_nod_detected=bool(head_nod_detected),
            eyes_closed_consecutive=int(self._eyes_closed_consecutive),
            ear_below_threshold=bool(eye_closed),
            mar_above_threshold=bool(mouth_open),
            pitch_above_threshold=bool(head_angle_candidate),
        )

        debug = {
            "calibrated": self.state.calibrated,
            "ear_threshold": self.state.ear_threshold,
            "ear_baseline": self.state.ear_baseline,
            "ear_threshold_locked": self.state.calibrated,
            "ear_raw": ear_raw,
            "mar_raw": mar_raw,
            "mar_threshold": self.state.mar_threshold,
            "mar_baseline": self.state.mar_baseline,
            "mar_dynamic_mu": mar_status.mu,
            "mar_dynamic_sigma": mar_status.sigma,
            "mar_dynamic_progress": mar_status.progress,
            "pitch_raw": rel_pitch,
            "rel_yaw": rel_yaw,
            "blink_frequency": blink_frequency,
            "yawn_frequency": yawn_frequency,
            "current_yawn_frames": self._current_yawn_frames,
            "head_nod_counter": self._head_nod_counter,
            "head_motion_recent": self._head_motion_recent,
            "head_angle_candidate": head_angle_candidate,
            "head_nod_candidate": head_nod_candidate,
            "face_detected": raw.face_detected,
        }
        return signals, debug

    @staticmethod
    def _trim_timestamps(queue: deque[float], now: float, window_seconds: float) -> None:
        while queue and (now - queue[0]) > window_seconds:
            queue.popleft()
