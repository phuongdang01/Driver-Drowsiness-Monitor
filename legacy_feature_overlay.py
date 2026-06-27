"""Optional legacy feature overlay utilities.

This module preserves bundle-style temporal metrics for debug/telemetry only.
It is intentionally decoupled from FSM state decisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
import time

import numpy as np


@dataclass
class LegacyFeatureSnapshot:
    max_closure_duration: float
    closure_count_long: int
    blink_rate: float
    perclos: float
    yawn_rate: float
    nod_rate: float
    avg_ear: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "max_closure_duration": self.max_closure_duration,
            "closure_count_long": float(self.closure_count_long),
            "blink_rate": self.blink_rate,
            "perclos": self.perclos,
            "yawn_rate": self.yawn_rate,
            "nod_rate": self.nod_rate,
            "avg_ear": self.avg_ear,
        }


class DrowsinessFeatureExtractor:
    """Legacy temporal feature extractor from bundle runtime.

    Inputs are per-frame measurements and boolean events. Output is a compact
    temporal snapshot that can be rendered as overlay text.
    """

    def __init__(
        self,
        window_duration: float = 60.0,
        ear_threshold: float = 0.23,
        closure_time_threshold: float = 0.5,
    ) -> None:
        self.window_duration = window_duration
        self.ear_threshold = ear_threshold
        self.closure_time_threshold = closure_time_threshold
        self.data_buffer: Deque[Tuple[float, float, float, float, float, float, bool, bool, bool]] = deque()

    def update(
        self,
        *,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        roll: float,
        is_blink: bool,
        is_yawn: bool,
        is_nod: bool,
    ) -> None:
        current_time = time.time()
        self.data_buffer.append(
            (current_time, ear, mar, pitch, yaw, roll, is_blink, is_yawn, is_nod)
        )

        while self.data_buffer and current_time - self.data_buffer[0][0] > self.window_duration:
            self.data_buffer.popleft()

    def get_features(self) -> LegacyFeatureSnapshot:
        if not self.data_buffer:
            return LegacyFeatureSnapshot(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        data = list(self.data_buffer)
        times = [x[0] for x in data]
        ears = [x[1] for x in data]
        is_blinks = [x[6] for x in data]
        is_yawns = [x[7] for x in data]
        is_nods = [x[8] for x in data]
        duration = times[-1] - times[0] if times[-1] - times[0] > 0 else 1.0
        avg_frame_time = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else (1.0 / 30.0)

        max_closure_duration = 0.0
        current_closure = 0.0
        closure_count_long = 0
        for ear in ears:
            if ear < self.ear_threshold:
                current_closure += avg_frame_time
            else:
                if current_closure > max_closure_duration:
                    max_closure_duration = current_closure
                if current_closure > self.closure_time_threshold:
                    closure_count_long += 1
                current_closure = 0.0

        if current_closure > max_closure_duration:
            max_closure_duration = current_closure
        if current_closure > self.closure_time_threshold:
            closure_count_long += 1

        blink_count = int(sum(1 for x in is_blinks if x))
        blink_rate = (blink_count / duration) * 60.0

        frames_closed = sum(1 for e in ears if e < self.ear_threshold)
        perclos = frames_closed / len(ears) if ears else 0.0

        yawn_count = int(sum(1 for x in is_yawns if x))
        yawn_rate = (yawn_count / duration) * 60.0

        nod_count = int(sum(1 for x in is_nods if x))
        nod_rate = (nod_count / duration) * 60.0

        return LegacyFeatureSnapshot(
            max_closure_duration=max_closure_duration,
            closure_count_long=closure_count_long,
            blink_rate=blink_rate,
            perclos=perclos,
            yawn_rate=yawn_rate,
            nod_rate=nod_rate,
            avg_ear=float(np.mean(ears)),
        )


def format_overlay_lines(snapshot: LegacyFeatureSnapshot) -> List[str]:
    """Prepare compact overlay strings for debug rendering."""
    return [
        f"Legacy CloseMax: {snapshot.max_closure_duration:.2f}s",
        f"Legacy PERCLOS: {snapshot.perclos:.2f}",
        f"Legacy BlinkRate: {snapshot.blink_rate:.1f}/m",
        f"Legacy YawnRate: {snapshot.yawn_rate:.1f}/m",
        f"Legacy NodRate: {snapshot.nod_rate:.1f}/m",
    ]
