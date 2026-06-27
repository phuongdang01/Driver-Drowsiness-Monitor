"""
PERCLOS Calculator
Thesis §2.2, §3.4: Percentage of time eyes are closed over a sliding window.
A cumulative measure of vigilance degradation, more robust than single-frame events.
"""

from collections import deque


class PERCLOSCalculator:
    """Calculates PERCLOS over a sliding time window."""

    def __init__(self, window_seconds: float = 60.0, fps: float = 30.0):
        """
        Args:
            window_seconds: Duration of the sliding window in seconds.
            fps: Expected frames per second for window sizing.
        """
        self.window_seconds = window_seconds
        self.fps = fps
        self.window_size = max(1, int(window_seconds * fps))
        self.eye_closed_flags = deque(maxlen=self.window_size)

    def update(self, eye_closed: bool) -> float:
        """
        Add a new frame's eye state and return current PERCLOS value.

        Args:
            eye_closed: True if eyes are detected as closed this frame.

        Returns:
            PERCLOS value in [0, 1] — fraction of window with eyes closed.
        """
        self.eye_closed_flags.append(1 if eye_closed else 0)
        if len(self.eye_closed_flags) == 0:
            return 0.0
        return sum(self.eye_closed_flags) / len(self.eye_closed_flags)

    def reset(self):
        """Reset the calculator state."""
        self.eye_closed_flags.clear()

    @property
    def current_value(self) -> float:
        """Get current PERCLOS without adding a new sample."""
        if len(self.eye_closed_flags) == 0:
            return 0.0
        return sum(self.eye_closed_flags) / len(self.eye_closed_flags)
