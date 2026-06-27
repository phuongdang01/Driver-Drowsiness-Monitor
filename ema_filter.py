"""
Exponential Moving Average (EMA) Filter
Thesis §3.6: Smooths noisy signals (EAR, MAR, head pitch) to reduce false triggers.
Formula: x̂_t = α·x_t + (1-α)·x̂_{t-1}
"""


class EMAFilter:
    """Exponential Moving Average filter for smoothing time-series signals."""

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 < alpha < 1).
                   Higher = more responsive, lower = smoother.
                   0.3 is a good default for drowsiness signals.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.value = None

    def update(self, raw_value: float) -> float:
        """Update filter with new raw value and return smoothed value."""
        if self.value is None:
            self.value = raw_value
        else:
            self.value = self.alpha * raw_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset filter state."""
        self.value = None

    @property
    def is_initialized(self) -> bool:
        return self.value is not None
