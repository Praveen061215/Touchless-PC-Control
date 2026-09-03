"""
filters.py
----------
Adaptive signal filtering algorithms for human-computer interaction.
Implements the 1€ Filter (Casiez et al., CHI 2012) to dynamically eliminate
jitter during stationary poses while preserving low latency during rapid movements.
"""

import math
import time
from typing import Tuple, Optional


class LowPassFilter:
    """Standard first-order exponential low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.set_alpha(alpha)
        self.y: Optional[float] = None
        self.s: Optional[float] = None

    def set_alpha(self, alpha: float) -> None:
        self.alpha = max(1e-4, min(1.0, float(alpha)))

    def filter(self, value: float) -> float:
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        return self.s

    def last_value(self) -> Optional[float]:
        return self.s

    def reset(self) -> None:
        self.y = None
        self.s = None


class OneEuroFilter:
    """
    1€ Filter: Adaptive Low-Pass Filter with dynamic cutoff frequency.
    
    Reference:
        Casiez, G., Roussel, N., & Vogel, D. (2012).
        1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems.
        ACM CHI 2012.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.freq = max(1.0, float(freq))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_filter = LowPassFilter(self._alpha(self.min_cutoff))
        self._dx_filter = LowPassFilter(self._alpha(self.d_cutoff))
        self._last_time: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * max(1e-4, cutoff))
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if timestamp is not None and self._last_time is not None:
            dt = timestamp - self._last_time
            if dt > 1e-5:
                self.freq = 1.0 / dt
        self._last_time = timestamp

        prev_x = self._x_filter.y
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq

        self._dx_filter.set_alpha(self._alpha(self.d_cutoff))
        edx = self._dx_filter.filter(dx)

        cutoff = self.min_cutoff + self.beta * abs(edx)
        self._x_filter.set_alpha(self._alpha(cutoff))
        return self._x_filter.filter(x)

    def reset(self) -> None:
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None


class PointFilter2D:
    """Applies coordinate smoothing to 2D (x, y) coordinates."""

    def __init__(self, filter_type: str = "one-euro", smoothening: float = 7.0,
                 min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0, freq: float = 30.0):
        self.filter_type = filter_type.lower()
        self.smoothening = smoothening

        self.filter_x = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)

        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        if self.filter_type == "one-euro":
            fx = self.filter_x.filter(x, timestamp)
            fy = self.filter_y.filter(y, timestamp)
            self.prev_x, self.prev_y = fx, fy
            return fx, fy

        elif self.filter_type == "ema":
            if self.prev_x is None or self.prev_y is None:
                self.prev_x, self.prev_y = x, y
            else:
                self.prev_x += (x - self.prev_x) / self.smoothening
                self.prev_y += (y - self.prev_y) / self.smoothening
            return self.prev_x, self.prev_y

        # 'none' or raw fallback
        self.prev_x, self.prev_y = x, y
        return x, y

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()
        self.prev_x = None
        self.prev_y = None
