"""
test_filters.py
---------------
Unit tests for adaptive signal filters (OneEuroFilter, LowPassFilter, PointFilter2D).
"""

import unittest
import numpy as np
from filters import LowPassFilter, OneEuroFilter, PointFilter2D


class TestFilters(unittest.TestCase):
    def test_low_pass_filter(self):
        lpf = LowPassFilter(alpha=0.5)
        # First sample should return itself
        self.assertEqual(lpf.filter(10.0), 10.0)
        # Second sample should be 0.5 * 20 + 0.5 * 10 = 15.0
        self.assertAlmostEqual(lpf.filter(20.0), 15.0)

    def test_one_euro_filter_reduces_jitter(self):
        filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.007)
        # Simulate noisy stationary signal around 100.0
        np.random.seed(42)
        raw_signals = [100.0 + np.random.normal(0, 1.5) for _ in range(50)]
        filtered_signals = []
        for i, val in enumerate(raw_signals):
            filtered_signals.append(filt.filter(val, timestamp=i * (1.0 / 30.0)))

        raw_std = np.std(raw_signals[10:])
        filtered_std = np.std(filtered_signals[10:])
        # Filtered variance should be significantly smaller than raw noise
        self.assertLess(filtered_std, raw_std)

    def test_one_euro_filter_rapid_motion(self):
        filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05)
        # Fast step change from 0 to 500
        t0 = 0.0
        filt.filter(0.0, timestamp=t0)
        out = filt.filter(500.0, timestamp=t0 + 0.033)
        # High beta should allow rapid tracking without being bogged down
        self.assertGreater(out, 150.0)

    def test_point_filter_2d(self):
        pf = PointFilter2D(filter_type="one-euro")
        x, y = pf.filter(100.0, 200.0, timestamp=0.0)
        self.assertEqual(x, 100.0)
        self.assertEqual(y, 200.0)

        # EMA mode test
        pf_ema = PointFilter2D(filter_type="ema", smoothening=5.0)
        x1, y1 = pf_ema.filter(10.0, 20.0)
        x2, y2 = pf_ema.filter(20.0, 30.0)
        self.assertAlmostEqual(x2, 12.0)
        self.assertAlmostEqual(y2, 22.0)


if __name__ == "__main__":
    unittest.main()
