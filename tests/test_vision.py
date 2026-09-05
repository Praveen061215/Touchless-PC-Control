"""
test_vision.py
--------------
Unit tests for vision enhancement tools (LowLightEnhancer).
"""

import unittest
import numpy as np
from vision_utils import LowLightEnhancer


class TestVisionUtils(unittest.TestCase):
    def test_enhancer_disabled_by_default(self):
        enhancer = LowLightEnhancer(enabled=False)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 30
        result = enhancer.enhance(frame)
        np.testing.assert_array_equal(frame, result)

    def test_enhancer_toggle(self):
        enhancer = LowLightEnhancer(enabled=False)
        self.assertFalse(enhancer.enabled)
        state = enhancer.toggle()
        self.assertTrue(state)
        self.assertTrue(enhancer.enabled)

    def test_enhancer_boosts_luminance(self):
        enhancer = LowLightEnhancer(enabled=True, alpha=1.5, beta=30)
        # Low-light frame with dark pixels (value 20)
        dark_frame = np.ones((64, 64, 3), dtype=np.uint8) * 20
        enhanced = enhancer.enhance(dark_frame)
        self.assertEqual(enhanced.shape, dark_frame.shape)
        # Average brightness should be significantly higher
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(dark_frame)))


if __name__ == "__main__":
    unittest.main()
