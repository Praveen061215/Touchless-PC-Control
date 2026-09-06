"""
test_feedback.py
----------------
Unit tests for visual and auditory feedback manager.
"""

import unittest
import numpy as np
from feedback import RippleEffect, FeedbackManager


class TestFeedback(unittest.TestCase):
    def test_ripple_lifecycle(self):
        ripple = RippleEffect(50, 50, max_radius=20, speed=5)
        self.assertEqual(ripple.radius, 6)
        alive = ripple.update()
        self.assertTrue(alive)
        self.assertEqual(ripple.radius, 11)

        # Fast forward till finished
        while alive:
            alive = ripple.update()
        self.assertFalse(alive)
        self.assertGreaterEqual(ripple.radius, 20)

    def test_feedback_manager_ripples(self):
        fm = FeedbackManager(sound_enabled=False)
        fm.add_ripple(100, 100)
        self.assertEqual(len(fm.ripples), 1)

        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        out = fm.update_and_draw(dummy_img)
        self.assertEqual(out.shape, dummy_img.shape)

    def test_feedback_manager_trigger_click(self):
        fm = FeedbackManager(sound_enabled=False)
        fm.trigger_click(150, 150, is_right=True)
        self.assertEqual(len(fm.ripples), 1)
        self.assertEqual(fm.ripples[0].color, (220, 0, 255))


if __name__ == "__main__":
    unittest.main()
