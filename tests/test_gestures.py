"""
test_gestures.py
----------------
Unit tests for gesture recognition, classification rules, and stabilizer buffer.
"""

import unittest
from gestures import classify_gesture, GestureStabilizer, calculate_adaptive_pinch_dist


def make_dummy_landmarks(thumb_tip=(50, 50), index_tip=(100, 100)):
    """Helper to generate 21 hand landmarks with custom thumb and index tip coordinates."""
    lm = [[i, 0, 0] for i in range(21)]
    lm[4] = [4, thumb_tip[0], thumb_tip[1]]
    lm[8] = [8, index_tip[0], index_tip[1]]
    return lm


class TestGestureClassification(unittest.TestCase):
    def test_fist_gesture(self):
        lm = make_dummy_landmarks()
        g, _ = classify_gesture(fingers=[0, 0, 0, 0, 0], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "FIST")

    def test_open_palm_idle_and_swipes(self):
        lm = make_dummy_landmarks()
        g, _ = classify_gesture(fingers=[1, 1, 1, 1, 1], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "OPEN_PALM")

        # Swipe Right
        g, _ = classify_gesture(fingers=[1, 1, 1, 1, 1], lm=lm, vx=65, vy=0, swipe_vel=50)
        self.assertEqual(g, "PALM_R")

        # Swipe Left
        g, _ = classify_gesture(fingers=[1, 1, 1, 1, 1], lm=lm, vx=-65, vy=0, swipe_vel=50)
        self.assertEqual(g, "PALM_L")

        # Swipe Down
        g, _ = classify_gesture(fingers=[1, 1, 1, 1, 1], lm=lm, vx=0, vy=65, swipe_vel=50)
        self.assertEqual(g, "PALM_D")

    def test_thumb_up_and_shaka(self):
        lm = make_dummy_landmarks()
        g, _ = classify_gesture(fingers=[1, 0, 0, 0, 0], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "THUMB_UP")

        g, _ = classify_gesture(fingers=[1, 0, 0, 0, 1], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "BRIGHTNESS")

    def test_screenshot(self):
        lm = make_dummy_landmarks()
        g, _ = classify_gesture(fingers=[0, 1, 1, 1, 0], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "SCREENSHOT")

    def test_two_finger_scroll_and_rclick(self):
        lm = make_dummy_landmarks()
        # Still -> RCLICK
        g, _ = classify_gesture(fingers=[0, 1, 1, 0, 0], lm=lm, vx=0, vy=0)
        self.assertEqual(g, "RCLICK")

        # Moving down -> SCROLL_D
        g, _ = classify_gesture(fingers=[0, 1, 1, 0, 0], lm=lm, vx=0, vy=30, scroll_vel=20)
        self.assertEqual(g, "SCROLL_D")

        # Moving up -> SCROLL_U
        g, _ = classify_gesture(fingers=[0, 1, 1, 0, 0], lm=lm, vx=0, vy=-30, scroll_vel=20)
        self.assertEqual(g, "SCROLL_U")

    def test_index_move_and_pinch(self):
        # Far apart tips -> MOVE
        lm_far = make_dummy_landmarks(thumb_tip=(0, 0), index_tip=(100, 100))
        g, _ = classify_gesture(fingers=[0, 1, 0, 0, 0], lm=lm_far, vx=0, vy=0, pinch_dist=42)
        self.assertEqual(g, "MOVE")

        # Close tips -> PINCH
        lm_close = make_dummy_landmarks(thumb_tip=(50, 50), index_tip=(55, 55))
        g, _ = classify_gesture(fingers=[0, 1, 0, 0, 0], lm=lm_close, vx=0, vy=0, pinch_dist=42)
        self.assertEqual(g, "PINCH")

    def test_gesture_stabilizer(self):
        stab = GestureStabilizer(window_size=3)
        self.assertEqual(stab.update("MOVE"), "IDLE")  # 1st sample
        self.assertEqual(stab.update("MOVE"), "IDLE")  # 2nd sample
        self.assertEqual(stab.update("MOVE"), "MOVE")  # 3rd sample: majority MOVE

        # Single frame glitch to FIST should be suppressed
        self.assertEqual(stab.update("FIST"), "MOVE")

    def test_adaptive_pinch_distance(self):
        # Normal bounding box
        d_norm = calculate_adaptive_pinch_dist((50, 50, 160, 200), base_dist=42.0)
        self.assertAlmostEqual(d_norm, 42.0)

        # Hand far away (small bbox)
        d_small = calculate_adaptive_pinch_dist((50, 50, 80, 100), base_dist=42.0)
        self.assertLess(d_small, 42.0)
        self.assertGreaterEqual(d_small, 28.0)


if __name__ == "__main__":
    unittest.main()
