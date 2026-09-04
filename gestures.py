"""
gestures.py
-----------
Core gesture recognition, classification rules, and noise stabilization.
Designed as pure functions and stateful helper classes decoupled from camera hardware.
"""

import math
from collections import deque
from typing import List, Tuple, Dict, Any, Optional


def landmark_distance(lm: List[List[int]], id1: int, id2: int) -> float:
    """Euclidean pixel distance between two landmark coordinates [id, x, y]."""
    return math.hypot(lm[id1][1] - lm[id2][1], lm[id1][2] - lm[id2][2])


def calculate_adaptive_pinch_dist(bbox: Tuple[int, int, int, int], base_dist: float = 42.0) -> float:
    """
    Scale pinch distance threshold proportionally to hand bounding box width
    so gestures remain reliable at varying distances from the camera lens.
    """
    _, _, bw, _ = bbox
    # Base tuned for typical hand bounding-box width ~160px
    return max(28.0, min(65.0, base_dist * (bw / 160.0)))


def classify_gesture(
    fingers: List[int],
    lm: List[List[int]],
    vx: float,
    vy: float,
    pinch_dist: float = 42.0,
    swipe_vel: float = 55.0,
    scroll_vel: float = 22.0
) -> Tuple[str, float]:
    """
    Classify hand pose into gesture token and pinch distance.
    Rules are evaluated in priority order to guarantee deterministic resolution.
    
    Parameters
    ----------
    fingers : list[int]
        [thumb, index, middle, ring, pinky] state (1=up, 0=down).
    lm : list[list[int]]
        List of 21 hand landmarks [[id, x, y], ...].
    vx, vy : float
        Palm movement velocity / displacement over recent history.
    pinch_dist, swipe_vel, scroll_vel : float
        Threshold parameters from config.
        
    Returns
    -------
    gesture : str
        Identified gesture token.
    pd : float
        Distance between thumb tip (id 4) and index tip (id 8).
    """
    pd = landmark_distance(lm, 4, 8)
    num_up = sum(fingers)

    # 1. Fist (all fingers down)
    if num_up == 0:
        return "FIST", pd

    # 2. Open palm (all fingers up)
    if num_up == 5:
        if abs(vy) > swipe_vel and abs(vy) > abs(vx) and vy > 0:
            return "PALM_D", pd
        if abs(vx) > swipe_vel and abs(vx) > abs(vy):
            return ("PALM_R" if vx > 0 else "PALM_L"), pd
        return "OPEN_PALM", pd

    # 3. Thumb only -> Confirm
    if fingers == [1, 0, 0, 0, 0]:
        return "THUMB_UP", pd

    # 4. Thumb + Pinky (Shaka) -> Brightness control
    if fingers == [1, 0, 0, 0, 1]:
        return "BRIGHTNESS", pd

    # 5. Index + Middle + Ring (3 fingers) -> Screenshot
    if not fingers[0] and fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
        return "SCREENSHOT", pd

    # 6. Thumb + Index only -> Zoom (pinched) or Volume (L-shape)
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return ("ZOOM" if pd < pinch_dist else "VOLUME"), pd

    # 7. Index + Middle (Two fingers) -> Scroll or Right Click
    if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        if abs(vy) > scroll_vel:
            return ("SCROLL_U" if vy < 0 else "SCROLL_D"), pd
        return "RCLICK", pd

    # 8. Index up (Thumb optional/irrelevant) -> Move, Swipe, or Pinch
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        if pd < pinch_dist:
            return "PINCH", pd
        if abs(vx) > swipe_vel and abs(vx) > abs(vy):
            return ("SWIPE_R" if vx > 0 else "SWIPE_L"), pd
        return "MOVE", pd

    return "IDLE", pd


class GestureStabilizer:
    """Majority-vote ring buffer to filter single-frame sensor flutter and noise."""

    def __init__(self, window_size: int = 4):
        self.window_size = max(1, window_size)
        self.buffer = deque(maxlen=self.window_size)
        self.current_stable = "IDLE"

    def update(self, raw_gesture: str) -> str:
        self.buffer.append(raw_gesture)
        if len(self.buffer) < self.window_size:
            return self.current_stable

        counts: Dict[str, int] = {}
        for g in self.buffer:
            counts[g] = counts.get(g, 0) + 1

        winner = max(counts, key=lambda k: counts[k])
        self.current_stable = winner
        return winner

    def reset(self) -> None:
        self.buffer.clear()
        self.current_stable = "IDLE"
