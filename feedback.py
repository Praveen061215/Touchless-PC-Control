"""
feedback.py
-----------
Visual and auditory sensory feedback for Touchless PC Control.
Provides animated ripple effects for clicks and low-latency asynchronous audio cues.
"""

import threading
import platform
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np


class RippleEffect:
    """Represents a single expanding and fading circular ripple on screen."""

    def __init__(self, x: int, y: int, color: Tuple[int, int, int] = (0, 230, 80),
                 max_radius: int = 32, speed: int = 4):
        self.x = int(x)
        self.y = int(y)
        self.color = color
        self.radius = 6
        self.max_radius = max_radius
        self.speed = speed
        self.alpha = 1.0

    def update(self) -> bool:
        """Advance ripple radius and fade alpha. Returns True if ripple is still alive."""
        self.radius += self.speed
        progress = self.radius / self.max_radius
        self.alpha = max(0.0, 1.0 - progress)
        return self.radius < self.max_radius

    def draw(self, img: np.ndarray) -> None:
        """Draw ripple on image using alpha blending."""
        if self.alpha <= 0.05:
            return
        overlay = img.copy()
        cv2.circle(overlay, (self.x, self.y), int(self.radius), self.color, 2)
        cv2.addWeighted(overlay, self.alpha, img, 1.0 - self.alpha, 0, img)


class FeedbackManager:
    """Coordinates visual ripple rings and asynchronous auditory beeps."""

    def __init__(self, sound_enabled: bool = False):
        self.sound_enabled = sound_enabled
        self.ripples: List[RippleEffect] = []

    def add_ripple(self, x: int, y: int, color: Tuple[int, int, int] = (0, 230, 80)) -> None:
        """Add an expanding ripple ring at coordinate (x, y)."""
        self.ripples.append(RippleEffect(x, y, color=color))

    def update_and_draw(self, img: np.ndarray) -> np.ndarray:
        """Update and draw all active ripples in-place."""
        alive_ripples = []
        for r in self.ripples:
            if r.update():
                r.draw(img)
                alive_ripples.append(r)
        self.ripples = alive_ripples
        return img

    def play_sound_async(self, sound_type: str = "click") -> None:
        """Emit auditory beep asynchronously to avoid blocking the main vision loop."""
        if not self.sound_enabled:
            return

        def _worker():
            if platform.system() == "Windows":
                try:
                    import winsound
                    if sound_type == "click":
                        winsound.Beep(1200, 35)
                    elif sound_type == "rclick":
                        winsound.Beep(900, 45)
                    elif sound_type == "action":
                        winsound.Beep(1500, 50)
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

    def trigger_click(self, x: int, y: int, is_right: bool = False) -> None:
        """Convenience method to trigger both visual and auditory click cues."""
        color = (220, 0, 255) if is_right else (0, 230, 80)
        self.add_ripple(x, y, color=color)
        self.play_sound_async("rclick" if is_right else "click")
