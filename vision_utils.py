"""
vision_utils.py
---------------
Image processing and visual enhancement utilities.
Provides adaptive low-light enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE)
on the luminance channel of LAB color space, boosting landmark detection in dim environments.
"""

import cv2
import numpy as np


class LowLightEnhancer:
    """Enhances webcam frames under low-light or poor contrast lighting conditions."""

    def __init__(self, enabled: bool = False, alpha: float = 1.8, beta: int = 40, clahe_clip: float = 2.5):
        self.enabled = enabled
        self.alpha = float(alpha)
        self.beta = int(beta)
        self.clahe_clip = float(clahe_clip)
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))

    def toggle(self) -> bool:
        """Toggle low-light enhancement mode on or off."""
        self.enabled = not self.enabled
        return self.enabled

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance brightness and local contrast if enabled.
        Returns the original frame unchanged if disabled.
        """
        if not self.enabled or img is None or img.size == 0:
            return img

        # 1. Scale contrast and brightness linearly
        scaled = cv2.convertScaleAbs(img, alpha=self.alpha, beta=self.beta)

        # 2. Apply CLAHE strictly on the Luminance (L) channel to avoid color distortion
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)

        # 3. Merge channels and return back in BGR space
        merged = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
