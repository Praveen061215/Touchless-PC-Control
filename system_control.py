"""
system_control.py
-----------------
Cross-platform system control utilities for master audio volume and display brightness.
Includes defensive fallbacks and status checks to prevent application crashes on unsupported platforms.
"""

import logging
import platform

logger = logging.getLogger(__name__)


class AudioController:
    """Manages system master audio volume with PyCaw on Windows and safe fallbacks."""

    def __init__(self):
        self.available = False
        self._volume_ctrl = None
        self.min_vol = -65.25
        self.max_vol = 0.0
        self._current_scalar = 0.5

        if platform.system() == "Windows":
            try:
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                speakers = AudioUtilities.GetSpeakers()
                if hasattr(speakers, "EndpointVolume"):
                    self._volume_ctrl = speakers.EndpointVolume
                else:
                    from ctypes import cast, POINTER
                    from comtypes import CLSCTX_ALL
                    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    self._volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

                vol_range = self._volume_ctrl.GetVolumeRange()
                self.min_vol, self.max_vol = vol_range[0], vol_range[1]
                self.available = True
                logger.info("System audio controller initialized successfully.")
            except Exception as e:
                logger.warning(f"Audio controller initialization failed: {e}")
                self.available = False
        else:
            logger.info("Audio controller: Non-Windows platform detected, using fallback stub.")

    def set_volume_scalar(self, scalar: float) -> bool:
        """
        Set master volume using a normalized scalar (0.0 to 1.0).
        Returns True if successful, False otherwise.
        """
        clamped = max(0.0, min(1.0, float(scalar)))
        self._current_scalar = clamped

        if not self.available or self._volume_ctrl is None:
            return False

        try:
            import numpy as np
            vol_db = float(np.interp(clamped * 100.0, [0.0, 100.0], [self.min_vol, self.max_vol]))
            self._volume_ctrl.SetMasterVolumeLevel(vol_db, None)
            return True
        except Exception as e:
            logger.debug(f"Error setting volume: {e}")
            return False

    def get_volume_scalar(self) -> float:
        """Get current volume as a normalized scalar between 0.0 and 1.0."""
        if not self.available or self._volume_ctrl is None:
            return self._current_scalar

        try:
            curr_db = self._volume_ctrl.GetMasterVolumeLevel()
            import numpy as np
            return float(np.interp(curr_db, [self.min_vol, self.max_vol], [0.0, 1.0]))
        except Exception:
            return self._current_scalar


class BrightnessController:
    """Manages screen brightness via screen_brightness_control with safe fallbacks."""

    def __init__(self):
        self.available = False
        self._current_level = 50

        try:
            import screen_brightness_control as sbc
            self._sbc = sbc
            current = self._sbc.get_brightness()
            if current and len(current) > 0:
                self._current_level = int(current[0])
            self.available = True
            logger.info("Screen brightness controller initialized successfully.")
        except Exception as e:
            logger.warning(f"Brightness controller initialization failed: {e}")
            self._sbc = None
            self.available = False

    def set_brightness(self, level: int) -> bool:
        """
        Set screen brightness as an integer percentage (0 to 100).
        Returns True if successful, False otherwise.
        """
        clamped = max(0, min(100, int(level)))
        self._current_level = clamped

        if not self.available or self._sbc is None:
            return False

        try:
            self._sbc.set_brightness(clamped)
            return True
        except Exception as e:
            logger.debug(f"Error setting brightness: {e}")
            return False

    def get_brightness(self) -> int:
        """Get current display brightness percentage."""
        if not self.available or self._sbc is None:
            return self._current_level

        try:
            vals = self._sbc.get_brightness()
            if vals and len(vals) > 0:
                self._current_level = int(vals[0])
            return self._current_level
        except Exception:
            return self._current_level
