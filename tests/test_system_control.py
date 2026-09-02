"""
test_system_control.py
----------------------
Unit tests for system_control module (AudioController, BrightnessController).
"""

import unittest
from system_control import AudioController, BrightnessController


class TestSystemControl(unittest.TestCase):
    def test_audio_controller_initialization(self):
        audio = AudioController()
        # Should initialize without raising exceptions
        self.assertIsInstance(audio.available, bool)
        self.assertGreaterEqual(audio.get_volume_scalar(), 0.0)
        self.assertLessEqual(audio.get_volume_scalar(), 1.0)

    def test_audio_controller_clamping(self):
        audio = AudioController()
        # Test scalar clamping logic
        audio.set_volume_scalar(-0.5)
        self.assertGreaterEqual(audio.get_volume_scalar(), 0.0)
        audio.set_volume_scalar(1.5)
        self.assertLessEqual(audio.get_volume_scalar(), 1.0)

    def test_brightness_controller_initialization(self):
        brightness = BrightnessController()
        self.assertIsInstance(brightness.available, bool)
        level = brightness.get_brightness()
        self.assertGreaterEqual(level, 0)
        self.assertLessEqual(level, 100)

    def test_brightness_controller_clamping(self):
        brightness = BrightnessController()
        brightness.set_brightness(-10)
        self.assertGreaterEqual(brightness.get_brightness(), 0)
        brightness.set_brightness(150)
        self.assertLessEqual(brightness.get_brightness(), 100)


if __name__ == "__main__":
    unittest.main()
