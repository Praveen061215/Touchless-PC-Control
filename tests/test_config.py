"""
test_config.py
--------------
Unit tests for configuration manager.
"""

import os
import tempfile
import unittest
from config import AppConfig, CameraConfig, GestureConfig, UIConfig


class TestConfig(unittest.TestCase):
    def test_default_values(self):
        cfg = AppConfig()
        self.assertEqual(cfg.camera.width, 640)
        self.assertEqual(cfg.camera.height, 480)
        self.assertEqual(cfg.gesture.pinch_distance, 42)
        self.assertTrue(cfg.ui.show_hud)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "test_config.json")
            cfg = AppConfig(
                camera=CameraConfig(width=1280, height=720),
                gesture=GestureConfig(pinch_distance=50),
                ui=UIConfig(sound_feedback=True)
            )
            cfg.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded = AppConfig.load(tmp_path)
            self.assertEqual(loaded.camera.width, 1280)
            self.assertEqual(loaded.camera.height, 720)
            self.assertEqual(loaded.gesture.pinch_distance, 50)
            self.assertTrue(loaded.ui.sound_feedback)

    def test_load_fallback_on_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "corrupt.json")
            with open(tmp_path, "w") as f:
                f.write("{invalid_json: true")
            
            loaded = AppConfig.load(tmp_path)
            self.assertIsInstance(loaded, AppConfig)
            self.assertEqual(loaded.camera.width, 640)


if __name__ == "__main__":
    unittest.main()
