"""
config.py
---------
Configuration management for Touchless PC Control.
Supports loading and saving settings via JSON, with robust fallback defaults.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    flip_horizontal: bool = True


@dataclass
class GestureConfig:
    control_margin: int = 80             # frameR margin in px
    filter_type: str = "one-euro"        # 'one-euro', 'ema', or 'none'
    smoothening: float = 7.0             # EMA smoothing fallback factor
    one_euro_min_cutoff: float = 1.0     # Min cutoff frequency (Hz) for OneEuroFilter
    one_euro_beta: float = 0.007         # Speed coefficient for OneEuroFilter
    one_euro_d_cutoff: float = 1.0       # Cutoff frequency for derivative

    pinch_distance: int = 42             # px thumb-to-index distance for pinch
    drag_min_distance: int = 16          # px movement required to initiate drag
    gesture_stability_frames: int = 4    # majority-vote window size
    rclick_hold_frames: int = 6          # frames to hold 2 fingers for right-click

    click_cooldown: float = 0.40         # seconds
    rclick_cooldown: float = 0.60
    scroll_cooldown: float = 0.07
    action_cooldown: float = 0.80
    volume_cooldown: float = 0.14
    zoom_cooldown: float = 0.10

    swipe_velocity_threshold: float = 55.0
    scroll_velocity_threshold: float = 22.0


@dataclass
class UIConfig:
    show_hud: bool = True
    show_bounding_box: bool = True
    show_landmarks: bool = True
    low_light_mode: bool = False
    low_light_alpha: float = 1.8         # Contrast multiplier
    low_light_beta: int = 40             # Brightness offset
    low_light_clahe_clip: float = 2.5    # CLAHE clip limit
    sound_feedback: bool = False         # Beep on click/actions


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def load(cls, filepath: str = "config.json") -> "AppConfig":
        """Load configuration from a JSON file, or return defaults if not found."""
        if not os.path.exists(filepath):
            config = cls()
            config.save(filepath)
            return config

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            camera_data = data.get("camera", {})
            gesture_data = data.get("gesture", {})
            ui_data = data.get("ui", {})

            return cls(
                camera=CameraConfig(**{k: v for k, v in camera_data.items() if k in CameraConfig.__dataclass_fields__}),
                gesture=GestureConfig(**{k: v for k, v in gesture_data.items() if k in GestureConfig.__dataclass_fields__}),
                ui=UIConfig(**{k: v for k, v in ui_data.items() if k in UIConfig.__dataclass_fields__})
            )
        except Exception as e:
            print(f"[WARN] Failed to load {filepath} ({e}). Using default config.")
            return cls()

    def save(self, filepath: str = "config.json") -> None:
        """Save current configuration to a JSON file with pretty indentation."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4)
        except Exception as e:
            print(f"[WARN] Failed to save {filepath}: {e}")


# Default global instance
default_config = AppConfig()
