"""
cli.py
------
Command-line interface and diagnostic utilities for Touchless PC Control.
Provides argument parsing and hardware environment diagnostics.
"""

import argparse
import os
import platform
import sys
import cv2
import pyautogui
from config import AppConfig
from system_control import AudioController, BrightnessController


def parse_arguments(args=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Touchless PC Control -- Advanced Virtual Mouse via Computer Vision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", "-c", type=str, default="config.json",
                        help="Path to JSON configuration file")
    parser.add_argument("--camera", "-cam", type=int, default=None,
                        help="Webcam device index (e.g. 0, 1)")
    parser.add_argument("--width", type=int, default=None,
                        help="Camera capture width")
    parser.add_argument("--height", type=int, default=None,
                        help="Camera capture height")
    parser.add_argument("--filter", choices=["one-euro", "ema", "none"], default=None,
                        help="Smoothing filter algorithm for cursor movement")
    parser.add_argument("--no-hud", action="store_true",
                        help="Run without displaying the OpenCV HUD preview window")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run hardware and environment diagnostics report and exit")
    return parser.parse_args(args)


def run_diagnostics() -> int:
    """Perform comprehensive checks on cameras, audio, brightness, and models."""
    print("=" * 60)
    print(" [*] TOUCHLESS PC CONTROL -- SYSTEM DIAGNOSTICS")
    print("=" * 60)

    # 1. Environment
    print(f"\n[+] OS Platform     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[+] Python Version  : {sys.version.split()[0]}")

    # 2. Screen Dimensions
    try:
        w_scr, h_scr = pyautogui.size()
        print(f"[+] Display Screen  : {w_scr}x{h_scr} px")
    except Exception as e:
        print(f"[-] Display Screen  : Detection failed ({e})")

    # 3. Model Asset Check
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[+] Hand Model Task : FOUND ({size_mb:.2f} MB)")
    else:
        print(f"[-] Hand Model Task : MISSING at {model_path}")

    # 4. Camera Probing
    print("\n[+] Probing Available Webcams...")
    found_cameras = 0
    for idx in range(3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            print(f"    - Camera Index {idx}: READY (Default: {w}x{h} @ {fps} FPS)")
            cap.release()
            found_cameras += 1
        else:
            print(f"    - Camera Index {idx}: Unavailable")

    # 5. Audio Interface
    audio = AudioController()
    if audio.available:
        print(f"\n[+] Audio Control   : READY (PyCaw Master Volume)")
    else:
        print(f"\n[-] Audio Control   : FALLBACK (Direct volume control unavailable)")

    # 6. Brightness Interface
    brightness = BrightnessController()
    if brightness.available:
        print(f"[+] Brightness Ctrl : READY ({brightness.get_brightness()}% current)")
    else:
        print(f"[-] Brightness Ctrl : FALLBACK (Hardware brightness unavailable)")

    print("\n" + "=" * 60)
    print(" Diagnostics completed successfully.")
    print("=" * 60)
    return 0
