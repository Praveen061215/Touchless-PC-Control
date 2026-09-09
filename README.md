# 🖐️ Touchless PC Control

A high-performance virtual mouse and computer control suite driven by real-time hand gestures. Control cursor navigation, left/right clicks, dragging, scrolling, audio volume, display brightness, and window shortcuts via your webcam using OpenCV and MediaPipe.

Built with **Python**, **OpenCV**, **MediaPipe Tasks API**, and **PyAutoGUI**.

---

## ✨ Key Features

- 🎯 **Sub-Pixel Cursor Tracking**: Adaptive **1€ Filter** (Casiez et al., CHI 2012) dynamically eliminates hand tremors when stationary while delivering zero-lag responsiveness during fast sweeps.
- 🖐️ **13+ Intuitive Gestures**: Move, left-click, drag-and-drop, right-click, scroll, screenshot, volume, brightness, window navigation, and emergency stop.
- 🌙 **Low-Light CLAHE Boost**: Built-in luminance-channel contrast equalization ensures rock-solid tracking even in poorly lit environments (toggle on-the-fly with `L`).
- 🔊 **Sensory Feedback**: Visual click ripple animations and asynchronous auditory feedback cues (toggle with `S`).
- ⚙️ **Configurable & Scriptable**: Centralized `config.json` and a full CLI with hardware diagnostics (`--diagnostics`).
- 🧪 **Comprehensive Test Suite**: Automated unit tests for gesture classification, filters, controllers, and configuration.

---

## 📋 Requirements

- Python 3.9+
- A working webcam
- Supported OS: Windows (full support including hardware volume & brightness), macOS / Linux (cursor and mouse navigation supported with graceful stubs)

---

## ⚙️ Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/Praveen061215/Touchless-PC-Control.git
cd Touchless-PC-Control
```

**2. Install dependencies**
```bash
py -m pip install -r requirements.txt
```

**3. Run Hardware Diagnostics**
```bash
py VirtualMouse.py --diagnostics
```

**4. Start the Virtual Mouse**
```bash
py VirtualMouse.py
```

---

## 🎮 Gesture Reference

| Gesture | Pose | Action |
|---|---|---|
| ☝️ **Move Cursor** | Index finger extended | Moves the system mouse pointer |
| 🤌 **Left Click** | Thumb + Index pinch (< 0.55s) | Left mouse click |
| ↕️ **Drag & Drop** | Thumb + Index pinch & move | Mouse drag; release pinch to drop |
| ✌️ **Right Click** | Index + Middle fingers held still (~6 frames) | Right mouse click |
| ↕️ **Scroll** | Index + Middle fingers moving vertically | Natural page scrolling |
| 🤚 **Pause / Resume** | Open palm (5 fingers held still) | Toggle pause state |
| 👈 **Previous** | Open palm + swipe left | Browser / App Back (`Alt + ←`) |
| 👉 **Next** | Open palm + swipe right | Browser / App Forward (`Alt + →`) |
| ⬇️ **Show Desktop** | Open palm + swipe down | Minimize all windows (`Win + D`) |
| ✊ **Emergency Stop** | Closed fist | Immediate control freeze |
| 👍 **Confirm** | Thumb extended only | Enter key press |
| 🤙 **Brightness** | Thumb + Pinky extended (Shaka) | Move hand up / down to adjust screen brightness |
| 🔊 **Volume Control** | Thumb + Index in L-shape | Move hand up / down to adjust system master volume |
| 🔍 **Zoom In / Out** | Thumb + Index spread / close | Dynamic zoom (`Ctrl +` / `Ctrl -`) |
| 📸 **Screenshot** | Index + Middle + Ring fingers extended | Screen snip shortcut (`Win + Shift + S`) |
| ↔️ **App Switcher** | Index finger rapid horizontal swipe | Window switcher (`Alt + Tab`) |

### Keyboard Shortcuts

- `Q`: Quit application cleanly
- `L`: Toggle **Low-Light CLAHE Enhancement** mode
- `S`: Toggle **Auditory Sound Feedback**

---

## 💻 CLI Options

```bash
usage: VirtualMouse.py [-h] [--config CONFIG] [--camera CAMERA]
                       [--width WIDTH] [--height HEIGHT]
                       [--filter {one-euro,ema,none}] [--no-hud] [--debug]
                       [--diagnostics]

options:
  -h, --help            Show help message and exit
  --config CONFIG, -c   Path to custom JSON configuration file (default: config.json)
  --camera CAMERA, -cam Webcam device index (e.g. 0, 1)
  --width WIDTH         Override camera frame capture width (e.g. 1280)
  --height HEIGHT       Override camera frame capture height (e.g. 720)
  --filter FILTER       Smoothing filter algorithm: 'one-euro', 'ema', or 'none'
  --no-hud              Headless mode: runs without displaying OpenCV preview window
  --debug               Enable verbose debug logging
  --diagnostics         Run hardware and environment probe and print diagnostic report
```

---

## 📁 Project Architecture

```
Touchless PC Control/
├── .github/workflows/ci.yml # Automated multi-platform test runner
├── config.py                # Dataclass settings loader & validator
├── config.json              # User preferences & tunable parameters
├── cli.py                   # Argument parsing & system diagnostics
├── feedback.py              # Visual click ripples & asynchronous audio cues
├── filters.py               # 1€ adaptive filter and 2D coordinate smoothing
├── gestures.py              # Gesture rules, classification & stability buffer
├── HandTrackingModule.py    # MediaPipe Tasks HandLandmarker wrapper
├── pyproject.toml           # PEP 621 packaging metadata
├── requirements.txt         # Production dependencies
├── run_tests.py             # Test discovery and execution runner
├── system_control.py        # Safe audio volume & display brightness controllers
├── tests/                   # Automated unit test suite (28 test cases)
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_feedback.py
│   ├── test_filters.py
│   ├── test_gestures.py
│   ├── test_system_control.py
│   └── test_vision.py
├── vision_utils.py          # Adaptive low-light CLAHE luminance enhancer
└── VirtualMouse.py          # Main application orchestrator & gesture HUD
```

---

## 🧪 Running Tests

Execute the full automated test suite with standard Python:
```bash
py run_tests.py
```
Or via `pytest`:
```bash
py -m pytest
```

---

## 🔧 Tuning Configuration

All operational parameters can be adjusted in [`config.json`](config.json):
- `gesture.filter_type`: Choose between `"one-euro"`, `"ema"`, or `"none"`.
- `gesture.one_euro_min_cutoff` & `one_euro_beta`: Tune sensitivity to hand jitter vs high-speed tracking.
- `gesture.gesture_stability_frames`: Majority-voting window size (default: 4 frames).
- `ui.low_light_mode`: Enable automatic low-light enhancement at startup.
- `ui.sound_feedback`: Enable audio clicks on gestures.

---

## 👤 Author

Created and maintained by **Praveen** ([@Praveen061215](https://github.com/Praveen061215)).

## 📄 License

This project is licensed under the [MIT License](LICENSE).