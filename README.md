# 🖐️ Touchless PC Control

Control your PC entirely with hand gestures using your webcam — no physical contact required.

Built with **Python**, **OpenCV**, **MediaPipe Tasks API (v1.0+)**, and **PyAutoGUI**.

---

## 📋 Requirements

- Python 3.9+
- A working webcam
- Windows / macOS / Linux

## ⚙️ Setup

**1. Install dependencies**
```bash
py -m pip install -r requirements.txt
```

**2. Download the hand landmark model** *(already included — `hand_landmarker.task`)*

If you ever need to re-download it:
```bash
py -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task'); print('Done')"
```

**3. Run the virtual mouse**
```bash
py VirtualMouse.py
```

---

## 🎮 Gesture Reference

| Finger / Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Move the mouse cursor |
| 🤌 **Thumb pinch — still** | Left click |
| ↕️ **Thumb pinch — move** | Drag |
| ✌️ **Two fingers — hold still** | Right click |
| ↕️ **Two fingers — move up/down** | Scroll |
| 🤚 **Open palm — still** | Pause / Activate toggle |
| 👈 **Open palm — swipe left** | Previous (Alt + ←) |
| 👉 **Open palm — swipe right** | Next (Alt + →) |
| ✊ **Fist** | Emergency stop |
| 👍 **Thumb only** | Confirm (Enter) |
| 🤙 **Thumb + Pinky (shaka)** | Open App (Win + R) |
| ↔️ **Index — fast swipe** | Change Window (Alt + Tab) |
| 🔊 **Thumb + Index L-shape** | Volume (move hand up/down) |
| 🔍 **Thumb + Index pinch/spread** | Zoom In / Out (Ctrl +/-) |
| **Q key** | Quit the program |

> **Tips**
> - Keep your hand inside the **purple corner brackets** (control zone) for accurate cursor mapping.
> - For **right-click**: hold two fingers still for ~6 frames before it fires (prevents accidental triggers while scrolling).
> - For **drag**: pinch thumb+index and then move your hand. Release the pinch to drop.
> - For **volume**: form an L-shape (thumb + index extended), then move hand **up** = louder, **down** = quieter.
> - For **zoom**: pinch thumb+index close, then **spread** = zoom in, **close** = zoom out.

---

## 📁 Project Structure

```
Touchless PC Control/
├── HandTrackingModule.py   # MediaPipe Tasks wrapper (hand detection + gestures)
├── VirtualMouse.py         # Main application — 13-gesture control suite
├── hand_landmarker.task    # MediaPipe hand landmark model (binary)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔧 Tuning

Open `VirtualMouse.py` and adjust these constants at the top:

| Variable | Default | Effect |
|---|---|---|
| `smoothening` | `7` | Higher = smoother but laggier mouse |
| `PINCH_DIST` | `42` | px distance between thumb+index to detect pinch |
| `DRAG_MIN` | `16` | px movement after pinch before drag activates |
| `CLICK_COOL` | `0.40` | Seconds between consecutive left clicks |
| `RCLICK_HOLD` | `6` | Frames two-fingers must be held before right click |
| `SWIPE_VEL` | `55` | Net px movement over 6 frames to register a swipe |
| `SCROLL_VEL` | `22` | Net px movement over 6 frames to trigger scroll |
| `frameR` | `80` | Size of the control zone margin |
