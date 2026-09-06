"""
VirtualMouse.py
---------------
Touchless PC Control — Advanced Gesture Suite

  Gesture                   │  Action
  ──────────────────────────┼────────────────────────────────
  Index only                │  Move cursor
  Thumb+Index pinch (still) │  Left click
  Thumb+Index pinch (move)  │  Drag
  Two fingers (still, hold) │  Right click
  Two fingers + vertical    │  Scroll up / down
  Open palm (still)         │  Pause / Resume toggle
  Open palm + swipe left    │  Previous  (Alt + ←)
  Open palm + swipe right   │  Next      (Alt + →)
  Open palm + swipe down    │  Show Desktop (Win + D)
  Fist                      │  Emergency stop
  Thumb only                │  Confirm   (Enter)
  Thumb + Pinky (Shaka)     │  Brightness (move hand up/down)
  Index + fast swipe        │  Change Window (Alt + Tab)
  Thumb+Index L-shape       │  Volume (move hand up/down)
  Thumb+Index pinch/spread  │  Zoom In / Out (Ctrl +/-)
  Index + Middle + Ring     │  Screenshot (Win + Shift + S)

  Press Q to quit.
"""

import cv2
import logging
import math
import time
import numpy as np
import pyautogui
from collections import deque
import HandTrackingModule as htm

from config import AppConfig
from system_control import AudioController, BrightnessController
from filters import PointFilter2D
from gestures import classify_gesture, GestureStabilizer, calculate_adaptive_pinch_dist
from vision_utils import LowLightEnhancer
from feedback import FeedbackManager

# ═══════════════════════════════════════════════════════════════ CONFIGURATION ══
app_config   = AppConfig.load("config.json")

wCam, hCam   = app_config.camera.width, app_config.camera.height
frameR       = app_config.gesture.control_margin
smoothening  = app_config.gesture.smoothening

PINCH_DIST   = app_config.gesture.pinch_distance
DRAG_MIN     = app_config.gesture.drag_min_distance

CLICK_COOL   = app_config.gesture.click_cooldown
RCLICK_COOL  = app_config.gesture.rclick_cooldown
SCROLL_COOL  = app_config.gesture.scroll_cooldown
ACT_COOL     = app_config.gesture.action_cooldown
VOL_COOL     = app_config.gesture.volume_cooldown
ZOOM_COOL    = app_config.gesture.zoom_cooldown

SWIPE_VEL    = app_config.gesture.swipe_velocity_threshold
SCROLL_VEL   = app_config.gesture.scroll_velocity_threshold
RCLICK_HOLD  = app_config.gesture.rclick_hold_frames

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0


# ═══════════════════════════════════════════════════════════ CAMERA & DETECTOR ══
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
cap.set(cv2.CAP_PROP_FPS, 30)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera connection.")

detector   = htm.HandDetector(maxHands=1, detectionCon=0.65, trackCon=0.65)
wScr, hScr = pyautogui.size()

print(f"[INFO] Screen {wScr}×{hScr}  |  Camera {wCam}×{hCam}")
print("[INFO] Gesture control active. Press Q in the window to quit.")

# ═════════════════════════════════════════════════════════ AUDIO & BRIGHTNESS ══
audio_ctrl      = AudioController()
brightness_ctrl = BrightnessController()
enhancer        = LowLightEnhancer(
    enabled=app_config.ui.low_light_mode,
    alpha=app_config.ui.low_light_alpha,
    beta=app_config.ui.low_light_beta,
    clahe_clip=app_config.ui.low_light_clahe_clip
)
feedback        = FeedbackManager(sound_enabled=app_config.ui.sound_feedback)

# ══════════════════════════════════════════════════════════════════════ STATE ══
plocX, plocY    = wScr / 2, hScr / 2
pTime           = time.time()

t_lclick = t_rclick = t_scroll = t_act = t_vol = t_zoom = t_screenshot = 0.0

paused          = False
dragging        = False
pinch_start_t   = None
pinch_start_pos = None
prev_pinch_d    = None      # previous pinch distance (for zoom delta)
rclick_frames   = 0         # two-finger hold counter before right-click fires
last_gesture    = None
gesture_label   = "Idle"

CX = deque(maxlen=10)       # palm-center X ring buffer (velocity calculation)
CY = deque(maxlen=10)       # palm-center Y ring buffer

# ── Gesture stability stabilizer ──────────────────────────────────────────────
gesture_stabilizer = GestureStabilizer(window_size=app_config.gesture.gesture_stability_frames)

consecutive_fails = 0

# HUD animation state
hud_vol_bar = 0.0
hud_brightness_bar = 0.0
show_vol_frames = 0
show_brightness_frames = 0


# ══════════════════════════════════════════════════════════════════ HELPERS ══
def ldist(lm, a, b):
    """Euclidean pixel distance between two hand landmarks."""
    return math.hypot(lm[a][1] - lm[b][1], lm[a][2] - lm[b][2])


def vel(buf):
    """Net displacement (newest − oldest) from a ring buffer."""
    return (buf[-1] - buf[0]) if len(buf) >= 2 else 0.0


cursor_filter = PointFilter2D(
    filter_type=app_config.gesture.filter_type,
    smoothening=app_config.gesture.smoothening,
    min_cutoff=app_config.gesture.one_euro_min_cutoff,
    beta=app_config.gesture.one_euro_beta,
    d_cutoff=app_config.gesture.one_euro_d_cutoff,
    freq=float(app_config.camera.fps)
)

def can(t_last, cd):
    return time.time() - t_last > cd


def smooth_pos(x1, y1):
    """Map camera control-zone coordinate → smoothed screen coordinate."""
    global plocX, plocY
    sx = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
    sy = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
    cx, cy = cursor_filter.filter(sx, sy, timestamp=time.time())
    plocX, plocY = cx, cy
    return max(0.0, min(wScr - 1, cx)), max(0.0, min(hScr - 1, cy))





# ─── Gesture → HUD colour map ─────────────────────────────────────────────────
_GC = {
    "Idle":           (140, 140, 140),
    "Move":           (255, 200,   0),
    "Pinching...":    (  0, 200, 255),
    "Left Click":     (  0, 230,  80),
    "Drag":           (255, 130,   0),
    "Two Fingers":    (190,   0, 255),
    "Right Click":    (220,   0, 255),
    "Scroll Up":      ( 30, 180, 255),
    "Scroll Down":    ( 30, 180, 255),
    "STOP":           ( 20,  20, 255),
    "Pause":          (100, 100, 100),
    "Paused":         (100, 100, 100),
    "Resumed":        (  0, 230,  80),
    "< Previous":     (  0, 210, 190),
    "Next >":         (  0, 210, 190),
    "Show Desktop":   (  0, 210, 190),
    "Confirm":        (  0, 230,  80),
    "Open App":       (  0, 200, 255),
    "Screenshot":     (255, 255, 255),
    "Brightness":     (255, 200,   0),
    "Change Window":  (255, 210,   0),
    "Vol Up":         (  0, 180, 255),
    "Vol Down":       (  0, 180, 255),
    "Volume":         (  0, 180, 255),
    "Zoom In":        (180, 255,   0),
    "Zoom Out":       (180, 255,   0),
    "Zoom":           (180, 255,   0),
}


def draw_hud(img, fps, label, fingers, is_paused, rclick_progress=0.0, vol_bar=0.0, bright_bar=0.0, show_v=0, show_b=0, lm_idx8=None, is_low_light=False):
    h, w = img.shape[:2]

    # Bottom translucent status bar
    bar = img.copy()
    cv2.rectangle(bar, (0, h - 72), (w, h), (8, 10, 18), -1)
    cv2.addWeighted(bar, 0.72, img, 0.28, 0, img)

    # Pause overlay — dim tint + message
    if is_paused:
        tint = img.copy()
        cv2.rectangle(tint, (0, 0), (w, h), (0, 0, 55), -1)
        cv2.addWeighted(tint, 0.20, img, 0.80, 0, img)
        msg  = "PAUSED  --  Open palm to resume"
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)[0]
        cv2.putText(img, msg, ((w - size[0]) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, (55, 55, 255), 2, cv2.LINE_AA)

    # FPS counter (bottom-left)
    cv2.putText(img, f"FPS {int(fps):>2}", (12, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (70, 255, 70), 2, cv2.LINE_AA)

    # Gesture label (centered)
    col  = _GC.get(label, (255, 255, 255))
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)[0]
    cv2.putText(img, label, ((w - size[0]) // 2, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2, cv2.LINE_AA)

    # Finger state dots  T I M R P  (bottom-right)
    for i, (s, lbl) in enumerate(zip(fingers, ["T", "I", "M", "R", "P"])):
        dot_col = (0, 220, 0) if s else (50, 50, 50)
        dx = w - 138 + i * 27
        cv2.circle(img, (dx, h - 46), 10, dot_col, -1)
        cv2.putText(img, lbl, (dx - 5, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (185, 185, 185), 1)

    # Title (top-left)
    cv2.putText(img, "Touchless PC Control", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    if is_low_light:
        cv2.putText(img, "[LOW-LIGHT: ON]", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    # Control-zone corner accents
    cl = 22
    for (fx, fy) in [(frameR, frameR), (wCam - frameR, frameR),
                     (frameR, hCam - frameR), (wCam - frameR, hCam - frameR)]:
        dx = cl if fx == frameR else -cl
        dy = cl if fy == frameR else -cl
        cv2.line(img, (fx, fy), (fx + dx, fy), (130, 0, 255), 2)
        cv2.line(img, (fx, fy), (fx, fy + dy), (130, 0, 255), 2)

    # Circular progress for right-click hold
    if rclick_progress > 0 and lm_idx8:
        cx, cy = lm_idx8
        angle = int(360 * rclick_progress)
        cv2.ellipse(img, (cx, cy), (24, 24), -90, 0, angle, (220, 0, 255), 3)

    # Sleek Volume Bar
    if show_v > 0:
        alpha = min(1.0, show_v / 15.0)
        v_h = int(vol_bar * 200) # 0 to 200px
        overlay = img.copy()
        cv2.rectangle(overlay, (20, h // 2 - 100), (40, h // 2 + 100), (40, 40, 40), -1)
        cv2.rectangle(overlay, (20, h // 2 + 100 - v_h), (40, h // 2 + 100), (0, 180, 255), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, f"{int(vol_bar*100)}%", (15, h // 2 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    # Sleek Brightness Bar
    if show_b > 0:
        alpha = min(1.0, show_b / 15.0)
        b_h = int(bright_bar * 200) # 0 to 200px
        overlay = img.copy()
        cv2.rectangle(overlay, (w - 40, h // 2 - 100), (w - 20, h // 2 + 100), (40, 40, 40), -1)
        cv2.rectangle(overlay, (w - 40, h // 2 + 100 - b_h), (w - 20, h // 2 + 100), (255, 200, 0), -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.putText(img, f"{int(bright_bar*100)}%", (w - 50, h // 2 + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    return img


# ══════════════════════════════════════════════════════════════════ MAIN LOOP ══
print("[INFO] Hold index finger up to move cursor. Open palm to pause.")

while True:
    ok, img = cap.read()
    if not ok:
        consecutive_fails += 1
        if consecutive_fails > 15:
            print("[ERROR] Webcam feed lost. Exiting.")
            break
        continue
    consecutive_fails = 0

    img        = cv2.flip(img, 1)
    img        = enhancer.enhance(img)
    hands, img = detector.findHands(img, draw=True)

    fingers       = [0, 0, 0, 0, 0]
    gesture       = "IDLE"
    gesture_label = "Paused" if paused else "Idle"
    now           = time.time()
    pd            = 0.0

    if hands:
        hand   = hands[0]
        lm     = hand["lmList"]
        cx, cy = hand["center"]

        CX.append(cx)
        CY.append(cy)
        vx = vel(CX)        # palm horizontal velocity
        vy = vel(CY)        # palm vertical velocity

        fingers     = detector.fingersUp(hand)
        raw_gesture, _ = classify_gesture(
            fingers, lm, vx, vy,
            pinch_dist=PINCH_DIST,
            swipe_vel=SWIPE_VEL,
            scroll_vel=SCROLL_VEL
        )
        gesture     = gesture_stabilizer.update(raw_gesture)
        pd          = calculate_adaptive_pinch_dist(hand["bbox"], base_dist=PINCH_DIST)

        # ── Release drag when gesture moves away from PINCH ─────────────────────
        was_dragging = dragging
        if dragging and gesture != "PINCH":
            pyautogui.mouseUp()
            dragging = False

        # ── Trigger click when PINCH ends without having dragged ────────────────
        if last_gesture == "PINCH" and gesture != "PINCH":
            if not was_dragging and pinch_start_t is not None:
                if (now - pinch_start_t) < 0.55 and can(t_lclick, CLICK_COOL):
                    pyautogui.click()
                    feedback.trigger_click(int(lm[8][1]), int(lm[8][2]), is_right=False)
                    t_lclick = now
                    print("[INFO] Left Click")
            pinch_start_t   = None
            pinch_start_pos = None

        # ═══════════════════════════ GESTURE DISPATCH ═══════════════════════════
        if not paused:

            # ── FIST — emergency stop ────────────────────────────────────────────
            if gesture == "FIST":
                rclick_frames = 0
                gesture_label = "STOP"

            # ── OPEN PALM — toggle pause ─────────────────────────────────────────
            elif gesture == "OPEN_PALM":
                if last_gesture != "OPEN_PALM" and can(t_act, ACT_COOL):
                    paused = True
                    t_act  = now
                    print("[INFO] PAUSED")
                gesture_label = "Pause"

            # ── PALM NAV — Alt+Left / Alt+Right (prev/next) ──────────────────────
            elif gesture == "PALM_L":
                if last_gesture != "PALM_L" and can(t_act, ACT_COOL):
                    pyautogui.hotkey('alt', 'left')
                    t_act = now
                    print("[INFO] < Previous (Alt+Left)")
                gesture_label = "< Previous"

            elif gesture == "PALM_R":
                if last_gesture != "PALM_R" and can(t_act, ACT_COOL):
                    pyautogui.hotkey('alt', 'right')
                    t_act = now
                    print("[INFO] Next > (Alt+Right)")
                gesture_label = "Next >"

            elif gesture == "PALM_D":
                if last_gesture != "PALM_D" and can(t_act, ACT_COOL):
                    pyautogui.hotkey('win', 'd')
                    t_act = now
                    print("[INFO] Show Desktop (Win+D)")
                gesture_label = "Show Desktop"

            # ── THUMB UP — confirm (Enter) ───────────────────────────────────────
            elif gesture == "THUMB_UP":
                if last_gesture != "THUMB_UP" and can(t_act, ACT_COOL):
                    pyautogui.press('enter')
                    t_act = now
                    print("[INFO] Confirm (Enter)")
                gesture_label = "Confirm"

            # ── THUMB + PINKY — Brightness Control ────────────────────────────────
            elif gesture == "BRIGHTNESS":
                rclick_frames = 0
                show_brightness_frames = 30
                if brightness_ctrl.available:
                    b_val = np.interp(lm[0][2], [100, 380], [100, 0])
                    hud_brightness_bar = hud_brightness_bar * 0.8 + (b_val / 100.0) * 0.2
                    brightness_ctrl.set_brightness(int(hud_brightness_bar * 100))
                    gesture_label = "Brightness"
                else:
                    gesture_label = "Brightness (N/A)"

            # ── SCREENSHOT — 3 Fingers (Index + Middle + Ring) ────────────────────
            elif gesture == "SCREENSHOT":
                if last_gesture != "SCREENSHOT" and can(t_screenshot, ACT_COOL):
                    pyautogui.hotkey('win', 'shift', 's')
                    t_screenshot = now
                    print("[INFO] Screenshot (Win+Shift+S)")
                gesture_label = "Screenshot"

            # ── MOVE — cursor movement ───────────────────────────────────────────
            elif gesture == "MOVE":
                rclick_frames = 0
                sx, sy = smooth_pos(lm[8][1], lm[8][2])
                pyautogui.moveTo(sx, sy)
                plocX, plocY = sx, sy
                cv2.circle(img, (lm[8][1], lm[8][2]), 14, (255, 200, 0), cv2.FILLED)
                gesture_label = "Move"

            # ── SWIPE — change window (Alt+Tab) ──────────────────────────────────
            elif gesture in ("SWIPE_L", "SWIPE_R"):
                if last_gesture not in ("SWIPE_L", "SWIPE_R") and can(t_act, ACT_COOL):
                    pyautogui.hotkey('alt', 'tab')
                    t_act = now
                    print("[INFO] Change Window (Alt+Tab)")
                gesture_label = "Change Window"

            # ── PINCH — left click or drag ───────────────────────────────────────
            elif gesture == "PINCH":
                rclick_frames = 0
                sx, sy = smooth_pos(lm[8][1], lm[8][2])

                if pinch_start_t is None:
                    pinch_start_t   = now
                    pinch_start_pos = (sx, sy)

                moved = math.hypot(sx - pinch_start_pos[0], sy - pinch_start_pos[1])

                if not dragging and moved > DRAG_MIN:
                    pyautogui.mouseDown()
                    dragging = True

                if dragging:
                    pyautogui.moveTo(sx, sy)
                    gesture_label = "Drag"
                else:
                    # Visual: midpoint circle between thumb and index
                    mx = (lm[4][1] + lm[8][1]) // 2
                    my = (lm[4][2] + lm[8][2]) // 2
                    cv2.circle(img, (lm[8][1], lm[8][2]), 14, (0, 200, 255), cv2.FILLED)
                    cv2.circle(img, (mx, my), 8, (0, 255, 200), cv2.FILLED)
                    gesture_label = "Pinching..."

                plocX, plocY = sx, sy

            # ── RCLICK — hold two fingers to right-click ──────────────────────────
            elif gesture == "RCLICK":
                rclick_frames += 1
                if rclick_frames >= RCLICK_HOLD and can(t_rclick, RCLICK_COOL):
                    pyautogui.rightClick()
                    feedback.trigger_click(int(lm[8][1]), int(lm[8][2]), is_right=True)
                    t_rclick      = now
                    rclick_frames = 0
                    print("[INFO] Right Click")
                gesture_label = ("Right Click"
                                 if rclick_frames >= RCLICK_HOLD // 2
                                 else "Two Fingers")

            # ── SCROLL UP / DOWN ─────────────────────────────────────────────────
            elif gesture == "SCROLL_U":
                rclick_frames = 0
                if can(t_scroll, SCROLL_COOL):
                    amt = max(1, min(5, int(abs(vy) / 10)))
                    pyautogui.scroll(amt)
                    t_scroll = now
                gesture_label = "Scroll Up"

            elif gesture == "SCROLL_D":
                rclick_frames = 0
                if can(t_scroll, SCROLL_COOL):
                    amt = max(1, min(5, int(abs(vy) / 10)))
                    pyautogui.scroll(-amt)
                    t_scroll = now
                gesture_label = "Scroll Down"

            # ── VOLUME — thumb+index L-shape, move hand up/down ──────────────────
            elif gesture == "VOLUME":
                rclick_frames = 0
                show_vol_frames = 30
                if audio_ctrl.available:
                    v_val = np.interp(lm[8][2], [100, 380], [100, 0])
                    hud_vol_bar = hud_vol_bar * 0.8 + (v_val / 100.0) * 0.2
                    audio_ctrl.set_volume_scalar(hud_vol_bar)
                    gesture_label = "Volume"
                else:
                    gesture_label = "Volume (N/A)"

            # ── ZOOM — thumb+index pinch/spread → Ctrl+= / Ctrl+- ────────────────
            elif gesture == "ZOOM":
                rclick_frames = 0
                if prev_pinch_d is not None and can(t_zoom, ZOOM_COOL):
                    delta = pd - prev_pinch_d
                    if delta > 5:               # fingers spreading → zoom in
                        pyautogui.hotkey('ctrl', '=')
                        t_zoom = now
                        gesture_label = "Zoom In"
                    elif delta < -5:            # fingers closing → zoom out
                        pyautogui.hotkey('ctrl', '-')
                        t_zoom = now
                        gesture_label = "Zoom Out"
                    else:
                        gesture_label = "Zoom"
                else:
                    gesture_label = "Zoom"
                prev_pinch_d = pd

            else:
                rclick_frames = 0

            # Reset zoom accumulator when not in ZOOM mode
            if gesture != "ZOOM":
                prev_pinch_d = None

        else:
            # ─── PAUSED — only OPEN_PALM toggles resume ───────────────────────────
            if (gesture == "OPEN_PALM"
                    and last_gesture != "OPEN_PALM"
                    and can(t_act, ACT_COOL)):
                paused = False
                t_act  = now
                print("[INFO] RESUMED")
                gesture_label = "Resumed"
            else:
                gesture_label = "Paused"

        last_gesture = gesture

    else:
        # ── No hand in frame ──────────────────────────────────────────────────────
        if dragging:
            pyautogui.mouseUp()
            dragging      = False
            pinch_start_t = None          # suppress accidental click on release
        elif pinch_start_t is not None and (now - pinch_start_t) < 0.55:
            if can(t_lclick, CLICK_COOL):
                pyautogui.click()
                t_lclick = now
                print("[INFO] Left Click (pinch release)")

        pinch_start_t   = None
        pinch_start_pos = None
        rclick_frames   = 0
        last_gesture    = None
        CX.clear()
        CY.clear()
        gesture_stabilizer.reset()  # reset stability buffer — no stale gestures
        cursor_filter.reset()       # reset smoothing filter on hand exit

    # ── FPS & Display ──────────────────────────────────────────────────────────
    cTime = time.time()
    fps   = 1.0 / max(cTime - pTime, 1e-6)
    pTime = cTime

    # decrement HUD counters
    show_vol_frames = max(0, show_vol_frames - 1)
    show_brightness_frames = max(0, show_brightness_frames - 1)

    # Render click feedback ripples
    img = feedback.update_and_draw(img)

    img = draw_hud(img, fps, gesture_label, fingers, paused,
                   rclick_progress=(rclick_frames / RCLICK_HOLD) if rclick_frames > 0 else 0.0,
                   vol_bar=hud_vol_bar, bright_bar=hud_brightness_bar,
                   show_v=show_vol_frames, show_b=show_brightness_frames,
                   lm_idx8=(lm[8][1], lm[8][2]) if hands else None,
                   is_low_light=enhancer.enabled)
    cv2.imshow("Touchless PC Control", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        new_state = enhancer.toggle()
        print(f"[INFO] Low-light mode: {'ON' if new_state else 'OFF'}")
    elif key == ord('s'):
        feedback.sound_enabled = not feedback.sound_enabled
        print(f"[INFO] Sound feedback: {'ON' if feedback.sound_enabled else 'OFF'}")

# ── Cleanup ────────────────────────────────────────────────────────────────────
if dragging:
    pyautogui.mouseUp()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Virtual Mouse stopped.")

# End of script
