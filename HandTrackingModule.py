"""
HandTrackingModule.py
---------------------
Hand detection and landmark tracking using MediaPipe Tasks API (v1.0+).
"""

import cv2
import math
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python  # type: ignore[attr-defined]
from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import]

# ── Drawing helpers from the new Tasks API ────────────────────────────────────
_drawing_utils  = mp_vision.drawing_utils
_drawing_styles = mp_vision.drawing_styles
_HandConns      = mp_vision.HandLandmarksConnections

# Fingertip landmark IDs (Thumb, Index, Middle, Ring, Pinky)
TIP_IDS = [4, 8, 12, 16, 20]

# Default model path (same directory as this file)
_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hand_landmarker.task")


class HandDetector:
    """
    Wraps MediaPipe HandLandmarker for easy hand detection and gesture logic.
    Supports the new mp.tasks API (mediapipe >= 1.0).
    """

    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7,
                 model_path=_DEFAULT_MODEL):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=maxHands,
            min_hand_detection_confidence=float(detectionCon),
            min_hand_presence_confidence=float(detectionCon),
            min_tracking_confidence=float(trackCon),
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_result = None

    # ── Public Methods ─────────────────────────────────────────────────────────

    def findHands(self, img, draw=True):
        """
        Detect hands in a BGR image.

        Returns
        -------
        allHands : list[dict]
            Each dict has keys: 'lmList' (list of [id, x, y]),
            'bbox' (x, y, w, h), 'center' (cx, cy), 'type' ('Left'|'Right').
        img : np.ndarray
            The (optionally annotated) BGR image.
        """
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)
        self._last_result = result

        allHands = []

        if result.hand_landmarks:
            for idx, (landmarks, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)):

                lmList = []
                xList, yList = [], []

                for lm_id, lm in enumerate(landmarks):
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmList.append([lm_id, px, py])
                    xList.append(px)
                    yList.append(py)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bw, bh = xmax - xmin, ymax - ymin

                # Handedness label is from the model's POV (unmirrored camera).
                # Since we flip the frame before calling findHands, we flip label.
                label = handedness[0].category_name   # "Left" or "Right"
                flipped_label = "Right" if label == "Left" else "Left"

                myHand = {
                    "lmList": lmList,
                    "bbox":   (xmin, ymin, bw, bh),
                    "center": (xmin + bw // 2, ymin + bh // 2),
                    "type":   flipped_label,
                }
                allHands.append(myHand)

                if draw:
                    self._draw_hand(img, landmarks, xmin, ymin, bw, bh,
                                    flipped_label)

        return allHands, img

    def fingersUp(self, myHand):
        """
        Returns [thumb, index, middle, ring, pinky] as 1 (up) or 0 (down).
        Correctly handles both Left and Right hands.
        """
        lm = myHand["lmList"]
        hand_type = myHand["type"]
        fingers = []

        # Thumb — compare x (lateral direction differs per hand)
        if hand_type == "Right":
            fingers.append(1 if lm[TIP_IDS[0]][1] < lm[TIP_IDS[0] - 1][1] else 0)
        else:
            fingers.append(1 if lm[TIP_IDS[0]][1] > lm[TIP_IDS[0] - 1][1] else 0)

        # Other 4 fingers — compare y (tip above knuckle = up)
        for i in range(1, 5):
            fingers.append(1 if lm[TIP_IDS[i]][2] < lm[TIP_IDS[i] - 2][2] else 0)

        return fingers

    def findDistance(self, p1, p2, img, lmList, draw=True, r=12, t=2):
        """
        Euclidean distance between two landmarks.
        Returns (distance, img, [x1,y1,x2,y2,cx,cy]).
        """
        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        dist = math.hypot(x2 - x1, y2 - y1)
        return dist, img, [x1, y1, x2, y2, cx, cy]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _draw_hand(self, img, landmarks, xmin, ymin, bw, bh, label):
        """Draw landmark skeleton, bounding box, and label on img."""
        # Build NormalizedLandmarkList-like iterable for drawing_utils
        h, w = img.shape[:2]

        lm_style   = _drawing_styles.get_default_hand_landmarks_style()
        conn_style = _drawing_styles.get_default_hand_connections_style()

        # Draw each connection manually using pixel coords (drawing_utils
        # expects NormalizedLandmark objects so we do it ourselves)
        for conn in _HandConns.HAND_CONNECTIONS:
            s, e = conn.start, conn.end
            x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
            x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

        # Draw landmark dots
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Bounding box + label
        cv2.rectangle(img,
                      (xmin - 20, ymin - 20),
                      (xmin + bw + 20, ymin + bh + 20),
                      (0, 230, 80), 2)
        cv2.putText(img, label, (xmin - 20, ymin - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 80), 2)

    def __del__(self):
        if hasattr(self, "landmarker"):
            self.landmarker.close()
