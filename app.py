import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
from collections import deque

# ---------------- CONFIG ----------------
TEXT_BAR_HEIGHT = 80

DWELL_TIME = 0.9
SMOOTHING = 7

KEYS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM ")
]

ROWS = len(KEYS)
COLS = max(len(r) for r in KEYS)

# ---------------- TTS ----------------
tts = pyttsx3.init()
tts.setProperty("rate", 180)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh

LEFT_EYE  = [33, 133]
RIGHT_EYE = [362, 263]

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ---------------- STATE ----------------
sx = deque(maxlen=SMOOTHING)
sy = deque(maxlen=SMOOTHING)

current_key = None
key_start_time = None
typed_text = ""

# ---------------- UTILS ----------------
def clamp(v):
    return max(0.0, min(1.0, v))

def draw_keyboard(frame, selected):
    h, w, _ = frame.shape
    key_w = w // COLS
    key_h = (h - 80) // ROWS

    for r in range(ROWS):
        for c in range(len(KEYS[r])):
            x1 = c * key_w
            y1 = 80 + r * key_h
            x2 = x1 + key_w
            y2 = y1 + key_h

            key = KEYS[r][c]
            color = (255, 255, 255)

            if key == selected:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "SPACE" if key == " " else key
            cv2.putText(frame, label, (x1 + 10, y1 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_text_bar(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (40, 40, 40), -1)
    cv2.putText(frame, text, (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

def gaze_to_key(gx, gy, frame_h):
    keyboard_top = TEXT_BAR_HEIGHT
    keyboard_height = frame_h - keyboard_top

    # Convert normalized gy → pixel
    y_px = gy * frame_h

    # Ignore gaze above keyboard
    if y_px < keyboard_top:
        return None

    # Normalize within keyboard only
    gy_kb = (y_px - keyboard_top) / keyboard_height

    r = int(gy_kb * ROWS)
    c = int(gx * COLS)

    if 0 <= r < ROWS and 0 <= c < len(KEYS[r]):
        return KEYS[r][c]

    return None


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as face_mesh:


    cv2.namedWindow("Iris Keyboard", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
    "Iris Keyboard",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        selected = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark

            def pt(i):
                return np.array([lm[i].x * w, lm[i].y * h])

            # Left eye
            le_l, le_r = pt(LEFT_EYE[0]), pt(LEFT_EYE[1])
            li = np.mean([pt(i) for i in LEFT_IRIS], axis=0)
            lw = np.linalg.norm(le_r - le_l)

            # Right eye
            re_l, re_r = pt(RIGHT_EYE[0]), pt(RIGHT_EYE[1])
            ri = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)
            rw = np.linalg.norm(re_r - re_l)

        # Horizontal gaze (eye-relative)
            gx = ((li[0] - le_l[0]) / lw + (ri[0] - re_l[0]) / rw) / 2

# Vertical gaze (SCREEN-relative)
            gy = (li[1] + ri[1]) / (2 * h)

            gx, gy = clamp(1.0-gx), clamp(gy)

            sx.append(gx)
            sy.append(gy)

            gx, gy = np.mean(sx), np.mean(sy)

            key = gaze_to_key(gx,gy,h)

            if key == current_key:
                if key_start_time and time.time() - key_start_time >= DWELL_TIME:
                    typed_text += key
                    speak("space" if key == " " else key)
                    key_start_time = time.time() + 1.0
            else:
                current_key = key 
                key_start_time = time.time() if key else None   

            selected = key

        draw_text_bar(frame, typed_text)
        draw_keyboard(frame, selected)

        cv2.imshow("Iris Keyboard", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
