import time
from collections import deque

import cv2
import numpy as np
import pyttsx3

EAR_CLOSED_MAX = 1.60     
MIN_BLINK_DURATION = 0.05 
CALIBRATION_SECONDS = 1.5
SMOOTHING_WINDOW = 5      
KEYBOARD_ROWS = 3
KEYBOARD_COLS = 9
KEY_H = 160               
KEY_TEXT_SCALE = 1.2



keys = [
    list("abcdefghi"),
    list("jklmnopqr"),
    list("stuvwxyz ")
]
keys[2][-1] = "<SPACE>"

engine = pyttsx3.init()

def audio_say(text):
    try:
        engine.say(text)
        engine.run()
    except Exception:
        pass

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def eye_aspect_ratio_6pts(pts):
    if pts is None or len(pts) < 6:
        return None
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    if C == 0:
        return None
    return (A + B) / (2.0 * C)

def safe_ratio(r, default=0.5):
    try:
        r = float(r)
        if np.isnan(r) or r is None:
            return default
        return float(np.clip(r, 0.0, 1.0))
    except Exception:
        return default

def get_grid_from_ratios(h_ratio, v_ratio):
    h_ratio = safe_ratio(h_ratio, 0.5)
    v_ratio = safe_ratio(v_ratio, 0.5)
    col = int(h_ratio * KEYBOARD_COLS)
    col = max(0, min(KEYBOARD_COLS - 1, col))
    if v_ratio < 0.33:
        row = 0
    elif v_ratio < 0.66:
        row = 1
    else:
        row = 2
    return row, col

def draw_keyboard(frame, selected_row, selected_col):
    h, w = frame.shape[:2]
    key_w = w // KEYBOARD_COLS
    start_y = h - KEY_H * KEYBOARD_ROWS
    for r in range(KEYBOARD_ROWS):
        for c in range(KEYBOARD_COLS):
            x = c * key_w
            y = start_y + r * KEY_H
            key = keys[r][c] if c < len(keys[r]) else ""
            if r == selected_row and c == selected_col:
                cv2.rectangle(frame, (x+2, y+2),
                              (x + key_w - 2, y + KEY_H - 2),
                              (0, 180, 0), -1)
                cv2.putText(frame, key,
                            (x + 20, y + KEY_H//2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            KEY_TEXT_SCALE, (0, 0, 0), 3)
            else:
                cv2.rectangle(frame, (x+2, y+2),
                              (x + key_w - 2, y + KEY_H - 2),
                              (120, 120, 120), 3)
                cv2.putText(frame, key,
                            (x + 20, y + KEY_H//2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            KEY_TEXT_SCALE, (255, 255, 255), 2)
    return frame

def build_eye_from_landmarks(lms, side="left"):
    lms = np.array(lms)  # (106, 2)
    xs = lms[:, 0]
    ys = lms[:, 1]
    x_mid = (xs.min() + xs.max()) / 2.0
    y_mid = (ys.min() + ys.max()) / 2.0

    if side == "left":
        mask = xs < x_mid
    else:
        mask = xs > x_mid

    eye_region = lms[mask]
    if len(eye_region) < 6:
        return None

    left_pt = eye_region[np.argmin(eye_region[:, 0])]
    right_pt = eye_region[np.argmax(eye_region[:, 0])]
    top_pt = eye_region[np.argmin(eye_region[:, 1])]
    bottom_pt = eye_region[np.argmax(eye_region[:, 1])]

    def nearest_two(base, pts):
        d = np.linalg.norm(pts - base, axis=1)
        idx = np.argsort(d)[:2]
        return pts[idx]

    upper_candidates = eye_region[eye_region[:, 1] <= y_mid]
    lower_candidates = eye_region[eye_region[:, 1] >= y_mid]

    if len(upper_candidates) < 2 or len(lower_candidates) < 2:
        pts6 = np.array([left_pt, top_pt, top_pt, right_pt, bottom_pt, bottom_pt])
        return pts6

    upper_two = nearest_two(top_pt, upper_candidates)
    lower_two = nearest_two(bottom_pt, lower_candidates)

    pts6 = np.array([
        left_pt,
        upper_two[0],
        upper_two[1],
        right_pt,
        lower_two[1],
        lower_two[0],
    ])
    return pts6

def main():
    print("Script started, importing InsightFace...")
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        print("InsightFace import failed:", e)
        print("Install packages: pip install insightface onnxruntime")
        input("Press Enter to exit...")
        return

    app = FaceAnalysis(name="buffalo_l")
    print("Loading UniFace models (this may take a few seconds)...")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        input("Press Enter to exit...")
        return
    print("Camera opened.")

    cv2.namedWindow("UniFace Eye Keyboard (ESC to quit)", cv2.WINDOW_NORMAL)

    gaze_queue = deque(maxlen=SMOOTHING_WINDOW)
    sentence = ""

    eyes_closed_start = None

    print(f"Calibration: look at the screen center for {CALIBRATION_SECONDS:.1f} seconds...")
    calib_h, calib_v = [], []
    t0 = time.time()
    while time.time() - t0 < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera during calibration")
            break
        faces = app.get(frame)
        if faces:
            face = faces[0]
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                lms = face.landmark_2d_106
                xs = lms[:, 0]
                ys = lms[:, 1]
                h_ratio = float((xs.mean() - xs.min()) / max(xs.max() - xs.min(), 1))
                v_ratio = float((ys.mean() - ys.min()) / max(ys.max() - ys.min(), 1))
                calib_h.append(h_ratio)
                calib_v.append(v_ratio)

        disp = frame.copy()
        cv2.putText(disp, "Calibration: look center", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.imshow("UniFace Eye Keyboard (ESC to quit)", disp)
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    center_h = float(np.mean(calib_h)) if len(calib_h) > 0 else 0.5
    center_v = float(np.mean(calib_v)) if len(calib_v) > 0 else 0.5
    print(f"Calibration done: center_h={center_h:.3f}, center_v={center_v:.3f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera in main loop")
            break

        faces = app.get(frame)
        frame_h, frame_w = frame.shape[:2]

        h_ratio = 0.5
        v_ratio = 0.5
        smooth_h = 0.5
        row, col = 1, 1
        ear = 1.0

        if faces:
            face = faces[0]
            if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                lms = face.landmark_2d_106

                le = build_eye_from_landmarks(lms, side="left")
                re = build_eye_from_landmarks(lms, side="right")
                if le is not None and re is not None:
                    left_ear = eye_aspect_ratio_6pts(le)
                    right_ear = eye_aspect_ratio_6pts(re)
                    ear_values = [v for v in (left_ear, right_ear) if v is not None]
                    ear = float(np.mean(ear_values)) if ear_values else 1.0
                else:
                    ear = 1.0

                print(f"EAR live: {ear:.3f}")

                xs = lms[:, 0]
                ys = lms[:, 1]
                face_cx = xs.mean()
                face_cy = ys.mean()
                h_ratio = float((face_cx - xs.min()) / max(xs.max() - xs.min(), 1))
                v_ratio = float((face_cy - ys.min()) / max(ys.max() - ys.min(), 1))

                gaze_queue.append(h_ratio)
                smooth_h = float(np.mean(gaze_queue)) if len(gaze_queue) > 0 else h_ratio

                row, col = get_grid_from_ratios(smooth_h, v_ratio)

                cv2.putText(frame, f"EAR:{ear:.3f}", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"H:{safe_ratio(smooth_h):.2f} V:{safe_ratio(v_ratio):.2f}", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                if le is not None:
                    for (x, y) in le:
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                if re is not None:
                    for (x, y) in re:
                        cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

                if ear is not None and ear <= EAR_CLOSED_MAX:
                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                else:
                    if eyes_closed_start is not None:
                        blink_duration = time.time() - eyes_closed_start
                        eyes_closed_start = None
                        print(f"Blink ended. Duration={blink_duration:.3f}s, EAR={ear:.3f}")
                        if blink_duration >= MIN_BLINK_DURATION:
                            selected_key = keys[row][col] if col < len(keys[row]) else ""
                            if selected_key == "<SPACE>":
                                sentence += " "
                            else:
                                sentence += selected_key
                            print("Selected (single):", selected_key, "->", sentence)
                            audio_say(f"{selected_key} selected")

        else:
            row, col = get_grid_from_ratios(0.5, 0.5)

        safe_h = safe_ratio(smooth_h if 'smooth_h' in locals() else 0.5)
        safe_v = safe_ratio(v_ratio if 'v_ratio' in locals() else 0.5)

        frame = draw_keyboard(frame, row, col)
        cx = int(safe_h * frame_w)
        cv2.circle(frame, (cx, 40), 12, (255, 0, 0), -1)

        cv2.rectangle(frame, (10, 10),
                      (min(1200, frame.shape[1] - 10), 70),
                      (30, 30, 30), -1)
        cv2.putText(frame, sentence[-60:], (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)

        cv2.imshow("UniFace Eye Keyboard (ESC to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord("c"):
            sentence = ""

    cap.release()
    cv2.destroyAllWindows()
    print("main() finished")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()