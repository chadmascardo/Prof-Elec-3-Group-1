import cv2
import numpy as np
import os
import time
q
# ============================================================
# CONFIGURATION
# ============================================================

IP_CAM_URL      = "http://172.19.245.134:8080/videofeed"  # CHANGE IP if needed
KNOWN_FACES_DIR = "known_faces"   # folder with reference photos

CONFIDENCE_MIN  = 0.55   # 0.0 to 1.0 — lower = more matches, higher = stricter
PROCESS_EVERY   = 5      # only analyze every Nth frame (higher = faster but less responsive)
DISPLAY_SIZE    = (640, 480)   # display resolution
PROCESS_SIZE    = (320, 240)   # processing resolution (smaller = faster)

# ============================================================
# LOAD DETECTOR
# ============================================================

def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf  = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Could not load Haar Cascade from: {path}")
    return clf


# ============================================================
# LOAD KNOWN FACES
# ============================================================

def load_known_faces(detector):
    known = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        print(f"Created '{KNOWN_FACES_DIR}/' folder — add photos there and restart")
        return known

    print("Loading known faces...")
    for filename in sorted(os.listdir(KNOWN_FACES_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name  = os.path.splitext(filename)[0]
        path  = os.path.join(KNOWN_FACES_DIR, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"  SKIP: Could not read {filename}")
            continue

        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"  SKIP: No face found in {filename}")
            continue

        (x, y, w, h) = faces[0]
        face_crop     = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        known.append({"name": name, "face": face_crop})
        print(f"  OK: {name}")

    print(f"Loaded {len(known)} face(s): {', '.join(p['name'] for p in known)}\n")
    return known


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize(face_gray, known_faces):
    face_resized = cv2.resize(face_gray, (100, 100))
    best_name    = "Unknown"
    best_score   = 0.0

    for person in known_faces:
        result = cv2.matchTemplate(face_resized, person["face"], cv2.TM_CCOEFF_NORMED)
        score  = float(result[0][0])
        if score > best_score:
            best_score = score
            best_name  = person["name"]

    if best_score >= CONFIDENCE_MIN:
        return best_name, best_score
    return "Unknown", best_score


# ============================================================
# MAIN
# ============================================================

def main():
    detector    = load_detector()
    known_faces = load_known_faces(detector)

    if not known_faces:
        print("No known faces loaded — add photos to known_faces/ and restart")
        return

    # Connect to IP cam
    print(f"Connecting to camera: {IP_CAM_URL}")
    cap = cv2.VideoCapture(IP_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not connect to camera!")
        print("  → Make sure IP Webcam app is running on your phone")
        print("  → Check that the IP address is correct")
        return

    print("Connected!")
    print("Controls: [Q] Quit  [S] Save frame  [R] Reset\n")

    frame_count = 0
    last_faces  = []        # (x, y, w, h, name, score)
    fps_time    = time.time()
    fps         = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection — reconnecting...")
            time.sleep(1)
            cap = cv2.VideoCapture(IP_CAM_URL)
            continue

        frame_count += 1
        display = cv2.resize(frame, DISPLAY_SIZE)

        # ── Process every Nth frame only ─────────────────────
        if frame_count % PROCESS_EVERY == 0:
            small = cv2.resize(frame, PROCESS_SIZE)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            scale_x = DISPLAY_SIZE[0] / PROCESS_SIZE[0]
            scale_y = DISPLAY_SIZE[1] / PROCESS_SIZE[1]

            raw_faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(40, 40)
            )

            last_faces = []
            for (x, y, w, h) in (raw_faces if len(raw_faces) else []):
                # Scale coords back to display size
                dx = int(x * scale_x)
                dy = int(y * scale_y)
                dw = int(w * scale_x)
                dh = int(h * scale_y)

                # Crop face from display frame for recognition
                face_crop  = cv2.cvtColor(
                    display[dy:dy+dh, dx:dx+dw], cv2.COLOR_BGR2GRAY
                )
                name, score = recognize(face_crop, known_faces)
                last_faces.append((dx, dy, dw, dh, name, score))

        # ── Draw results on every frame ───────────────────────
        for (x, y, w, h, name, score) in last_faces:
            color = (0, 210, 0) if name != "Unknown" else (0, 0, 210)
            label = f"{name} {score:.0%}"

            # Box
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(display, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(display, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # ── FPS counter ───────────────────────────────────────
        now = time.time()
        if now - fps_time >= 1.0:
            fps      = frame_count
            fps_time = now
            frame_count = 0

        # ── Status bar ────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (DISPLAY_SIZE[0], 28), (20, 20, 20), -1)
        status = f"FPS: {fps}  |  Faces: {len(last_faces)}  |  Known: {len(known_faces)}  |  [Q] Quit  [S] Save"
        cv2.putText(display, status, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Face Recognition - Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fn = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(fn, display)
            print(f"Saved: {fn}")
        elif key == ord('r'):
            last_faces  = []
            frame_count = 0
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()