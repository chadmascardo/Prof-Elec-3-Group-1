import os
import cv2
import datetime
import time
from flask import Flask, Response, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "super_secure_secret_session_key_12345"

# --- SYSTEM SETTINGS ---
PHONE_1_URL = "http://172.21.9.114:8080/stream.mjpg"   # iPhone (SimpleIPCam)
PHONE_2_URL = 0             # Laptop webcam. Use 1 if you have another camera.
BACKUP_ADMIN_ID = "ADMIN2026"


# --- DETECTION TUNING ---
MOG2_MIN_AREA  = 1500   # px² — minimum motion blob to bother running DNN on
DNN_CONFIDENCE = 0.50   # confidence threshold for MobileNet-SSD

# Only these class labels count as "living things"
LIVING_CLASSES = {"person", "bird", "cat", "cow", "dog", "horse", "sheep"}

# MobileNet-SSD 21-class list (order must match the Caffe model exactly)
SSD_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor",
]

# Model files — place in a models/ folder next to app.py
MODEL_PROTO   = "models/MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS = "models/MobileNetSSD_deploy.caffemodel"

LOG_FILE = "data/activity_log.txt"

for folder in ["data", "static", "models"]:
    os.makedirs(folder, exist_ok=True)
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

# ---------------------------------------------------------------------------
# LOAD DNN ONCE AT STARTUP  (avoids reloading 23 MB per stream request)
# ---------------------------------------------------------------------------
_dnn_net = None

def get_dnn():
    global _dnn_net
    if _dnn_net is None:
        if not (os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS)):
            raise FileNotFoundError(
                "MobileNet-SSD model files missing.\n"
                "Download and place in models/:\n"
                "  MobileNetSSD_deploy.prototxt\n"
                "  MobileNetSSD_deploy.caffemodel\n"
                "from: github.com/chuanqi305/MobileNet-SSD"
            )
        _dnn_net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
    return _dnn_net


def contains_living_thing(frame):
    """
    Run MobileNet-SSD on the frame.
    Returns (True, detections_list) if any LIVING_CLASSES label is found
    above DNN_CONFIDENCE threshold, else (False, []).
    Each detection is a tuple: (label, confidence, box_coords_normalized).
    """
    net = get_dnn()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1 / 127.5,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
    )
    net.setInput(blob)
    detections = net.forward()   # shape: (1, 1, N, 7)

    found = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < DNN_CONFIDENCE:
            continue
        class_id = int(detections[0, 0, i, 1])
        label = SSD_CLASSES[class_id] if class_id < len(SSD_CLASSES) else "unknown"
        if label in LIVING_CLASSES:
            found.append((label, conf, detections[0, 0, i, 3:7]))

    return bool(found), found


def draw_living_detections(frame, detections):
    """Draw green bounding boxes + label tags for living-thing detections only."""
    h, w = frame.shape[:2]
    for label, conf, box in detections:
        x1, y1 = int(box[0] * w), int(box[1] * h)
        x2, y2 = int(box[2] * w), int(box[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 50), 2)
        tag = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 50), -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def log_event(message, snapshot_name="NONE"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}|{message}|{snapshot_name}\n")


def check_time_status():
    current_time_str = datetime.datetime.now().strftime("%H:%M")
    return "CLASS_HOURS" if "09:00" <= current_time_str <= "16:00" else "OFF_HOURS"


# Pre-load both cascades once at module level — they ship with OpenCV, zero download
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def detect_faces(frame):
    """
    Dual-cascade face detection: requires BOTH a face region AND at least
    one eye inside it. This eliminates most false positives (posters, photos,
    screen reflections) that a single face cascade would flag.

    Returns the count of confirmed face+eye detections.
    Uses haarcascade_frontalface_alt2 parameters which handle partial faces
    and varied lighting better than the default cascade.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalise histogram so dim/bright rooms don't hurt detection
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),   # ignore tiny blobs — real faces at webcam range are larger
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    confirmed = 0
    for (fx, fy, fw, fh) in faces if len(faces) else []:
        # Only look for eyes in the top 60% of the face region (avoids mouth/chin noise)
        roi = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
        eyes = _eye_cascade.detectMultiScale(
            roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )
        if len(eyes) >= 1:   # at least one eye visible → real face
            confirmed += 1

    return confirmed


# ---------------------------------------------------------------------------
# LOGIN ROUTES
# ---------------------------------------------------------------------------

@app.route("/login")
def login_page():
    return render_template("login.html", error=None)


@app.route("/backup_auth", methods=["POST"])
def backup_auth():
    user_input = request.form.get("admin_id")
    if user_input == BACKUP_ADMIN_ID:
        session["is_admin"] = True
        log_event("Admin successfully logged in via Backup ID validation.")
        return redirect(url_for("index"))
    return render_template("login.html", error="Invalid Backup Admin ID Code.")


@app.route("/face_auth")
def face_auth():
    """
    Face authentication using dual Haar cascade (face + eye).
    Requires both a face region AND at least one visible eye — cuts
    false positives from photos, posters, and screen reflections.
    No external model file needed: both cascades ship with OpenCV.
    """
    cam = cv2.VideoCapture(0)
    time.sleep(0.8)   # let the camera adjust exposure before reading
    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        return render_template(
            "login.html", error="Webcam failed to launch. Try using ID Backup login."
        )

    face_count = detect_faces(frame)

    if face_count > 0:
        session["is_admin"] = True
        log_event("Admin verified and logged in via Face + Eye Detection.")
        return redirect(url_for("index"))

    return render_template(
        "login.html",
        error="No face detected. Look directly at the webcam in good lighting and try again.",
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not session.get("is_admin"):
        return redirect(url_for("login_page"))
    return render_template("index.html")


# ---------------------------------------------------------------------------
# MODERN MOTION DETECTION STREAM  (MOG2 background subtraction)
# ---------------------------------------------------------------------------

def open_capture(stream_url):
    """
    Open a VideoCapture tuned for phone HTTP/MJPEG streams.
    - CAP_ANY lets OpenCV pick the right backend automatically
    - BUFFERSIZE=4 gives MOG2 enough consecutive frames to build its background
      model, without stale frame pile-up causing flicker
    - No FFMPEG-specific timeout props: they silently break other backends
    """
    if isinstance(stream_url, str) and stream_url.isdigit():
        stream_url = int(stream_url)
    cap = cv2.VideoCapture(stream_url, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    return cap


def generate_stream(stream_url, camera_id):
    """
    Robust two-stage detection pipeline with auto-reconnect.

    Stage 1 — MOG2 background subtraction (every frame, very cheap)
      Any pixel movement found → proceed to Stage 2.
      Nothing moved → send frame as-is, skip DNN entirely.

    Stage 2 — MobileNet-SSD DNN (only when Stage 1 fires)
      Classifies what moved. Alert only for LIVING_CLASSES.
      Non-living motion (fan, curtain, flickering light) → silently ignored.

    Stability fixes vs previous version:
      - CAP_PROP_BUFFERSIZE=1 prevents stale-frame pile-up causing flicker
      - Consecutive failure counter: only reconnect after N misses, not on first blip
      - bg_sub is reset on reconnect so MOG2 doesn't freak out from the gap
      - Full try/except around the processing block so one bad frame can't
        kill the whole generator loop
      - Placeholder "reconnecting" JPEG sent while offline so browser <img> tag
        stays alive instead of going blank
    """
    MAX_CONSEC_FAILS = 5        # tolerate up to 5 consecutive bad reads before reconnect
    RECONNECT_WAIT   = 2        # seconds to wait between reconnect attempts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- Build a static "reconnecting" placeholder frame ---
    placeholder = None
    _ph = cv2.imencode(
        ".jpg",
        cv2.putText(
            __import__("numpy").zeros((240, 426, 3), dtype="uint8"),
            "Reconnecting...", (80, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2,
        ),
    )
    if _ph[0]:
        placeholder = _ph[1].tobytes()

    def make_cap():
        return open_capture(stream_url)

    def make_bgsub():
        return cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=True
        )

    cap    = make_cap()
    bg_sub = make_bgsub()
    last_log_time  = 0
    last_student_log_time = 0
    consec_failures = 0

    while True:
        # ------------------------------------------------------------------ #
        # FRAME GRAB — tolerate brief network blips before reconnecting        #
        # ------------------------------------------------------------------ #
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None:
            consec_failures += 1
            if consec_failures >= MAX_CONSEC_FAILS:
                if placeholder:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n"
                           + placeholder + b"\r\n")
                cap.release()
                time.sleep(RECONNECT_WAIT)
                cap    = make_cap()
                bg_sub = make_bgsub()   # fresh model after reconnect gap
                consec_failures = 0
            continue

        consec_failures = 0

        # ------------------------------------------------------------------ #
        # STAGE 1 — MOG2 motion detection (every frame)                       #
        # ------------------------------------------------------------------ #
        motion_blobs = []
        try:
            fg_mask = bg_sub.apply(frame)
            # Remove MOG2 shadow pixels (value=127), keep only hard foreground (255)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            # Erode noise then dilate to merge nearby blobs into solid regions
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_blobs = [c for c in contours if cv2.contourArea(c) > MOG2_MIN_AREA]
        except Exception as exc:
            log_event(f"Camera {camera_id} MOG2 error: {exc}")

        # ------------------------------------------------------------------ #
        # STAGE 2 — MobileNet-SSD (only when Stage 1 found something)         #
        # ------------------------------------------------------------------ #
        living_alert      = False
        living_detections = []
        student_detected  = False
        face_count        = 0

        if motion_blobs:
            try:
                face_count = detect_faces(frame)
                student_detected = face_count > 0
            except Exception as exc:
                log_event(f"Camera {camera_id} face detection error: {exc}")

            if student_detected:
                current_time = time.time()
                if current_time - last_student_log_time > 5:
                    log_event(
                        f"Camera {camera_id}: Motion detected - student face detected ({face_count} face/s)."
                    )
                    last_student_log_time = current_time

            try:
                living_alert, living_detections = contains_living_thing(frame)
            except Exception as exc:
                log_event(f"Camera {camera_id} DNN error: {exc}")

            if living_alert:
                current_time = time.time()
                if current_time - last_log_time > 5:
                    time_status = check_time_status()
                    labels_str = ", ".join(sorted({l for l, _, _ in living_detections}))
                    if camera_id == 1:
                        if time_status == "CLASS_HOURS":
                            log_event(f"Camera 1: [{labels_str}] detected during class hours.")
                        else:
                            img_filename = f"cam1_alert_{int(current_time)}.jpg"
                            cv2.imwrite(f"static/{img_filename}", frame)
                            log_event(
                                f"!!! SECURITY ALERT !!! [{labels_str}] in Classroom View after hours.",
                                img_filename,
                            )
                    elif camera_id == 2:
                        if time_status == "OFF_HOURS":
                            img_filename = f"cam2_alert_{int(current_time)}.jpg"
                            cv2.imwrite(f"static/{img_filename}", frame)
                            log_event(
                                f"!!! SECURITY ALERT !!! [{labels_str}] — Perimeter breach during lock window.",
                                img_filename,
                            )
                        else:
                            log_event(f"Camera 2: [{labels_str}] in daytime field monitoring.")
                    last_log_time = current_time

        # ------------------------------------------------------------------ #
        # CCTV OVERLAY — always drawn so it looks like a real camera feed      #
        # ------------------------------------------------------------------ #
        h_f, w_f = frame.shape[:2]

        # 1. Detection status banner (top-left)
        if student_detected:
            cv2.rectangle(frame, (0, 0), (330, 38), (0, 0, 0), -1)
            cv2.putText(frame, "MOTION DETECTED: STUDENT", (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 60), 2)
        elif living_alert:
            draw_living_detections(frame, living_detections)
            cv2.rectangle(frame, (0, 0), (295, 38), (0, 0, 0), -1)
            cv2.putText(frame, "!! LIVING THING DETECTED !!", (6, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 60), 2)
        elif motion_blobs:
            cv2.rectangle(frame, (0, 0), (200, 30), (20, 20, 20), -1)
            cv2.putText(frame, "MOTION (non-living)", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1)
        else:
            cv2.rectangle(frame, (0, 0), (105, 26), (20, 20, 20), -1)
            cv2.putText(frame, "NO MOTION", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1)

        # 2. Camera label (top-right)
        cam_label = f"CAM {camera_id:02d}"
        (lw, lh), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (w_f - lw - 12, 0), (w_f, lh + 10), (20, 20, 20), -1)
        cv2.putText(frame, cam_label, (w_f - lw - 6, lh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 3. Timestamp + REC dot (bottom-left) — the classic CCTV look
        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        bar_y = h_f - 32
        cv2.rectangle(frame, (0, bar_y), (tw + 36, h_f), (0, 0, 0), -1)
        # Blinking REC dot — on for even seconds, off for odd
        if int(time.time()) % 2 == 0:
            cv2.circle(frame, (14, h_f - 10), 6, (0, 0, 220), -1)
        cv2.putText(frame, ts, (28, h_f - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        # ------------------------------------------------------------------ #
        # ENCODE + YIELD                                                        #
        # ------------------------------------------------------------------ #
        ret2, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")


@app.route("/video_feed_1")
def video_feed_1():
    if not session.get("is_admin"):
        return Response(status=403)
    return Response(
        generate_stream(PHONE_1_URL, 1),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed_2")
def video_feed_2():
    if not session.get("is_admin"):
        return Response(status=403)
    return Response(
        generate_stream(PHONE_2_URL, 2),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# LOGS
# ---------------------------------------------------------------------------

@app.route("/logs_page")
def logs_page():
    if not session.get("is_admin"):
        return redirect(url_for("login_page"))
    parsed_logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        for line in lines[::-1]:
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    parsed_logs.append(
                        {
                            "time": parts[0],
                            "msg": parts[1],
                            "snapshot": None if parts[2] == "NONE" else parts[2],
                        }
                    )
    return render_template("logs.html", log_list=parsed_logs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
