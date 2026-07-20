import os
import cv2
import csv
import numpy as np
import datetime
import time
from flask import Flask, Response, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "super_secure_secret_session_key_12345"

# --- SYSTEM SETTINGS ---
PHONE_1_URL = "http://172.19.254.179:8080/stream.mjpg"   # iPhone (SimpleIPCam)
PHONE_2_URL = "0026"           # Android (IP Webcam)
BACKUP_ADMIN_ID = "ADMIN2026"

# --- DETECTION TUNING ---
MOG2_MIN_AREA  = 1500   # px² — minimum motion blob to bother running DNN on
DNN_CONFIDENCE = 0.50   # confidence threshold for MobileNet-SSD
DNN_FRAME_INTERVAL = 5  # run the DNN every Nth motion frame; reuse boxes in between

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

# --- CSV FILE INFRASTRUCTURE PATHS ---
LOG_FILE_CSV   = "data/motion_logs.csv"
STUDENTS_CSV   = "data/students.csv"
SCHEDULES_CSV  = "data/schedules.csv"
ENROLLMENT_CSV = "data/enrollment.csv"

# Automatically create all system operational directories
for folder in ["data", "static", "models"]:
    os.makedirs(folder, exist_ok=True)

# Initialize CSV structures with standard headers explicitly mapping requirements
def init_csv_files():
    if not os.path.exists(LOG_FILE_CSV):
        with open(LOG_FILE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "camera_id", "status", "snapshot"])
            
    if not os.path.exists(STUDENTS_CSV):
        with open(STUDENTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["studentID", "lastname", "firstname"])
            
    if not os.path.exists(SCHEDULES_CSV):
        with open(SCHEDULES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["courseCode", "days", "time", "room"])
            # Auto-populating example blocks given in requirements document
            writer.writerow(["11012", "MWF", "09:00-11:00", "IC1001"])
            writer.writerow(["22024", "TTH", "12:00-16:00", "IC1002"])

    if not os.path.exists(ENROLLMENT_CSV):
        with open(ENROLLMENT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["studentID", "courseCode"])
            writer.writerow(["202500001", "11012"])

init_csv_files()

# Track active state variables globally across stream loops for empty-room triggers
room_states = {1: "IDLE", 2: "IDLE"}
last_living_detection_time = {1: 0.0, 2: 0.0}

# ---------------------------------------------------------------------------
# LOAD DNN ONCE AT STARTUP
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
                "  MobileNetSSD_deploy.caffemodel"
            )
        _dnn_net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
    return _dnn_net


def contains_living_thing(frame):
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
    detections = net.forward()

    found = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < DNN_CONFIDENCE:
            continue
        class_id = int(detections[0, 0, i, 1])
        label = SSD_CLASSES[class_id] if class_id < len(SSD_CLASSES) else "unknown"
        if label in LIVING_CLASSES:
            box = detections[0, 0, i, 3:7]
            # DNN can emit inf/NaN coords on noisy frames — drop those boxes
            if np.isfinite(box).all():
                found.append((label, conf, box))

    return bool(found), found


def draw_living_detections(frame, detections):
    h, w = frame.shape[:2]
    for label, conf, box in detections:
        # Sanitize: skip non-finite boxes, clamp coords into the frame
        if not np.isfinite(box).all():
            continue
        x1 = int(np.clip(box[0] * w, 0, w - 1))
        y1 = int(np.clip(box[1] * h, 0, h - 1))
        x2 = int(np.clip(box[2] * w, 0, w - 1))
        y2 = int(np.clip(box[3] * h, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 50), 2)
        tag = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 50), -1)
        cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


# ---------------------------------------------------------------------------
# CORE UTILITIES (CSV SPECIFIC LOGGING & CAMERA-SPECIFIC TIME CHECKING)
# ---------------------------------------------------------------------------

def log_event_csv(camera_id, status_message, snapshot_name="NONE"):
    """Appends structural system tracking occurrences into the CSV log workbook."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, camera_id, status_message, snapshot_name])


def check_dynamic_schedule(camera_id):
    """Parses schedules.csv dynamically using custom runtime blocks per camera."""
    now = datetime.datetime.now()
    
    # Map Python day identifiers to academic character tags
    day_map = ["M", "T", "W", "H", "F", "S", "U"]
    current_day_char = day_map[now.weekday()]
    current_time_str = now.strftime("%H:%M")
    
    if not os.path.exists(SCHEDULES_CSV):
        return "OFF_HOURS"

    with open(SCHEDULES_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if current_day_char in row["days"]:
                try:
                    start_str, end_str = row["time"].split("-")
                    start_str, end_str = start_str.strip(), end_str.strip()
                    
                    # Enforce camera-specific rules using the file logic windows
                    if camera_id == 1 and (start_str >= "09:00" and end_str <= "11:00"):
                        if start_str <= current_time_str <= end_str:
                            return "CLASS_HOURS"
                    elif camera_id == 2 and (start_str >= "12:00" and end_str <= "16:00"):
                        if start_str <= current_time_str <= end_str:
                            return "CLASS_HOURS"
                except ValueError:
                    continue
                    
    return "OFF_HOURS"


_face_cascade = None
_eye_cascade  = None

def _load_cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if _face_cascade.empty() or _eye_cascade.empty():
        raise RuntimeError(
            "Haar cascade files failed to load — broken OpenCV install. "
            "Run: pip uninstall opencv-python opencv-contrib-python, "
            "then: pip install \"opencv-contrib-python<5\""
        )


def detect_faces(frame):
    _load_cascades()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )

    confirmed = 0
    for (fx, fy, fw, fh) in faces if len(faces) else []:
        roi = gray[fy : fy + int(fh * 0.6), fx : fx + fw]
        eyes = _eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        if len(eyes) >= 1:
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
        log_event_csv("SYS", "Admin successfully logged in via Backup ID validation.")
        return redirect(url_for("index"))
    return render_template("login.html", error="Invalid Backup Admin ID Code.")


@app.route("/face_auth")
def face_auth():
    try:
        cam = cv2.VideoCapture(0)
        time.sleep(0.8)
        ret, frame = cam.read()
        cam.release()

        if not ret or frame is None:
            return render_template("login.html", error="Webcam failed to launch. Try using ID Backup login.")

        face_count = detect_faces(frame)
    except Exception as exc:
        log_event_csv("SYS", f"Face auth error: {exc}")
        return render_template(
            "login.html",
            error="Face recognition hit an internal error (see motion_logs.csv). Use ID Backup login.",
        )
    if face_count > 0:
        session["is_admin"] = True
        log_event_csv("SYS", "Admin verified and logged in via Face + Eye Detection.")
        return redirect(url_for("index"))

    return render_template("login.html", error="No face detected. Look directly at the webcam in good lighting and try again.")


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


@app.route("/upload")
def upload_page():
    """Render the CSV upload screen linked from the admin navigation."""
    if not session.get("is_admin"):
        return redirect(url_for("login_page"))
    return render_template("Upload.html")


# ---------------------------------------------------------------------------
# TWO-STAGE MOTION GENERATOR ENGINE WITH VACANCY MONITORING
# ---------------------------------------------------------------------------

def open_capture(stream_url):
    cap = cv2.VideoCapture(stream_url, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # freshest frame only — larger buffers add lag
    return cap


def generate_stream(stream_url, camera_id):
    global room_states, last_living_detection_time
    
    MAX_CONSEC_FAILS = 5
    RECONNECT_WAIT   = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

    cap    = open_capture(stream_url)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
    
    last_log_time  = 0
    consec_failures = 0

    # DNN frame-skip cache — the forward pass costs 100-300 ms on CPU, so run
    # it every Nth motion frame and reuse the last boxes in between.
    frames_since_dnn = DNN_FRAME_INTERVAL   # force a run on the first motion frame
    last_living_alert = False
    last_living_detections = []

    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None:
            consec_failures += 1
            if consec_failures >= MAX_CONSEC_FAILS:
                if placeholder:
                    yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
                cap.release()
                time.sleep(RECONNECT_WAIT)
                cap    = open_capture(stream_url)
                bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
                consec_failures = 0
            continue

        consec_failures = 0
        current_time = time.time()

        # --- STAGE 1 — MOG2 background motion calculation ---
        motion_blobs = []
        try:
            fg_mask = bg_sub.apply(frame)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_blobs = [c for c in contours if cv2.contourArea(c) > MOG2_MIN_AREA]
        except Exception as exc:
            log_event_csv(camera_id, f"MOG2 runtime processing fault: {exc}")

        # --- STAGE 2 — MobileNet-SSD Verification ---
        living_alert      = False
        living_detections = []

        if motion_blobs:
            if frames_since_dnn >= DNN_FRAME_INTERVAL:
                frames_since_dnn = 0
                try:
                    last_living_alert, last_living_detections = contains_living_thing(frame)
                except Exception as exc:
                    log_event_csv(camera_id, f"DNN evaluation pipeline fault: {exc}")
            frames_since_dnn += 1
            living_alert, living_detections = last_living_alert, last_living_detections

            if living_alert:
                last_living_detection_time[camera_id] = current_time
                
                # State Update: Check if shifting state into occupied mode
                if room_states[camera_id] != "OCCUPIED":
                    room_states[camera_id] = "OCCUPIED"
                    log_event_csv(camera_id, "Motion status: Area is now actively occupied.")

                # Logging anti-spam check (triggers every 5 seconds)
                if current_time - last_log_time > 5:
                    time_status = check_dynamic_schedule(camera_id)
                    labels_str = ", ".join(sorted({l for l, _, _ in living_detections}))
                    
                    if time_status == "CLASS_HOURS":
                        log_event_csv(camera_id, f"Routine monitoring activity: [{labels_str}] spotted during scheduled session.", "NONE")
                    else:
                        # Off-hours breach detected: write alert image file and log security event
                        img_filename = f"cam{camera_id}_alert_{int(current_time)}.jpg"
                        cv2.imwrite(f"static/{img_filename}", frame)
                        log_event_csv(camera_id, f"!!! SECURITY WARNING !!! Off-hours room occupation: [{labels_str}]", img_filename)
                        
                    last_log_time = current_time
        else:
            # Motion stopped — drop cached boxes, re-arm an immediate DNN run
            frames_since_dnn = DNN_FRAME_INTERVAL
            last_living_alert, last_living_detections = False, []

            # Empty Room Condition: If system was marked active but no targets have been
            # detected for 10 straight seconds, register the space as vacant.
            if room_states[camera_id] == "OCCUPIED" and (current_time - last_living_detection_time[camera_id] > 10.0):
                room_states[camera_id] = "EMPTY"
                log_event_csv(camera_id, "Log tracking update: No users remaining in the area.", "NONE")

        # --- CCTV VIEW GRAPHICS OVERLAYS ---
        h_f, w_f = frame.shape[:2]

        if living_alert:
            try:
                draw_living_detections(frame, living_detections)
            except Exception as exc:
                log_event_csv(camera_id, f"Overlay drawing fault: {exc}")
            cv2.rectangle(frame, (0, 0), (295, 38), (0, 0, 0), -1)
            cv2.putText(frame, "!! LIVING THING DETECTED !!", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 60), 2)
        elif motion_blobs:
            cv2.rectangle(frame, (0, 0), (200, 30), (20, 20, 20), -1)
            cv2.putText(frame, "MOTION (non-living)", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1)
        else:
            cv2.rectangle(frame, (0, 0), (125, 26), (20, 20, 20), -1)
            cv2.putText(frame, f"ROOM {room_states[camera_id]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 200, 60) if room_states[camera_id] == "EMPTY" else (60, 60, 60), 1)

        cam_label = f"CAM {camera_id:02d}"
        (lw, lh), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (w_f - lw - 12, 0), (w_f, lh + 10), (20, 20, 20), -1)
        cv2.putText(frame, cam_label, (w_f - lw - 6, lh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        (tw, th), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (0, h_f - 32), (tw + 36, h_f), (0, 0, 0), -1)
        
        if int(time.time()) % 2 == 0:
            cv2.circle(frame, (14, h_f - 10), 6, (0, 0, 220), -1)
        cv2.putText(frame, ts, (28, h_f - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        ret2, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret2:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/video_feed_1")
def video_feed_1():
    if not session.get("is_admin"): return Response(status=403)
    return Response(generate_stream(PHONE_1_URL, 1), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed_2")
def video_feed_2():
    if not session.get("is_admin"): return Response(status=403)
    return Response(generate_stream(PHONE_2_URL, 2), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs_page")
def logs_page():
    if not session.get("is_admin"): return redirect(url_for("login_page"))
    parsed_logs = []
    if os.path.exists(LOG_FILE_CSV):
        with open(LOG_FILE_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_logs.append({
                    "time": row["timestamp"],
                    "msg": f"Camera {row['camera_id']}: {row['status']}",
                    "snapshot": None if row["snapshot"] == "NONE" else row["snapshot"]
                })
    return render_template("logs.html", log_list=parsed_logs[::-1])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)