import os
import csv
import io
import cv2
import datetime
import time
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super_secure_secret_session_key_12345")

# --- SYSTEM CONFIGURATION (from .env) ---
PHONE_1_URL   = os.getenv("PHONE_1_URL", "http://192.168.68.105:8080/stream.mjpg")
PHONE_2_URL   = os.getenv("PHONE_2_URL", "http://172.19.245.134:8080/videofeed")
BACKUP_ADMIN_ID = os.getenv("BACKUP_ADMIN_ID", "ADMIN2026")
MOG2_MIN_AREA = int(os.getenv("MOG2_MIN_AREA", 1500))
PROFILE_DIR   = "admin_profile"

# Ensure directories exist
for folder in ["static", PROFILE_DIR]:
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------

def get_db():
    """Open a new MySQL connection using .env credentials."""
    return mysql.connector.connect(
        host     = os.getenv("DB_HOST", "localhost"),
        port     = int(os.getenv("DB_PORT", 3306)),
        user     = os.getenv("DB_USER", "root"),
        password = os.getenv("DB_PASSWORD", ""),
        database = os.getenv("DB_NAME", "cctv_db"),
    )

def init_db():
    """Create tables if they don't exist yet."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS rooms (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            room_id    VARCHAR(50) NOT NULL UNIQUE,
            camera     VARCHAR(100) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS classrooms (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            room_id     VARCHAR(50) NOT NULL,
            offer_id    VARCHAR(50) NOT NULL,
            time        VARCHAR(50) NOT NULL,
            day_of_week VARCHAR(20) NOT NULL,
            UNIQUE KEY uq_schedule (room_id, offer_id, day_of_week, time)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS logs (
            id            INT AUTO_INCREMENT PRIMARY KEY,
            timestamp     DATETIME NOT NULL,
            message       TEXT NOT NULL,
            snapshot      VARCHAR(255) DEFAULT NULL
        )
        """,
    ]
    try:
        conn = get_db()
        cur  = conn.cursor()
        for stmt in ddl:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ [DB] Tables ready.")
    except Error as e:
        print(f"⚠️  [DB] init_db error: {e}")

init_db()

# ---------------------------------------------------------------------------
# LOGGING (DB + flat-file fallback)
# ---------------------------------------------------------------------------

def log_event(message, snapshot_name=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # DB
    try:
        conn = get_db()
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO logs (timestamp, message, snapshot) VALUES (%s, %s, %s)",
            (timestamp, message, snapshot_name)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Error as e:
        print(f"⚠️  [DB] log_event error: {e}")

def check_time_status():
    ph_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    t = ph_now.strftime("%H:%M")
    return "CLASS_HOURS" if "09:00" <= t <= "16:00" else "OFF_HOURS"

# ---------------------------------------------------------------------------
# FACE VERIFICATION
# ---------------------------------------------------------------------------

_face_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
is_model_trained = False

def train_admin_model():
    global is_model_trained
    img_path = os.path.join(PROFILE_DIR, "admin.jpg")
    if not os.path.exists(img_path):
        print(f"⚠️  No admin.jpg found at '{img_path}'. Face auth will use generic check.")
        is_model_trained = False
        return
    ref = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        is_model_trained = False
        return
    faces = _face_cascade.detectMultiScale(ref, 1.1, 5)
    roi   = ref[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]] if len(faces) else ref
    face_recognizer.train([roi], np.array([1]))
    is_model_trained = True
    print("✅ [Face] Model trained on admin.jpg.")

train_admin_model()

def verify_face_match(frame):
    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return "NO_FACE"
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    if not is_model_trained:
        return "GENERIC_MATCH"
    label, conf = face_recognizer.predict(roi)
    return "VERIFIED_ADMIN" if label == 1 and conf < 75 else "UNKNOWN_USER"

# ---------------------------------------------------------------------------
# AUTH ROUTES
# ---------------------------------------------------------------------------

@app.route("/login")
def login_page():
    return render_template("login.html", error=None)

@app.route("/backup_auth", methods=["POST"])
def backup_auth():
    if request.form.get("admin_id") == BACKUP_ADMIN_ID:
        session["is_admin"] = True
        log_event("Admin logged in via Backup ID.")
        return redirect(url_for("index"))
    return render_template("login.html", error="Invalid Backup Admin ID Code.")

@app.route("/face_auth")
def face_auth():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return render_template("login.html", error="Webcam access blocked. Close Teams/Zoom first.")
    time.sleep(0.8)
    ret, frame = cam.read()
    cam.release()
    if not ret or frame is None:
        return render_template("login.html", error="Webcam capture timed out. Try again.")
    result = verify_face_match(frame)
    if result in ["VERIFIED_ADMIN", "GENERIC_MATCH"]:
        session["is_admin"] = True
        log_event(f"Admin logged in via Face Verification ({result}).")
        return redirect(url_for("index"))
    elif result == "NO_FACE":
        return render_template("login.html", error="No face detected. Look straight at the webcam.")
    return render_template("login.html", error="Access Denied. Face does not match admin profile.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if not session.get("is_admin"):
        return redirect(url_for("login_page"))
    return render_template("index.html")

# ---------------------------------------------------------------------------
# CAMERA STREAMS
# ---------------------------------------------------------------------------

def generate_stream(stream_url, camera_id):
    cap = cv2.VideoCapture(stream_url, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=35, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    last_log_time = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(1)
            cap = cv2.VideoCapture(stream_url, cv2.CAP_ANY)
            continue

        fg_mask = bg_sub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(cv2.contourArea(c) > MOG2_MIN_AREA for c in contours)

        if motion_detected:
            now = time.time()
            if now - last_log_time > 5:
                if check_time_status() == "CLASS_HOURS":
                    log_event(f"Camera {camera_id}: Routine movement during class hours.")
                else:
                    fname = f"cam{camera_id}_alert_{int(now)}.jpg"
                    cv2.imwrite(f"static/{fname}", frame)
                    log_event(f"!!! SECURITY ALERT !!! Off-hours activity on Camera {camera_id}.", fname)
                last_log_time = now

        h_f, w_f = frame.shape[:2]
        if motion_detected:
            cv2.rectangle(frame, (0, 0), (190, 32), (0, 0, 255), -1)
            cv2.putText(frame, "MOVEMENT ALERT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (110, 26), (40, 40, 40), -1)
            cv2.putText(frame, "SYSTEM IDLE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM {camera_id} | {ts}", (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        ret2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret2:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

@app.route("/video_feed_1")
def video_feed_1():
    if not session.get("is_admin"): return Response(status=403)
    return Response(generate_stream(PHONE_1_URL, 1), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_2")
def video_feed_2():
    if not session.get("is_admin"): return Response(status=403)
    return Response(generate_stream(PHONE_2_URL, 2), mimetype="multipart/x-mixed-replace; boundary=frame")

# ---------------------------------------------------------------------------
# LOGS PAGE
# ---------------------------------------------------------------------------

@app.route("/logs_page")
def logs_page():
    if not session.get("is_admin"): return redirect(url_for("login_page"))
    log_list = []
    try:
        conn = get_db()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT timestamp, message, snapshot FROM logs ORDER BY timestamp DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for row in rows:
            log_list.append({
                "time":     str(row["timestamp"]),
                "msg":      row["message"],
                "snapshot": row["snapshot"],
            })
    except Error as e:
        print(f"⚠️  [DB] logs_page error: {e}")
    return render_template("logs.html", log_list=log_list)

# ---------------------------------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------------------------------

@app.route("/upload")
def upload_page():
    if not session.get("is_admin"): return redirect(url_for("login_page"))
    return render_template("upload.html")


@app.route("/upload/rooms", methods=["POST"])
def upload_rooms():
    if not session.get("is_admin"): return jsonify({"error": "Unauthorized"}), 403
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a valid .csv file."}), 400

    stream  = io.StringIO(file.stream.read().decode("utf-8-sig"))
    reader  = csv.DictReader(stream)

    # Normalize headers (strip whitespace, lowercase)
    required = {"room_id", "camera"}
    headers  = {h.strip().lower() for h in (reader.fieldnames or [])}
    missing  = required - headers
    if missing:
        return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

    inserted = updated = skipped = 0
    errors   = []

    try:
        conn = get_db()
        cur  = conn.cursor()
        for i, row in enumerate(reader, start=2):
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            room_id = row.get("room_id", "")
            camera  = row.get("camera", "")
            if not room_id or not camera:
                errors.append(f"Row {i}: empty room_id or camera — skipped.")
                skipped += 1
                continue
            cur.execute(
                """
                INSERT INTO rooms (room_id, camera)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE camera = VALUES(camera)
                """,
                (room_id, camera)
            )
            if cur.rowcount == 1:
                inserted += 1
            else:
                updated += 1
        conn.commit()
        cur.close()
        conn.close()
    except Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify({
        "success": True,
        "inserted": inserted,
        "updated":  updated,
        "skipped":  skipped,
        "errors":   errors,
    })


@app.route("/upload/classrooms", methods=["POST"])
def upload_classrooms():
    if not session.get("is_admin"): return jsonify({"error": "Unauthorized"}), 403
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a valid .csv file."}), 400

    stream   = io.StringIO(file.stream.read().decode("utf-8-sig"))
    reader   = csv.DictReader(stream)
    required = {"room_id", "offer_id", "time", "day_of_week"}
    headers  = {h.strip().lower() for h in (reader.fieldnames or [])}
    missing  = required - headers
    if missing:
        return jsonify({"error": f"Missing columns: {', '.join(missing)}"}), 400

    inserted = updated = skipped = 0
    errors   = []

    try:
        conn = get_db()
        cur  = conn.cursor()
        for i, row in enumerate(reader, start=2):
            row      = {k.strip().lower(): v.strip() for k, v in row.items()}
            room_id  = row.get("room_id", "")
            offer_id = row.get("offer_id", "")
            sched_time = row.get("time", "")
            dow      = row.get("day_of_week", "")
            if not all([room_id, offer_id, sched_time, dow]):
                errors.append(f"Row {i}: one or more empty fields — skipped.")
                skipped += 1
                continue
            cur.execute(
                """
                INSERT INTO classrooms (room_id, offer_id, time, day_of_week)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE time = VALUES(time)
                """,
                (room_id, offer_id, sched_time, dow)
            )
            if cur.rowcount == 1:
                inserted += 1
            else:
                updated += 1
        conn.commit()
        cur.close()
        conn.close()
    except Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify({
        "success":  True,
        "inserted": inserted,
        "updated":  updated,
        "skipped":  skipped,
        "errors":   errors,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)