import os
import cv2
import datetime
import time
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "super_secure_secret_session_key_12345"

# --- SYSTEM CONFIGURATION ---
# Replace these IP addresses with your phone streaming links if they change
PHONE_1_URL = "http://192.168.68.105:8080/stream.mjpg"   # iPhone (SimpleIPCam)
PHONE_2_URL = "http://172.19.245.134:8080/videofeed"     # Android (IP Webcam)
BACKUP_ADMIN_ID = "ADMIN2026"

MOG2_MIN_AREA = 1500
LOG_FILE = "data/activity_log.txt"
PROFILE_DIR = "admin_profile"

# Ensure all system directories exist automatically
for folder in ["data", "static", PROFILE_DIR]:
    os.makedirs(folder, exist_ok=True)
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

# Load cascade files for face boundary checking
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# FACE VERIFICATION ENGINE (LBPH PROFILE MATCHER)
# ---------------------------------------------------------------------------
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
is_model_trained = False

def train_admin_model():
    """Reads your profile photo on startup and trains the recognition algorithm"""
    global is_model_trained
    img_path = os.path.join(PROFILE_DIR, "admin.jpg")
    
    if not os.path.exists(img_path):
        print(f"⚠️ [WARNING]: No reference image found at '{img_path}'. Defaulting to simple face checking.")
        is_model_trained = False
        return

    reference_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if reference_img is None:
        print("⚠️ [ERROR]: Cannot parse image. Check file format.")
        is_model_trained = False
        return

    faces = _face_cascade.detectMultiScale(reference_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("⚠️ [WARNING]: No face detected in 'admin.jpg'. Training on full image surface grid.")
        face_recognizer.train([reference_img], np.array([1]))
    else:
        (x, y, w, h) = faces[0]
        face_roi = reference_img[y:y+h, x:x+w]
        face_recognizer.train([face_roi], np.array([1])) # Label 1 = Authorized Admin
    
    print("✅ [SUCCESS]: Face Verification Model successfully trained against admin.jpg snapshot matrix.")
    is_model_trained = True

# Boot training instantly on script startup
train_admin_model()

def verify_face_match(frame):
    """Compares incoming webcam capture against trained admin matrix signatures"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return "NO_FACE"

    (x, y, w, h) = faces[0]
    captured_face_roi = gray[y:y+h, x:x+w]

    if not is_model_trained:
        return "GENERIC_MATCH"

    label, confidence = face_recognizer.predict(captured_face_roi)
    
    # Mathematical confidence proximity (lower means closer match)
    if label == 1 and confidence < 75:
        return "VERIFIED_ADMIN"
    
    return "UNKNOWN_USER"

# ---------------------------------------------------------------------------
# CORE UTILITIES
# ---------------------------------------------------------------------------

def log_event(message, snapshot_name="NONE"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}|{message}|{snapshot_name}\n")

def check_time_status():
    current_time_str = datetime.datetime.now().strftime("%H:%M")
    # Routine Classroom Hours Window: 9:00 AM to 4:00 PM (16:00)
    return "CLASS_HOURS" if "09:00" <= current_time_str <= "16:00" else "OFF_HOURS"

# ---------------------------------------------------------------------------
# WEB APPLICATION CONTROLLER ROUTES
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
    # Make sure background applications like MS Teams/Zoom are completely closed!
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return render_template("login.html", error="Webcam access blocked. Make sure MS Teams/Zoom are completely closed from your background tray icons.")
        
    time.sleep(0.8) # Let camera light adjustments settle
    ret, frame = cam.read()
    cam.release()

    if not ret or frame is None:
        return render_template("login.html", error="Webcam frame capture timed out. Please try again.")

    result = verify_face_match(frame)

    if result in ["VERIFIED_ADMIN", "GENERIC_MATCH"]:
        session["is_admin"] = True
        log_event(f"Admin verified and logged in via Face Verification system ({result}).")
        return redirect(url_for("index"))
    elif result == "NO_FACE":
        return render_template("login.html", error="No face detected. Look straight into your webcam in a well-lit room.")
    else:
        return render_template("login.html", error="Access Denied! Face patterns do not match your authorized profile photo.")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/")
def index():
    if not session.get("is_admin"):
        return redirect(url_for("login_page"))
    return render_template("index.html")

# ---------------------------------------------------------------------------
# REAL-TIME MOTION STREAM HOOKS
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
            current_time = time.time()
            if current_time - last_log_time > 5: # 5-second anti-spam block
                time_status = check_time_status()

                if time_status == "CLASS_HOURS":
                    log_event(f"Camera {camera_id}: Routine movement logged during standard class periods.", snapshot_name="NONE")
                else:
                    img_filename = f"cam{camera_id}_alert_{int(current_time)}.jpg"
                    cv2.imwrite(f"static/{img_filename}", frame)
                    log_event(f"!!! SECURITY ALERT !!! Unexpected off-hours activity logged in room space.", img_filename)

                last_log_time = current_time

        # Visual Dashboard HUD Render Layout
        h_f, w_f = frame.shape[:2]
        if motion_detected:
            cv2.rectangle(frame, (0, 0), (190, 32), (0, 0, 255), -1)
            cv2.putText(frame, "MOVEMENT ALERT", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (110, 26), (40, 40, 40), -1)
            cv2.putText(frame, "SYSTEM IDLE", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM {camera_id} | {ts}", (10, h_f - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

        ret2, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        for line in lines[::-1]:
            if "|" in line:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    parsed_logs.append({
                        "time": parts[0], "msg": parts[1], "snapshot": None if parts[2] == "NONE" else parts[2]
                    })
    return render_template("logs.html", log_list=parsed_logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)