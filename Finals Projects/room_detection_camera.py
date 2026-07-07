"""
Room Detection Camera — Anomaly / Schedule-Based Motion Detection
====================================================================

What this does
---------------
- Watches one or more camera feeds for motion using frame differencing.
- Supports IP Webcam streams, so you can use phones/laptops as cameras
  instead of needing a USB webcam (see "Using IP Webcam" below).
- Checks each room's schedule to see if a class/activity is currently happening.
    - If motion happens DURING a scheduled activity  -> log it quietly, no alert.
    - If motion happens OUTSIDE a scheduled activity  -> treat it as an anomaly,
      save a snapshot, and notify the admin.
- Uses face detection to confirm a real person ("user"): OUTSIDE a scheduled
  activity, motion only becomes an anomaly (snapshot + alert) when a face is
  detected. Motion with no person is logged as "motion_no_user" without
  alerting, which cuts false alarms from pets, curtains, or light changes.
- Recognises WHO is present: each detected face is matched against the images
  in known_faces/ and labelled with the person's name, or "Unknown". Names
  appear in the alert, the snapshot, and the log.
- Applies a grace period around schedule boundaries (early/late arrivals are OK).
- Applies a cooldown so one continuous motion event doesn't spam the admin.
- Logs every event (scheduled or anomalous) to a CSV file for review.
- Falls back to console "fail-safe" alerts if a camera disconnects.

Requirements
------------
    pip install opencv-python face_detection facenet-pytorch --break-system-packages
    (face_detection + facenet-pytorch both run on torch + torchvision. Each
     model's weights download automatically on the first run and are cached
     afterwards, so the very first run needs internet access.)

Known faces (who is it)
-----------------------
Put reference photos in a known_faces/ folder next to this script, either:
    known_faces/Alice/photo1.jpg, known_faces/Alice/photo2.jpg   (folder per person)
    known_faces/Bob.jpg                                          (one image per person)
Both layouts work and can be mixed. Detected faces that don't match anyone in
that folder are labelled "Unknown". If the folder is missing or empty, everyone
is "Unknown" (detection/alerts still work).

Using IP Webcam (Android) to turn a phone into a camera
---------------------------------------------------------
1. Install the "IP Webcam" app (by Pavel Khlebovich) on the Android phone(s).
2. Connect the phone(s) AND the laptop running this script to the SAME Wi-Fi
   network (e.g. same router, or a phone hotspot the laptop joins).
3. Open the app, scroll down, tap "Start server". It will show an address
   like:  http://192.168.1.42:8080
4. The actual video stream URL to use is that address + "/video", e.g.:
       http://192.168.1.42:8080/video
   You can sanity-check it first by opening that address in a browser on
   the laptop — you should see a live preview page.
5. iPhone alternative: apps like "IP Camera Lite" work similarly, or you can
   just point `--source` at a laptop's built-in webcam (index 0) for one of
   the two devices instead of using a second phone.

Each room's cameras live in schedule.csv (the "camera1" and "camera2" columns),
so you don't pass camera URLs on the command line and there's no separate
cameras file. A room can have one or two cameras; both feeds are checked
against that room's schedule and are labelled "(cam 1)" / "(cam 2)" in the log,
snapshots, and preview windows.

Run (every room / camera in schedule.csv at once)
-------------------------------------------------
    python room_detection_camera.py

Run (a single room — starts one loop per camera it has)
-------------------------------------------------------
    python room_detection_camera.py --room "Lab 301"

    # --source runs just that one feed, overriding the CSV (handy for a quick
    # laptop-webcam or video-file test):
    python room_detection_camera.py --room "Lab 301" --source 0
    python room_detection_camera.py --room "Lab 301" --source video.mp4

Notes
-----
This is a working prototype meant as a starting point for a finals project,
not a production surveillance system. Swap `send_notification()` with real
SMS/email/push code, and `load_schedule()` with a real database/API call.
"""

import argparse
import csv
import datetime as dt
import os
import threading
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

GRACE_PERIOD_MINUTES = 15      # tolerance before/after a scheduled slot
ALERT_COOLDOWN_SECONDS = 120   # don't re-alert the same room more than once per 2 min
MOTION_AREA_THRESHOLD = 1500   # pixel area a contour must exceed to count as "motion"
SNAPSHOT_DIR = "snapshots"
LOG_FILE = "detection_log.csv"
SCHEDULE_FILE = "schedule.csv"

# Face detection — confirm a real person before treating motion as a user event.
# Available detectors (also printed at startup): "RetinaNetMobileNetV1" (fast,
# tiny weights), "RetinaNetResNet50" (more accurate, ~105 MB), "DSFDDetector"
# (heaviest). The chosen detector's weights auto-download on first use.
FACE_DETECTOR = "RetinaNetMobileNetV1"
FACE_CONFIDENCE = 0.5          # minimum score (0-1) to count a detection as a face
FACE_MAX_RESOLUTION = 720      # cap the longest side sent to the detector (speed)

# Face recognition (who is it) — matches detected faces to images in KNOWN_FACES_DIR.
KNOWN_FACES_DIR = "known_faces"       # folder-per-person or one-image-per-person
FACE_RECOGNITION_MODEL = "vggface2"   # facenet-pytorch InceptionResnetV1 weights
FACE_MATCH_THRESHOLD = 0.60           # min cosine similarity (0-1) to accept a match


# --------------------------------------------------------------------------
# Schedule handling
# --------------------------------------------------------------------------

def load_schedule(path=SCHEDULE_FILE):
    """
    Loads the room schedule AND each room's camera feeds from one CSV:

        room,day,start,end,course,camera1,camera2
        Lab 301,Tuesday,08:00,10:00,CS101,http://172.19.245.134:8080/video,http://172.19.245.99:8080/video
        Lab 302,,,,,http://172.19.241.230:8080/video,     # one camera, no class

    Returns (schedule, cameras):
      schedule = {room: [{day, start, end, course}, ...]}  # rows that have a day
      cameras  = {room: [source, ...]}                     # one or more feeds

    Any column whose name starts with "camera" (camera1, camera2, ...) is
    treated as a feed, so a room can have one or two (or more) cameras. Cameras
    are collected across all of a room's rows and de-duplicated. A row with no
    day is camera-only — e.g. a room with no classes today that you still want
    watched (motion there is always an anomaly).
    """
    schedule, cameras = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            room = (row.get("room") or "").strip()
            if not room:
                continue
            for key, val in row.items():
                if key and key.startswith("camera") and val and val.strip():
                    feeds = cameras.setdefault(room, [])
                    if val.strip() not in feeds:
                        feeds.append(val.strip())
            if (row.get("day") or "").strip():
                schedule.setdefault(room, []).append(row)
    return schedule, cameras


def is_schedule_active(schedule, room, now=None, grace_minutes=GRACE_PERIOD_MINUTES):
    """
    Returns (is_active, matching_entry) for whether `room` has a class
    happening right now, including the grace period.
    """
    now = now or dt.datetime.now()
    today = now.strftime("%A")
    grace = dt.timedelta(minutes=grace_minutes)

    for entry in schedule.get(room, []):
        if entry["day"] != today:
            continue
        start = dt.datetime.combine(now.date(), dt.datetime.strptime(entry["start"], "%H:%M").time()) - grace
        end = dt.datetime.combine(now.date(), dt.datetime.strptime(entry["end"], "%H:%M").time()) + grace
        if start <= now <= end:
            return True, entry
    return False, None


# --------------------------------------------------------------------------
# Notifications + logging
# --------------------------------------------------------------------------

def send_notification(room, now, frame=None, n_faces=None, names=""):
    """
    Stub for notifying the admin. Replace this with real SMS / email / push
    code (e.g. Twilio, SMTP, Firebase). Also saves a snapshot as evidence.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    snapshot_path = None
    if frame is not None:
        snapshot_path = os.path.join(
            SNAPSHOT_DIR, f"{room.replace(' ', '_')}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(snapshot_path, frame)

    who = "motion" if n_faces is None else f"{n_faces} person(s)"
    if names:
        who += f" [{names}]"
    message = f"[ALERT] Unscheduled {who} detected in {room} at {now.strftime('%Y-%m-%d %H:%M:%S')}"
    print(message)
    if snapshot_path:
        print(f"         snapshot saved: {snapshot_path}")
    # TODO: plug in real notification, e.g.:
    # send_email(to="admin@school.edu", subject="Room anomaly", body=message, attachment=snapshot_path)
    # send_sms(to="+639xxxxxxxxx", body=message)
    return snapshot_path


def log_event(room, now, status, course=None, snapshot_path=None, faces=None, names=""):
    """Appends a row to the CSV log. Creates the file with headers if needed.
    `faces` is the detected user count; `names` are the recognised people
    (both blank when detection/recognition didn't run)."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "room", "status", "course", "snapshot", "faces", "names"])
        writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), room, status, course or "",
                         snapshot_path or "", "" if faces is None else faces, names or ""])


# --------------------------------------------------------------------------
# Motion detection
# --------------------------------------------------------------------------

def detect_motion(prev_gray, gray, threshold=MOTION_AREA_THRESHOLD):
    """
    Simple frame-differencing motion detector.
    Returns True if a contour bigger than `threshold` pixels is found.
    """
    diff = cv2.absdiff(prev_gray, gray)
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) > threshold for c in contours)


# --------------------------------------------------------------------------
# Face detection (confirms a real person, not just motion)
# --------------------------------------------------------------------------

_face_detector = None
_face_detector_lock = threading.Lock()
_face_detector_off = False


def get_face_detector():
    """
    Lazily builds ONE shared face detector that every camera thread reuses
    (loading the model per-thread would be wasteful).

    Prints the available detector names on first load so you can see which
    model is in use, then loads FACE_DETECTOR. If the module is missing or the
    pretrained weights can't be downloaded, it prints why and returns None, and
    the system falls back to motion-only (motion alone is treated as a user).
    """
    global _face_detector, _face_detector_off
    if _face_detector is not None or _face_detector_off:
        return _face_detector
    with _face_detector_lock:
        if _face_detector is not None or _face_detector_off:
            return _face_detector
        try:
            import face_detection
            print(f"[face] available detectors: {', '.join(face_detection.available_detectors)}")
            print(f"[face] loading '{FACE_DETECTOR}' (confidence >= {FACE_CONFIDENCE}); "
                  f"first run downloads the model weights...")
            _face_detector = face_detection.build_detector(
                FACE_DETECTOR,
                confidence_threshold=FACE_CONFIDENCE,
                nms_iou_threshold=0.3,
                max_resolution=FACE_MAX_RESOLUTION,
            )
            print("[face] detector ready.")
        except Exception as e:
            _face_detector_off = True
            print(f"[face] Could not load face detector ({type(e).__name__}: {e}).")
            print("[face] Falling back to motion-only — motion alone will count as a user.")
    return _face_detector


def count_faces(frame_bgr):
    """
    Runs face detection on a BGR frame. Returns (n_faces, boxes) where boxes is
    an [N, 5] array of (xmin, ymin, xmax, ymax, score). Returns (None, None) if
    the detector is unavailable, so callers can fall back to motion-only.
    """
    detector = get_face_detector()
    if detector is None:
        return None, None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _face_detector_lock:            # the torch model is shared across threads
        boxes = detector.detect(rgb)
    return len(boxes), boxes


def draw_faces(frame_bgr, boxes, labels=None):
    """Draws detection boxes on a copy of the frame, labelled with the person's
    name when available (falls back to the detection score)."""
    out = frame_bgr.copy()
    if boxes is None:
        return out
    for i, (x0, y0, x1, y1, score) in enumerate(boxes):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        text = labels[i] if labels and i < len(labels) else f"{score:.2f}"
        cv2.putText(out, text, (int(x0), int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def classify_motion(active, n_faces):
    """
    Decides what a motion event means, given whether a class is scheduled and
    how many faces were detected (n_faces is None when the detector is off).

    Rule (confirm a *user* before alerting):
      - scheduled                       -> ("scheduled",       alert=False)
      - unscheduled + a face detected   -> ("anomaly",         alert=True)
      - unscheduled + no face detected  -> ("motion_no_user",  alert=False)
      - detector off (n_faces is None)  -> motion counts as a user (motion-only)
    """
    users_present = True if n_faces is None else n_faces > 0
    if active:
        return "scheduled", False
    if users_present:
        return "anomaly", True
    return "motion_no_user", False


# --------------------------------------------------------------------------
# Face recognition (who is it? — matches detected faces to known_faces/)
# --------------------------------------------------------------------------

_embedder = None
_embedder_lock = threading.Lock()
_embedder_off = False
_known_faces = None            # list of (name, embedding) built from KNOWN_FACES_DIR


def get_face_embedder():
    """
    Lazily loads the FaceNet embedding model (facenet-pytorch) once, shared by
    all threads. Returns the model, or None if unavailable — in which case
    every face is labelled "Unknown" and detection/alerts still work.
    """
    global _embedder, _embedder_off
    if _embedder is not None or _embedder_off:
        return _embedder
    with _embedder_lock:
        if _embedder is not None or _embedder_off:
            return _embedder
        try:
            from facenet_pytorch import InceptionResnetV1
            print(f"[face] loading recognition model (facenet-pytorch '{FACE_RECOGNITION_MODEL}'); "
                  f"first run downloads the weights...")
            _embedder = InceptionResnetV1(pretrained=FACE_RECOGNITION_MODEL).eval()
            print("[face] recognition model ready.")
        except Exception as e:
            _embedder_off = True
            print(f"[face] Recognition disabled ({type(e).__name__}: {e}). Faces will be 'Unknown'.")
    return _embedder


def embed_face(frame_bgr, box, margin=0.2):
    """Crops the face box (with margin), returns a 512-d L2-normalised embedding,
    or None if the embedder is unavailable / the crop is empty."""
    embedder = get_face_embedder()
    if embedder is None:
        return None
    import torch
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box[:4]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 127.5) / 128.0            # facenet fixed image standardization
    with _embedder_lock, torch.no_grad():
        emb = embedder(tensor)[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def list_known_face_images(directory=KNOWN_FACES_DIR):
    """Returns [(name, image_path), ...], supporting BOTH layouts (and a mix):
       known_faces/Alice/*.jpg   (folder per person)  -> name = folder name
       known_faces/Bob.jpg       (one image)          -> name = file stem
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    items = []
    if not os.path.isdir(directory):
        return items
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(exts):
                    items.append((entry, os.path.join(path, f)))
        elif entry.lower().endswith(exts):
            items.append((os.path.splitext(entry)[0], path))
    return items


def load_known_faces(directory=KNOWN_FACES_DIR):
    """Builds the known-face embedding DB once: detects the largest face in each
    reference image, embeds it, and stores (name, embedding)."""
    global _known_faces
    if _known_faces is not None:
        return _known_faces
    _known_faces = []
    images = list_known_face_images(directory)
    if not images:
        print(f"[face] No known faces in '{directory}/' — everyone will be 'Unknown'. "
              f"Add known_faces/<Name>/*.jpg or known_faces/<Name>.jpg to enable names.")
        return _known_faces
    for name, path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"[face] could not read {path}, skipping.")
            continue
        n, boxes = count_faces(img)
        if not n:
            print(f"[face] no face found in {path}, skipping.")
            continue
        largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        emb = embed_face(img, largest)
        if emb is not None:
            _known_faces.append((name, emb))
    people = sorted(set(n for n, _ in _known_faces))
    print(f"[face] loaded {len(_known_faces)} embedding(s) for {len(people)} people: "
          f"{', '.join(people) if people else '(none)'}")
    return _known_faces


def best_match(embedding, known, threshold=FACE_MATCH_THRESHOLD):
    """Returns the closest known name for an embedding, or 'Unknown'. Embeddings
    are L2-normalised, so the dot product is cosine similarity."""
    if embedding is None or not known:
        return "Unknown"
    best_name, best_sim = "Unknown", -1.0
    for name, kemb in known:
        sim = float(np.dot(embedding, kemb))
        if sim > best_sim:
            best_name, best_sim = name, sim
    return best_name if best_sim >= threshold else "Unknown"


def recognize_faces(frame_bgr, boxes):
    """Returns a list of names (or 'Unknown'), one per detected box."""
    known = load_known_faces()
    return [best_match(embed_face(frame_bgr, box), known) for box in boxes]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

class LatestFrameReader:
    """
    Fixes the classic IP-Webcam-over-WiFi lag problem.

    cv2.VideoCapture buffers frames internally. If your processing loop is
    even slightly slower than the incoming stream, frames queue up and you
    end up watching video that's seconds behind real time, with the delay
    growing over time.

    This class runs a background thread that constantly reads frames and
    only keeps the MOST RECENT one, throwing away anything older. The main
    loop always grabs the latest frame instead of working through a backlog.
    """

    def __init__(self, source):
        self.source = source
        self.cap = open_capture(source)
        # Ask OpenCV to keep as small a buffer as possible (driver-dependent,
        # not all backends honor this, which is exactly why we also do our
        # own "keep only the latest frame" logic below).
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            self.ok = ret
            self.frame = frame

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.2)
                continue
            ret, frame = self.cap.read()
            with self.lock:
                self.ok = ret
                if ret:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()


def open_capture(source):
    """Opens a cv2.VideoCapture, retrying a few times — useful for IP Webcam
    streams over WiFi which can be slow to respond on the first attempt."""
    for attempt in range(3):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(1)
    return cv2.VideoCapture(source)  # final attempt, caller checks isOpened()


def run(source, room, schedule_path=SCHEDULE_FILE, show_window=True, reconnect=True, sched_room=None):
    # `room` is the display label (e.g. "Lab 301 (cam 2)") used in logs, windows
    # and snapshots; `sched_room` is the real room whose schedule we check. They
    # differ only when one room has more than one camera.
    sched_room = sched_room or room
    schedule, _ = load_schedule(schedule_path)

    cap = LatestFrameReader(source)
    if not cap.isOpened():
        # Fail-safe: a camera that won't even open is itself worth alerting on.
        print(f"[FAIL-SAFE] Could not open camera/source '{source}' for {room}. Notifying admin.")
        log_event(room, dt.datetime.now(), status="camera_offline")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("[FAIL-SAFE] Camera opened but returned no frames.")
        return
    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (640, 360)), cv2.COLOR_BGR2GRAY)

    last_alert_time = 0.0
    print(f"[start] Monitoring '{room}' via {source} (press 'q' to quit if a window is shown)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # IP Webcam over WiFi can hiccup — try a few reconnects before
                # giving up and treating it as a real outage.
                print(f"[warn] Frame read failed for {room}, attempting reconnect...")
                cap.release()
                if reconnect:
                    cap = LatestFrameReader(source)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            prev_gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
                            continue
                print(f"[FAIL-SAFE] Lost connection to camera for {room}.")
                log_event(room, dt.datetime.now(), status="camera_offline")
                break

            # Downscale before processing — motion detection doesn't need full
            # resolution, and this alone cuts a lot of per-frame lag.
            frame_small = cv2.resize(frame, (640, 360))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            motion = detect_motion(prev_gray, gray)
            prev_gray = gray

            n_faces, boxes, names = None, None, None
            if motion:
                now = dt.datetime.now()
                n_faces, boxes = count_faces(frame)   # confirm a real person is present
                if n_faces:
                    names = recognize_faces(frame, boxes)   # who is it?
                active, entry = is_schedule_active(schedule, sched_room, now)
                status, is_anomaly = classify_motion(active, n_faces)
                who = ", ".join(names) if names else ""

                if not is_anomaly:
                    # Scheduled presence, or motion with no person: log quietly, no alert.
                    course = entry["course"] if active else None
                    log_event(room, now, status=status, course=course, faces=n_faces, names=who)
                elif time.time() - last_alert_time > ALERT_COOLDOWN_SECONDS:
                    # Unscheduled AND a person confirmed: alert with an annotated snapshot.
                    snapshot_path = send_notification(room, now, draw_faces(frame, boxes, names), n_faces, who)
                    log_event(room, now, status=status, snapshot_path=snapshot_path, faces=n_faces, names=who)
                    last_alert_time = time.time()
                else:
                    # Same anomaly still within the cooldown window: log, don't re-notify.
                    log_event(room, now, status="anomaly_suppressed_cooldown", faces=n_faces, names=who)

            if show_window:
                frame = draw_faces(frame, boxes, names)   # overlay detected faces + names
                users = "" if n_faces is None else f" | users: {n_faces}"
                label = "MOTION" if motion else "idle"
                cv2.putText(frame, f"{room} - {label}{users}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f"Room Detection Camera - {room}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyWindow(f"Room Detection Camera - {room}")


def run_multiple(cameras, schedule_path=SCHEDULE_FILE, show_window=True):
    """Runs one monitoring loop per camera, each in its own thread, so several
    devices — including two cameras in the same room — can be watched at once.

    `cameras` maps room -> list of sources, e.g.
        {"Lab 301": [url_a, url_b], "Lab 302": [url_c]}
    """
    threads = []
    for room, sources in cameras.items():
        for i, source in enumerate(sources, 1):
            src = int(source) if isinstance(source, str) and source.isdigit() else source
            label = f"{room} (cam {i})" if len(sources) > 1 else room
            t = threading.Thread(
                target=run, args=(src, label, schedule_path, show_window),
                kwargs={"sched_room": room}, daemon=True,
            )
            threads.append(t)
            t.start()
            time.sleep(0.5)  # stagger startup slightly

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[stop] Shutting down...")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Room Detection Camera - anomaly motion detection")
    parser.add_argument("--room", help="Room to monitor (must be a room in schedule.csv). Omit to watch every room in the CSV.")
    parser.add_argument("--source", help="Override the camera from schedule.csv: camera index (e.g. 0), IP Webcam URL, or video file path")
    parser.add_argument("--schedule", default=SCHEDULE_FILE, help="Path to schedule CSV file")
    parser.add_argument("--no-window", action="store_true", help="Run headless, no preview window")
    args = parser.parse_args()

    schedule, cameras = load_schedule(args.schedule)

    if args.room:
        # Single room: --source runs just that feed, else use its CSV camera(s).
        sources = [args.source] if args.source is not None else cameras.get(args.room, [])
        if not sources:
            parser.error(f"No camera for '{args.room}' in {args.schedule}. "
                         f"Add a camera value on one of its rows, or pass --source.")
        run_multiple({args.room: sources}, schedule_path=args.schedule, show_window=not args.no_window)
    elif cameras:
        # No room given: monitor every camera in the CSV.
        run_multiple(cameras, schedule_path=args.schedule, show_window=not args.no_window)
    else:
        parser.error(f"No cameras found in {args.schedule}. "
                     f"Add a camera column value for at least one room.")
