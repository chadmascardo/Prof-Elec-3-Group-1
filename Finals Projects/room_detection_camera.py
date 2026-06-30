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
- Applies a grace period around schedule boundaries (early/late arrivals are OK).
- Applies a cooldown so one continuous motion event doesn't spam the admin.
- Logs every event (scheduled or anomalous) to a CSV file for review.
- Falls back to console "fail-safe" alerts if a camera disconnects.

Requirements
------------
    pip install opencv-python --break-system-packages

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

Run (single camera/room)
-------------------------
    python room_detection_camera.py --source http://192.168.1.42:8080/video --room "Lab 301"
    python room_detection_camera.py --source 0 --room "Lab 302"          # laptop webcam
    python room_detection_camera.py --source video.mp4 --room "Lab 301"  # test with a file

Run (two devices / two rooms at once)
---------------------------------------
    python room_detection_camera.py --no-window --cameras cameras.json

Where cameras.json looks like:
    {
      "Lab 301": "http://192.168.1.42:8080/video",
      "Lab 302": "http://192.168.1.55:8080/video"
    }

Notes
-----
This is a working prototype meant as a starting point for a finals project,
not a production surveillance system. Swap `send_notification()` with real
SMS/email/push code, and `load_schedule()` with a real database/API call.
"""

import argparse
import csv
import datetime as dt
import json
import os
import threading
import time

import cv2

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

GRACE_PERIOD_MINUTES = 15      # tolerance before/after a scheduled slot
ALERT_COOLDOWN_SECONDS = 120   # don't re-alert the same room more than once per 2 min
MOTION_AREA_THRESHOLD = 1500   # pixel area a contour must exceed to count as "motion"
SNAPSHOT_DIR = "snapshots"
LOG_FILE = "detection_log.csv"
SCHEDULE_FILE = "schedule.json"


# --------------------------------------------------------------------------
# Schedule handling
# --------------------------------------------------------------------------

def load_schedule(path=SCHEDULE_FILE):
    """
    Loads the room schedule. Expected JSON format:

    {
      "Lab 301": [
        {"day": "Monday",  "start": "08:00", "end": "10:00", "course": "CS101"},
        {"day": "Monday",  "start": "13:00", "end": "15:00", "course": "CS210"}
      ],
      "Lab 302": [...]
    }

    If the file doesn't exist yet, creates a small example so the script
    can run out of the box.
    """
    if not os.path.exists(path):
        # Example leaves 3:00 PM - 5:00 PM open (no classes), which is handy
        # for testing the anomaly/alert path without waiting for the grace
        # period of a real class to end.
        example = {
            "Lab 301": [
                {"day": dt.datetime.now().strftime("%A"), "start": "08:00", "end": "10:00", "course": "CS101"},
                {"day": dt.datetime.now().strftime("%A"), "start": "10:00", "end": "12:00", "course": "CS150"},
                {"day": dt.datetime.now().strftime("%A"), "start": "13:00", "end": "15:00", "course": "CS210"},
                # 15:00 (3:00 PM) to 17:00 (5:00 PM) intentionally left blank
                {"day": dt.datetime.now().strftime("%A"), "start": "17:00", "end": "19:00", "course": "CS220"},
            ]
        }
        with open(path, "w") as f:
            json.dump(example, f, indent=2)
        print(f"[setup] No schedule file found, created example {path}")
    with open(path) as f:
        return json.load(f)


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

def send_notification(room, now, frame=None):
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

    message = f"[ALERT] Unscheduled motion detected in {room} at {now.strftime('%Y-%m-%d %H:%M:%S')}"
    print(message)
    if snapshot_path:
        print(f"         snapshot saved: {snapshot_path}")
    # TODO: plug in real notification, e.g.:
    # send_email(to="admin@school.edu", subject="Room anomaly", body=message, attachment=snapshot_path)
    # send_sms(to="+639xxxxxxxxx", body=message)
    return snapshot_path


def log_event(room, now, status, course=None, snapshot_path=None):
    """Appends a row to the CSV log. Creates the file with headers if needed."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "room", "status", "course", "snapshot"])
        writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), room, status, course or "", snapshot_path or ""])


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


def run(source, room, schedule_path=SCHEDULE_FILE, show_window=True, reconnect=True):
    schedule = load_schedule(schedule_path)

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

            if motion:
                now = dt.datetime.now()
                active, entry = is_schedule_active(schedule, room, now)

                if active:
                    # Expected presence: log quietly, no alert.
                    log_event(room, now, status="scheduled", course=entry["course"])
                else:
                    # Anomaly: respect the cooldown so we don't spam the admin.
                    if time.time() - last_alert_time > ALERT_COOLDOWN_SECONDS:
                        snapshot_path = send_notification(room, now, frame)
                        log_event(room, now, status="anomaly", snapshot_path=snapshot_path)
                        last_alert_time = time.time()
                    else:
                        # Still log it, just don't re-notify yet.
                        log_event(room, now, status="anomaly_suppressed_cooldown")

            if show_window:
                label = "MOTION" if motion else "idle"
                cv2.putText(frame, f"{room} - {label}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(f"Room Detection Camera - {room}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyWindow(f"Room Detection Camera - {room}")


def run_multiple(cameras, schedule_path=SCHEDULE_FILE, show_window=True):
    """Runs one monitoring loop per room/camera, each in its own thread, so
    two devices (e.g. two phones, or a phone + a laptop) can be watched at
    the same time."""
    threads = []
    for room, source in cameras.items():
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        t = threading.Thread(target=run, args=(src, room, schedule_path, show_window), daemon=True)
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
    parser.add_argument("--source", help="Camera index (e.g. 0), IP Webcam URL, or video file path")
    parser.add_argument("--room", help="Room/area name, must match an entry in schedule.json")
    parser.add_argument("--cameras", help="Path to a JSON file mapping room name -> source, for multiple devices at once")
    parser.add_argument("--schedule", default=SCHEDULE_FILE, help="Path to schedule JSON file")
    parser.add_argument("--no-window", action="store_true", help="Run headless, no preview window")
    args = parser.parse_args()

    if args.cameras:
        # Multi-device mode: e.g. {"Lab 301": "http://192.168.1.42:8080/video", "Lab 302": 0}
        with open(args.cameras) as f:
            cameras = json.load(f)
        run_multiple(cameras, schedule_path=args.schedule, show_window=not args.no_window)
    elif args.source and args.room:
        # Single device mode
        source = int(args.source) if args.source.isdigit() else args.source
        run(source, args.room, schedule_path=args.schedule, show_window=not args.no_window)
    else:
        parser.error("Provide either (--source and --room) for one camera, or --cameras for multiple.")