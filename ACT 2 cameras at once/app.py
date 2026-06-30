"""
Dual Camera Web Dashboard - Face Detection (Laptop + Phone)
-----------------------------------------------------------------
A Flask web server that shows TWO live camera feeds side by side in a
browser - your laptop's built-in webcam AND your phone's IP camera
stream - each running face detection independently using OpenCV's
DNN deep CNN face detector (avoids the MediaPipe solutions bug).

SETUP
1. Start your phone's camera streaming app (IP Webcam / SimpleIPCamera)
   and note its stream URL.

2. Install dependencies:
   pip install flask opencv-python

3. Edit PHONE_STREAM_URL and LAPTOP_CAM_INDEX below if needed.
   (LAPTOP_CAM_INDEX is usually 0 for a built-in webcam.)

4. Run:
   python app.py

   First run auto-downloads the DNN model files (deploy.prototxt and
   the .caffemodel) into this folder.

5. On the SAME laptop, open a browser and go to:
   http://127.0.0.1:5000
   You should see both camera feeds, each drawing a green box around
   any detected face.

   Other devices on the same WiFi can also view it by visiting:
   http://<your-laptop-ip>:5000
"""

import os
import urllib.request
import cv2
import numpy as np
import time
from flask import Flask, Response, render_template_string

# ----------------------------------------------------------------------
# CONFIG
LAPTOP_CAM_INDEX = 0
PHONE_STREAM_URL = "http://192.168.68.103:8080/video"

CONFIDENCE_THRESHOLD = 0.5

# How many CONSECUTIVE frames with no face must pass before we actually
# clear the display. This stops fast cameras (like a laptop webcam)
# from flickering the label on/off due to single borderline frames.
MISS_FRAMES_TO_CLEAR = 8
# ----------------------------------------------------------------------

PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def download_if_missing(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Done: {path}")


download_if_missing(PROTOTXT_PATH, PROTOTXT_URL)
download_if_missing(MODEL_PATH, MODEL_URL)

app = Flask(__name__)


class FaceCamera:
    """
    Wraps one video source (laptop webcam or phone stream) and keeps
    its own DNN face detector instance, so the two cameras don't
    interfere with each other when Flask handles them in parallel.
    """

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        # Each camera gets its own net instance - safer than sharing
        # one net across two threads at the same time.
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        # Smoothing state - remembers the last known face boxes so a
        # single missed frame doesn't instantly clear the display.
        self.last_boxes = []
        self.miss_streak = 0

    def get_frame(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
            time.sleep(0.5)
            return None

        ret, frame = self.cap.read()
        if not ret:
            self.cap = cv2.VideoCapture(self.source)
            return None

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        current_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                current_boxes.append((x1, y1, x2, y2, confidence))

        if current_boxes:
            # Faces found this frame - update memory and reset miss streak
            self.last_boxes = current_boxes
            self.miss_streak = 0
        else:
            # No faces found this frame - only clear after several
            # consecutive misses, to avoid flicker on borderline frames
            self.miss_streak += 1
            if self.miss_streak > MISS_FRAMES_TO_CLEAR:
                self.last_boxes = []

        for (x1, y1, x2, y2, confidence) in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{confidence * 100:.1f}%"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        face_count = len(self.last_boxes)
        status = f"{face_count} face(s)" if face_count else "No face detected"
        color = (0, 255, 0) if face_count else (255, 255, 255)
        cv2.putText(frame, f"{self.name}: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame


laptop_camera = FaceCamera(LAPTOP_CAM_INDEX, "Laptop")
phone_camera = FaceCamera(PHONE_STREAM_URL, "Phone")


def generate_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/laptop_feed')
def laptop_feed():
    return Response(generate_frames(laptop_camera),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/phone_feed')
def phone_feed():
    return Response(generate_frames(phone_camera),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dual Camera Face Detection Dashboard</title>
    <style>
        body {
            background: #1e1e1e;
            color: #f0f0f0;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1 {
            margin-top: 20px;
        }
        .camera-grid {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .camera-box {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 10px;
        }
        .camera-box h2 {
            margin: 0 0 10px 0;
            font-size: 18px;
        }
        img {
            width: 480px;
            max-width: 90vw;
            border-radius: 6px;
            display: block;
        }
    </style>
</head>
<body>
    <h1>Face Detection Dashboard</h1>
    <div class="camera-grid">
        <div class="camera-box">
            <h2>Laptop Camera</h2>
            <img src="{{ url_for('laptop_feed') }}">
        </div>
        <div class="camera-box">
            <h2>Phone Camera</h2>
            <img src="{{ url_for('phone_feed') }}">
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)