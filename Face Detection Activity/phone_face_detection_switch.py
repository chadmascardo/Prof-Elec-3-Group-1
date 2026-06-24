"""
Phone Stream Face Detection - Traditional vs Modern (Switchable)
-------------------------------------------------------------------
Connects to your phone's IP camera stream and lets you switch live
between two face detection methods:

  [1] Traditional - Haar Cascade (hand-crafted rules, fast but less accurate)
  [2] Modern       - Deep CNN via OpenCV DNN (SSD + ResNet-10, learned features)

SETUP
1. Start your phone's camera streaming app and note the stream URL.
2. Install dependency:
   pip install opencv-python
3. Edit PHONE_STREAM_URL below with your phone's stream URL.
4. Run:
   python phone_face_detection_switch.py

   First run auto-downloads the DNN model files (deploy.prototxt and
   the .caffemodel) into this folder. Haar Cascade's file comes
   bundled with opencv-python already, no download needed.

CONTROLS (while the video window is focused)
  1 = switch to Traditional (Haar Cascade)
  2 = switch to Modern (Deep CNN)
  q = quit
"""

import os
import urllib.request
import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG - change this to match your phone's stream address
PHONE_STREAM_URL = "http://172.18.16.212:8080/stream.mjpg"

CONFIDENCE_THRESHOLD = 0.5   # for the DNN (modern) method
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

MODE_TRADITIONAL = "traditional"
MODE_MODERN = "modern"


def download_if_missing(path, url):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Done: {path}")


def detect_traditional(frame, face_classifier):
    """Haar Cascade - hand-crafted edge/rectangle rules."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return len(faces)


def detect_modern(frame, net):
    """Deep CNN (SSD + ResNet-10) - learned features."""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{confidence * 100:.1f}%"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
    return face_count


def main():
    # --- Load Traditional method (Haar Cascade, bundled with opencv-python) ---
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # --- Load Modern method (Deep CNN via OpenCV DNN) ---
    download_if_missing(PROTOTXT_PATH, PROTOTXT_URL)
    download_if_missing(MODEL_PATH, MODEL_URL)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    cap = cv2.VideoCapture(PHONE_STREAM_URL)

    if not cap.isOpened():
        print(f"Could not open stream at {PHONE_STREAM_URL}")
        print("Check that the phone's camera app is running and the URL is correct.")
        return

    print(f"Connected to phone stream: {PHONE_STREAM_URL}")
    print("Press '1' = Traditional (Haar Cascade), '2' = Modern (Deep CNN), 'q' = quit")

    mode = MODE_MODERN  # start in Modern mode by default

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to stream, retrying...")
            continue

        if mode == MODE_TRADITIONAL:
            face_count = detect_traditional(frame, face_classifier)
            mode_label = "TRADITIONAL (Haar Cascade)"
            color = (0, 255, 255)
        else:
            face_count = detect_modern(frame, net)
            mode_label = "MODERN (Deep CNN)"
            color = (0, 255, 0)

        cv2.putText(
            frame, f"Mode: {mode_label}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        cv2.putText(
            frame, f"Faces: {face_count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        cv2.imshow("Phone Camera - Face Detection (Traditional vs Modern)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            mode = MODE_TRADITIONAL
            print("Switched to TRADITIONAL (Haar Cascade)")
        elif key == ord('2'):
            mode = MODE_MODERN
            print("Switched to MODERN (Deep CNN)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()