"""One-time download of the ONNX models used for face detection/recognition.

Run:  python download_models.py
"""
import urllib.request
from pathlib import Path

MODELS = {
    # YuNet face detector (OpenCV cv2.FaceDetectorYN)
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # SFace face recognizer (OpenCV cv2.FaceRecognizerSF)
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

dest = Path(__file__).parent / "models"
dest.mkdir(exist_ok=True)

for name, url in MODELS.items():
    target = dest / name
    if target.exists() and target.stat().st_size > 100_000:
        print(f"already have {name}")
        continue
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(url, target)
    print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")

print("done")
