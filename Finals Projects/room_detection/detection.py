"""Person (face) detection — confirms a real person, not just motion.

ISP — PersonDetector is deliberately tiny: one method. Recognition ("who")
      lives in recognition.py; drawing lives in display.py.
LSP — FaceDetectionPersonDetector degrades to the "unavailable" result if the
      model can't load, which callers already handle; any other subclass
      (e.g. a YOLO person detector) can be swapped in.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    """Result of a detection pass.

    count is None when the detector is unavailable — the classifier then falls
    back to motion-only mode (motion alone counts as a person). boxes is an
    [N, 5] array of (xmin, ymin, xmax, ymax, score) or None.
    """
    count: Optional[int]
    boxes: Optional[np.ndarray]

    @property
    def available(self) -> bool:
        return self.count is not None


UNAVAILABLE = PersonDetection(count=None, boxes=None)


class PersonDetector(ABC):
    """Is a person in this frame? (ISP: nothing about identity or drawing.)"""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> PersonDetection: ...


class FaceDetectionPersonDetector(PersonDetector):
    """Uses the `face_detection` package (RetinaFace family) to find faces.

    The torch model is expensive, so ONE instance is built lazily and shared
    by every camera thread — construction and inference are both locked. If
    the module is missing or the weights can't download, detection reports
    UNAVAILABLE and the system falls back to motion-only.
    """

    def __init__(self, detector_name: str, confidence: float, max_resolution: int):
        self._name = detector_name
        self._confidence = confidence
        self._max_resolution = max_resolution
        self._lock = threading.Lock()
        self._detector = None
        self._off = False

    def _get_detector(self):
        if self._detector is not None or self._off:
            return self._detector
        with self._lock:
            if self._detector is not None or self._off:
                return self._detector
            try:
                import face_detection
                print(f"[face] available detectors: {', '.join(face_detection.available_detectors)}")
                print(f"[face] loading '{self._name}' (confidence >= {self._confidence}); "
                      f"first run downloads the model weights...")
                self._detector = face_detection.build_detector(
                    self._name,
                    confidence_threshold=self._confidence,
                    nms_iou_threshold=0.3,
                    max_resolution=self._max_resolution,
                )
                print("[face] detector ready.")
            except Exception as e:
                self._off = True
                print(f"[face] Could not load face detector ({type(e).__name__}: {e}).")
                print("[face] Falling back to motion-only — motion alone will count as a user.")
        return self._detector

    def detect(self, frame_bgr: np.ndarray) -> PersonDetection:
        detector = self._get_detector()
        if detector is None:
            return UNAVAILABLE
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:  # the torch model is shared across camera threads
            boxes = detector.detect(rgb)
        return PersonDetection(count=len(boxes), boxes=boxes)
