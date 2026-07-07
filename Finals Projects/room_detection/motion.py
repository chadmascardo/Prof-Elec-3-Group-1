"""Motion detection (SRP: pixels in, "did something move?" out).

The detector owns its previous-frame state, so the monitor loop doesn't have
to shuttle prev_gray around. OCP: a background-subtraction (MOG2) detector
could be added as another MotionDetector subclass without touching callers.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class MotionDetector(ABC):
    """One camera's motion detector. Stateful — create one per camera feed."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> bool:
        """True if motion is detected between this frame and the previous one."""


class FrameDiffMotionDetector(MotionDetector):
    """Frame-differencing detector (same algorithm as the original script)."""

    def __init__(self, area_threshold: int, process_size: tuple = (640, 360)):
        self._area_threshold = area_threshold
        self._process_size = process_size
        self._prev_gray: np.ndarray = None

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Downscale before processing — motion detection doesn't need full
        # resolution, and this alone cuts a lot of per-frame lag.
        small = cv2.resize(frame_bgr, self._process_size)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def reset(self, frame_bgr: np.ndarray) -> None:
        """Re-prime the reference frame (used after a camera reconnect)."""
        self._prev_gray = self._to_gray(frame_bgr)

    def detect(self, frame_bgr: np.ndarray) -> bool:
        gray = self._to_gray(frame_bgr)
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > self._area_threshold for c in contours)
