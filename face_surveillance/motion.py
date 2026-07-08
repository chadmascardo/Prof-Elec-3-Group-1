"""Motion detection via MOG2 background subtraction.

Single responsibility: frame in -> "is something moving" out.
Stateful, so each camera gets its own instance (created in main.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from interfaces import MotionDetector


class Mog2MotionDetector(MotionDetector):
    def __init__(self, min_area: int = 1500, warmup_frames: int = 30,
                 history: int = 300):
        self._sub = cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=True)
        self._min_area = min_area
        self._warmup = warmup_frames
        self._seen = 0

    def detect(self, frame: np.ndarray) -> bool:
        mask = self._sub.apply(frame)
        self._seen += 1
        if self._seen < self._warmup:       # let the background model settle
            return False
        # drop shadows (127) and noise, then look for large moving blobs
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) >= self._min_area for c in contours)
