"""Camera IO (SRP: getting frames, nothing else).

FrameSource is the abstraction the monitor depends on (DIP); LatestFrameReader
is the concrete implementation that fixes IP-Webcam-over-WiFi lag. A future
PiCameraSource or RTSPSource just implements FrameSource (OCP).
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSource(ABC):
    """Minimal interface a camera must provide (ISP)."""

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """(ok, frame). frame is a BGR copy the caller may mutate."""

    @abstractmethod
    def is_opened(self) -> bool: ...

    @abstractmethod
    def release(self) -> None: ...


def open_capture(source) -> cv2.VideoCapture:
    """Opens a cv2.VideoCapture, retrying a few times — IP Webcam streams over
    WiFi can be slow to respond on the first attempt."""
    for _ in range(3):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
        time.sleep(1)
    return cv2.VideoCapture(source)  # final attempt, caller checks is_opened()


class LatestFrameReader(FrameSource):
    """Fixes the classic IP-Webcam-over-WiFi lag problem.

    cv2.VideoCapture buffers frames internally. If the processing loop is even
    slightly slower than the incoming stream, frames queue up and you end up
    watching video seconds behind real time. A background thread constantly
    reads and keeps only the MOST RECENT frame, so read() never sees a backlog.
    """

    def __init__(self, source):
        self._cap = open_capture(source)
        # Ask OpenCV for the smallest buffer it will give us (driver-dependent,
        # which is exactly why we also keep only the latest frame ourselves).
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ok = False
        self._stopped = False

        if self._cap.isOpened():
            ret, frame = self._cap.read()
            self._ok = ret
            self._frame = frame

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        while not self._stopped:
            if not self._cap.isOpened():
                time.sleep(0.2)
                continue
            ret, frame = self._cap.read()
            with self._lock:
                self._ok = ret
                if ret:
                    self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._ok, (None if self._frame is None else self._frame.copy())

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=1)
        self._cap.release()
