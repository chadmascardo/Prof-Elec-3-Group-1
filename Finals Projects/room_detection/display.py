"""Presentation: drawing overlays and showing preview windows.

LSP — NullDisplay substitutes for WindowDisplay anywhere (headless mode),
      honouring the same contract; RoomMonitor never checks which one it has.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np


def draw_faces(frame_bgr: np.ndarray, boxes,
               labels: Optional[List[str]] = None) -> np.ndarray:
    """Detection boxes on a copy of the frame, labelled with the person's
    name when available (falls back to the detection score)."""
    if boxes is None:
        return frame_bgr
    out = frame_bgr.copy()
    for i, (x0, y0, x1, y1, score) in enumerate(boxes):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        text = labels[i] if labels and i < len(labels) else f"{score:.2f}"
        cv2.putText(out, text, (int(x0), int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def annotate_status(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    """Status line ("<room> - MOTION | users: 2") in the top-left corner."""
    cv2.putText(frame_bgr, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame_bgr


class Display(ABC):
    """Where (and whether) frames are shown (ISP)."""

    @abstractmethod
    def show(self, title: str, frame: np.ndarray) -> bool:
        """Shows the frame. Returns True if the user asked to quit."""

    @abstractmethod
    def close(self, title: str) -> None: ...


class WindowDisplay(Display):
    """OpenCV preview window; 'q' quits that camera's loop."""

    def show(self, title: str, frame: np.ndarray) -> bool:
        cv2.imshow(title, frame)
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self, title: str) -> None:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass  # window was never actually shown (e.g. camera died first)


class NullDisplay(Display):
    """Headless mode (--no-window): same contract, shows nothing."""

    def show(self, title: str, frame: np.ndarray) -> bool:
        return False

    def close(self, title: str) -> None:
        pass
