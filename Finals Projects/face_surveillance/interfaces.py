"""Core domain types and abstractions.

SOLID roles:
- Interface Segregation: each ABC covers exactly one capability, so a class
  never depends on methods it does not use.
- Dependency Inversion: high-level code (CameraWorker, main) depends on these
  abstractions, never on concrete OpenCV/SQLite/disk implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    """One detected face in a frame."""
    box: Tuple[int, int, int, int]      # x, y, w, h
    score: float                        # detector confidence
    raw: np.ndarray                     # full detector row (incl. landmarks)
    name: str = "Unknown"               # filled in by a FaceRecognizer
    match_score: float = 0.0            # recognition similarity


@dataclass(frozen=True)
class Event:
    """One loggable occurrence on a camera."""
    ts: datetime
    camera: str
    kind: str                           # monitoring_started, motion_start,
                                        # identified, snapshot, motion_end,
                                        # camera_offline, camera_online, ...
    names: Tuple[str, ...] = ()
    course: Optional[str] = None        # scheduled class at that moment, if any
    snapshot: Optional[str] = None      # path of saved image, if any
    detail: str = ""


class MotionDetector(ABC):
    """Decides whether a frame contains motion. Stateful per camera."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool: ...


class FaceDetector(ABC):
    """Finds faces in a frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Sequence[FaceObservation]: ...


class FaceRecognizer(ABC):
    """Assigns an identity to a detected face."""

    @abstractmethod
    def identify(self, frame: np.ndarray,
                 face: FaceObservation) -> FaceObservation: ...


class ScheduleProvider(ABC):
    """Answers: is there a class in this room right now?"""

    @abstractmethod
    def current_course(self, room: str, when: datetime) -> Optional[str]: ...


class EventSink(ABC):
    """Receives events (database, console, ...). Must be thread-safe."""

    @abstractmethod
    def write(self, event: Event) -> None: ...


class SnapshotStore(ABC):
    """Persists evidence images. Returns the saved path, or None if skipped."""

    @abstractmethod
    def save(self, camera: str, frame: np.ndarray,
             when: datetime, tag: str) -> Optional[str]: ...
