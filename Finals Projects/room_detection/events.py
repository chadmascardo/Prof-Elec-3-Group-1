"""Events: what happened, what it means, and where it's recorded.

SRP — classification policy (EventClassifier) is separate from persistence
      (EventLog implementations).
OCP/DIP — CsvEventLog can be replaced by a database or REST log by
      implementing EventLog; RoomMonitor only sees the abstraction.
"""

import csv
import datetime as dt
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class DetectionEvent:
    """One logged occurrence (motion, anomaly, camera failure, ...)."""
    timestamp: dt.datetime
    room: str                       # display label, e.g. "Lab 301 (cam 2)"
    status: str                     # scheduled | anomaly | motion_no_user | ...
    course: Optional[str] = None
    snapshot_path: Optional[str] = None
    faces: Optional[int] = None     # detected user count (None = didn't run)
    names: str = ""                 # recognised people, comma-separated


class EventClassifier:
    """Decides what a motion event means (pure policy, no IO — easy to test).

    Rule (confirm a *user* before alerting):
      - scheduled                        -> ("scheduled",      alert=False)
      - unscheduled + a face detected    -> ("anomaly",        alert=True)
      - unscheduled + no face detected   -> ("motion_no_user", alert=False)
      - detector off (face_count None)   -> motion counts as a user (motion-only)
    """

    def classify(self, scheduled: bool, face_count: Optional[int]) -> Tuple[str, bool]:
        users_present = True if face_count is None else face_count > 0
        if scheduled:
            return "scheduled", False
        if users_present:
            return "anomaly", True
        return "motion_no_user", False


class EventLog(ABC):
    """Where events are recorded (ISP: write-only, one method)."""

    @abstractmethod
    def record(self, event: DetectionEvent) -> None: ...


class CsvEventLog(EventLog):
    """Appends events to a CSV file, creating headers on first write.
    A lock keeps rows intact when several camera threads log at once."""

    HEADERS = ["timestamp", "room", "status", "course", "snapshot", "faces", "names"]

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def record(self, event: DetectionEvent) -> None:
        with self._lock:
            file_exists = os.path.exists(self._path)
            with open(self._path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.HEADERS)
                writer.writerow([
                    event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    event.room,
                    event.status,
                    event.course or "",
                    event.snapshot_path or "",
                    "" if event.faces is None else event.faces,
                    event.names or "",
                ])
