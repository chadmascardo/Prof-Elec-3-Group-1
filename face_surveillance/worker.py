"""Per-camera monitoring thread.

Single responsibility: run the capture loop and occupancy state machine for
one camera. Dependency Inversion: receives only abstractions (MotionDetector,
FaceDetector, FaceRecognizer, ScheduleProvider, EventSink, SnapshotStore) —
it has no idea SQLite, MOG2, or YuNet exist.

Events produced:
  monitoring_started / monitoring_stopped
  camera_offline / camera_online
  motion_start   - area became occupied
  identified     - a known person was recognized (once per visit)
  snapshot       - image saved (only when no class is scheduled)
  motion_end     - area vacant again (after vacancy_timeout of no motion)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, Sequence, Set

import cv2
import numpy as np

from config import CameraConfig
from interfaces import (Event, EventSink, FaceDetector, FaceObservation,
                        FaceRecognizer, MotionDetector, ScheduleProvider,
                        SnapshotStore)


def annotate(frame: np.ndarray, faces: Sequence[FaceObservation],
             camera: str, when: datetime, course: Optional[str],
             motion: bool) -> np.ndarray:
    for f in faces:
        x, y, w, h = f.box
        known = f.name != "Unknown"
        color = (0, 200, 0) if known else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{f.name} {f.match_score:.2f}" if known else "Unknown"
        cv2.putText(frame, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    status = course if course else "NO CLASS"
    header = f"{camera}  {when:%Y-%m-%d %H:%M:%S}  [{status}]"
    if motion:
        header += "  MOTION"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(frame, header, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraConfig, motion: MotionDetector,
                 faces: FaceDetector, recognizer: FaceRecognizer,
                 schedule: ScheduleProvider, events: EventSink,
                 snapshots: SnapshotStore, *, process_width: int = 640,
                 vacancy_timeout: float = 10.0, face_interval: float = 0.5):
        super().__init__(name=f"cam-{camera.name}", daemon=True)
        self._cam = camera
        self._motion = motion
        self._faces = faces
        self._recognizer = recognizer
        self._schedule = schedule
        self._events = events
        self._snapshots = snapshots
        self._process_width = process_width
        self._vacancy_timeout = vacancy_timeout
        self._face_interval = face_interval
        self._stop = threading.Event()
        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()

    # ---- API for main thread -------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    # ---- helpers ---------------------------------------------------------
    def _emit(self, kind: str, *, names: Set[str] = frozenset(),
              course: Optional[str] = None, snapshot: Optional[str] = None,
              detail: str = "") -> None:
        self._events.write(Event(
            ts=datetime.now(), camera=self._cam.name, kind=kind,
            names=tuple(sorted(names)), course=course,
            snapshot=snapshot, detail=detail))

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self._process_width:
            return frame
        scale = self._process_width / w
        return cv2.resize(frame, (self._process_width, int(h * scale)))

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._cam.source)

    # ---- main loop -------------------------------------------------------
    def run(self) -> None:
        self._emit("monitoring_started")
        cap: Optional[cv2.VideoCapture] = None
        online = True
        read_failures = 0
        occupied = False
        episode_names: Set[str] = set()
        last_motion_t = 0.0
        last_face_t = 0.0

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                cap = self._open_capture()
                if not cap.isOpened():
                    if online:
                        self._emit("camera_offline",
                                   detail=f"source={self._cam.source}")
                        online = False
                    cap.release()
                    cap = None
                    self._stop.wait(5.0)
                    continue
                if not online:
                    self._emit("camera_online")
                    online = True

            ok, frame = cap.read()
            if not ok:
                read_failures += 1
                if read_failures > 25:
                    cap.release()
                    cap = None
                    read_failures = 0
                time.sleep(0.05)
                continue
            read_failures = 0

            frame = self._resize(frame)
            now = datetime.now()
            mono = time.monotonic()
            course = self._schedule.current_course(self._cam.name, now)
            in_motion = self._motion.detect(frame)
            observations: Sequence[FaceObservation] = ()

            if in_motion:
                last_motion_t = mono
                if not occupied:
                    occupied = True
                    episode_names = set()
                    self._emit("motion_start", course=course,
                               detail="area occupied")
                if mono - last_face_t >= self._face_interval:
                    last_face_t = mono
                    observations = [self._recognizer.identify(frame, f)
                                    for f in self._faces.detect(frame)]
                    for obs in observations:
                        if obs.name != "Unknown" and obs.name not in episode_names:
                            episode_names.add(obs.name)
                            self._emit("identified", names={obs.name},
                                       course=course,
                                       detail=f"score={obs.match_score:.2f}")
            elif occupied and mono - last_motion_t > self._vacancy_timeout:
                occupied = False
                self._emit("motion_end", names=episode_names,
                           course=course, detail="area vacant")

            annotated = annotate(frame, observations, self._cam.name,
                                 now, course, in_motion)

            # snapshot only when there is a detection OUTSIDE class hours
            if in_motion and course is None:
                path = self._snapshots.save(self._cam.name, annotated,
                                            now, "motion")
                if path:
                    self._emit("snapshot", names=episode_names,
                               snapshot=path,
                               detail=f"{len(observations)} face(s) visible")

            with self._latest_lock:
                self._latest = annotated
            time.sleep(0.01)

        if cap is not None:
            cap.release()
        if occupied:
            self._emit("motion_end", names=episode_names,
                       detail="monitor stopped while occupied")
        self._emit("monitoring_stopped")
