"""RoomMonitor — the high-level policy loop for one camera feed.

DIP — this is the point of the whole refactor: RoomMonitor depends ONLY on
abstractions (FrameSource via a factory, MotionDetector, PersonDetector,
FaceRecognizer, ScheduleChecker, EventClassifier, EventLog, Notifier,
SnapshotStore, Display). Every concrete class is injected by main.py, so this
loop never changes when a camera type, notifier channel, schedule backend, or
display mode does — it doesn't even know whether a window is shown (that's
NullDisplay's job, an LSP substitution).
"""

import datetime as dt
import threading
import time
from typing import Callable, List, Optional, Tuple

from .camera import FrameSource
from .detection import PersonDetection, PersonDetector
from .display import Display, annotate_status, draw_faces
from .events import DetectionEvent, EventClassifier, EventLog
from .motion import FrameDiffMotionDetector, MotionDetector
from .notifications import Notifier, SnapshotStore
from .recognition import FaceRecognizer
from .scheduling import ScheduleChecker


class RoomMonitor:
    """Watches one camera feed and applies the schedule/anomaly policy."""

    def __init__(self, *,
                 label: str,                 # display label, e.g. "Lab 301 (cam 2)"
                 sched_room: str,            # real room whose schedule is checked
                 source_factory: Callable[[], FrameSource],
                 motion_detector: MotionDetector,
                 person_detector: PersonDetector,
                 recognizer: FaceRecognizer,
                 schedule_checker: ScheduleChecker,
                 classifier: EventClassifier,
                 event_log: EventLog,
                 notifier: Notifier,
                 snapshots: SnapshotStore,
                 display: Display,
                 alert_cooldown_seconds: float):
        self._label = label
        self._sched_room = sched_room
        self._source_factory = source_factory
        self._motion = motion_detector
        self._persons = person_detector
        self._recognizer = recognizer
        self._schedule = schedule_checker
        self._classifier = classifier
        self._log = event_log
        self._notifier = notifier
        self._snapshots = snapshots
        self._display = display
        self._cooldown = alert_cooldown_seconds
        self._last_alert_time = 0.0

    # -- policy ------------------------------------------------------------

    def _log_offline(self) -> None:
        self._log.record(DetectionEvent(
            timestamp=dt.datetime.now(), room=self._label, status="camera_offline"))

    def _handle_motion(self, frame) -> Tuple[PersonDetection, List[str]]:
        """Classifies one motion event, logs it, and alerts if it's an anomaly.
        Returns the detection + names so the caller can draw the overlay."""
        now = dt.datetime.now()
        found = self._persons.detect(frame)                       # a real person?
        names: List[str] = []
        if found.count:
            names = self._recognizer.identify(frame, found.boxes)  # who is it?
        entry = self._schedule.current_entry(self._sched_room, now)
        status, is_anomaly = self._classifier.classify(entry is not None, found.count)
        who = ", ".join(names)

        if not is_anomaly:
            # Scheduled presence, or motion with no person: log quietly, no alert.
            self._log.record(DetectionEvent(
                timestamp=now, room=self._label, status=status,
                course=entry.course if entry else None,
                faces=found.count, names=who))
        elif time.time() - self._last_alert_time > self._cooldown:
            # Unscheduled AND a person confirmed: alert with an annotated snapshot.
            annotated = draw_faces(frame, found.boxes, names)
            snapshot_path = self._snapshots.save(self._label, now, annotated)
            event = DetectionEvent(
                timestamp=now, room=self._label, status=status,
                snapshot_path=snapshot_path, faces=found.count, names=who)
            self._notifier.notify(event)
            self._log.record(event)
            self._last_alert_time = time.time()
        else:
            # Same anomaly still within the cooldown window: log, don't re-notify.
            self._log.record(DetectionEvent(
                timestamp=now, room=self._label, status="anomaly_suppressed_cooldown",
                faces=found.count, names=who))
        return found, names

    # -- the loop ----------------------------------------------------------

    def run(self, reconnect: bool = True) -> None:
        cap = self._source_factory()
        if not cap.is_opened():
            # Fail-safe: a camera that won't even open is itself worth alerting on.
            print(f"[FAIL-SAFE] Could not open camera for {self._label}. Notifying admin.")
            self._log_offline()
            return

        ret, first = cap.read()
        if not ret or first is None:
            print("[FAIL-SAFE] Camera opened but returned no frames.")
            cap.release()
            return
        if isinstance(self._motion, FrameDiffMotionDetector):
            self._motion.reset(first)

        title = f"Room Detection Camera - {self._label}"
        print(f"[start] Monitoring '{self._label}' (press 'q' to quit if a window is shown)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # IP Webcam over WiFi can hiccup — try a reconnect before
                    # giving up and treating it as a real outage.
                    print(f"[warn] Frame read failed for {self._label}, attempting reconnect...")
                    cap.release()
                    if reconnect:
                        cap = self._source_factory()
                        if cap.is_opened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                if isinstance(self._motion, FrameDiffMotionDetector):
                                    self._motion.reset(frame)
                                continue
                    print(f"[FAIL-SAFE] Lost connection to camera for {self._label}.")
                    self._log_offline()
                    break

                motion = self._motion.detect(frame)
                found: Optional[PersonDetection] = None
                names: List[str] = []
                if motion:
                    found, names = self._handle_motion(frame)

                # Preview overlay — NullDisplay makes this a no-op when headless.
                frame = draw_faces(frame, found.boxes if found else None, names)
                users = "" if found is None or found.count is None else f" | users: {found.count}"
                annotate_status(frame, f"{self._label} - {'MOTION' if motion else 'idle'}{users}")
                if self._display.show(title, frame):
                    break
        finally:
            cap.release()
            self._display.close(title)


def run_all(monitors: List[RoomMonitor]) -> None:
    """One thread per monitor, so several cameras — including two in the same
    room — are watched at once."""
    threads = []
    for monitor in monitors:
        t = threading.Thread(target=monitor.run, daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.5)  # stagger startup slightly

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[stop] Shutting down...")
