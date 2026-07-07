"""
room_detection — SOLID refactor of room_detection_camera.py
============================================================

Same behaviour as the original script, restructured around the five SOLID
principles:

S — Single Responsibility: each module/class has one reason to change
    (scheduling, motion, person detection, recognition, logging, notifying,
    snapshots, display, camera IO, orchestration are all separate).
O — Open/Closed: new behaviour is added by writing a new subclass of an
    abstraction (e.g. an EmailNotifier, a DatabaseScheduleSource) and wiring
    it in main.py — no existing class needs to be modified.
L — Liskov Substitution: every implementation honours its base contract, so
    e.g. NullDisplay can replace WindowDisplay and NullFaceRecognizer can
    replace FacenetFaceRecognizer without RoomMonitor noticing.
I — Interface Segregation: many small ABCs (MotionDetector, PersonDetector,
    FaceRecognizer, Notifier, EventLog, ScheduleSource, FrameSource, Display)
    instead of one fat "Camera system" interface.
D — Dependency Inversion: RoomMonitor (high-level policy) depends only on
    the abstractions above; concrete classes are injected in main.py, the
    single composition root.

Run from the folder containing schedule.csv:

    python -m room_detection                  # every camera in schedule.csv
    python -m room_detection --room "Lab 301" # one room
    python -m room_detection --room "Lab 301" --source 0
"""

__all__ = ["main"]

from .main import main
