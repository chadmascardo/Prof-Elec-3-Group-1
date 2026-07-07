"""Composition root + CLI (the ONE place that knows about concrete classes).

DIP in practice: everything below the CLI is wired here — swap ConsoleNotifier
for an EmailNotifier, CsvScheduleSource for a database source, or add a camera
type, and this file is the only one that changes.
"""

import argparse
from typing import Dict, List

from .camera import LatestFrameReader
from .config import Config
from .detection import FaceDetectionPersonDetector
from .display import NullDisplay, WindowDisplay
from .events import CsvEventLog, EventClassifier
from .monitor import RoomMonitor, run_all
from .motion import FrameDiffMotionDetector
from .notifications import ConsoleNotifier, SnapshotStore
from .recognition import FacenetFaceRecognizer
from .scheduling import CsvScheduleSource, ScheduleChecker


def build_monitors(cameras: Dict[str, List[str]], schedule: CsvScheduleSource,
                   cfg: Config, show_window: bool) -> List[RoomMonitor]:
    """Wires one RoomMonitor per camera feed, sharing the heavyweight parts.

    Shared (thread-safe): schedule checker, person detector, recognizer, log,
    notifier, snapshot store, classifier, display.
    Per camera: the frame source and the (stateful) motion detector.
    """
    checker = ScheduleChecker(schedule, cfg.grace_period_minutes)
    person_detector = FaceDetectionPersonDetector(
        cfg.face_detector, cfg.face_confidence, cfg.face_max_resolution)
    recognizer = FacenetFaceRecognizer(
        person_detector, cfg.known_faces_dir,
        cfg.face_recognition_model, cfg.face_match_threshold)
    event_log = CsvEventLog(cfg.log_file)
    notifier = ConsoleNotifier()          # swap in EmailNotifier/CompositeNotifier here
    snapshots = SnapshotStore(cfg.snapshot_dir)
    classifier = EventClassifier()
    display = WindowDisplay() if show_window else NullDisplay()

    monitors = []
    for room, sources in cameras.items():
        for i, source in enumerate(sources, 1):
            src = int(source) if isinstance(source, str) and source.isdigit() else source
            label = f"{room} (cam {i})" if len(sources) > 1 else room
            monitors.append(RoomMonitor(
                label=label,
                sched_room=room,
                source_factory=lambda s=src: LatestFrameReader(s),
                motion_detector=FrameDiffMotionDetector(
                    cfg.motion_area_threshold,
                    (cfg.process_width, cfg.process_height)),
                person_detector=person_detector,
                recognizer=recognizer,
                schedule_checker=checker,
                classifier=classifier,
                event_log=event_log,
                notifier=notifier,
                snapshots=snapshots,
                display=display,
                alert_cooldown_seconds=cfg.alert_cooldown_seconds,
            ))
    return monitors


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser(
        description="Room Detection Camera - anomaly motion detection (SOLID refactor)")
    parser.add_argument("--room", help="Room to monitor (must be a room in the schedule CSV). "
                                       "Omit to watch every room in the CSV.")
    parser.add_argument("--source", help="Override the camera from the CSV: camera index "
                                         "(e.g. 0), IP Webcam URL, or video file path")
    parser.add_argument("--schedule", default=cfg.schedule_file, help="Path to schedule CSV file")
    parser.add_argument("--no-window", action="store_true", help="Run headless, no preview window")
    args = parser.parse_args()

    schedule = CsvScheduleSource(args.schedule)

    if args.room:
        # Single room: --source runs just that feed, else use its CSV camera(s).
        sources = [args.source] if args.source is not None else schedule.cameras.get(args.room, [])
        if not sources:
            parser.error(f"No camera for '{args.room}' in {args.schedule}. "
                         f"Add a camera value on one of its rows, or pass --source.")
        cameras = {args.room: sources}
    elif schedule.cameras:
        cameras = schedule.cameras   # no room given: monitor every camera in the CSV
    else:
        parser.error(f"No cameras found in {args.schedule}. "
                     f"Add a camera column value for at least one room.")

    run_all(build_monitors(cameras, schedule, cfg, show_window=not args.no_window))


if __name__ == "__main__":
    main()
