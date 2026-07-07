"""Composition root: wires concrete implementations into the abstractions.

This is the only module that knows every concrete class (Dependency
Inversion — dependencies point inward to interfaces.py).

Usage:
    python main.py            # headless
    python main.py --show     # with live annotated windows (q to quit)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from config import load_cameras
from events import (CompositeEventSink, ConsoleEventSink, MysqlEventSink,
                    SqliteEventSink)
from faces import SFaceRecognizer, YuNetFaceDetector
from motion import Mog2MotionDetector
from scheduling import CsvScheduleProvider
from snapshots import CooldownSnapshotStore, DiskSnapshotStore
from worker import CameraWorker

BASE = Path(__file__).resolve().parent

try:  # pull MYSQL_* variables from .env (optional dependency)
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

DETECT_MODEL = BASE / "models" / "face_detection_yunet_2023mar.onnx"
RECOG_MODEL = BASE / "models" / "face_recognition_sface_2021dec.onnx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face-detection room surveillance")
    p.add_argument("--cameras", type=Path, default=BASE / "cameras.json")
    p.add_argument("--schedule", type=Path, default=BASE / "schedule.csv")
    p.add_argument("--storage", choices=("mysql", "sqlite"), default="mysql",
                   help="where events are stored (default: mysql)")
    p.add_argument("--db", type=Path, default=BASE / "events.db",
                   help="sqlite file (only used with --storage sqlite)")
    env = os.environ.get  # precedence: CLI flag > .env / environment > default
    p.add_argument("--mysql-host", default=env("MYSQL_HOST", "127.0.0.1"),
                   help="overrides MYSQL_HOST from .env")
    p.add_argument("--mysql-port", type=int,
                   default=int(env("MYSQL_PORT", "3306")),
                   help="overrides MYSQL_PORT from .env")
    p.add_argument("--mysql-user", default=env("MYSQL_USER", "root"),
                   help="overrides MYSQL_USER from .env")
    p.add_argument("--mysql-password", default=env("MYSQL_PASSWORD", ""),
                   help="overrides MYSQL_PASSWORD from .env")
    p.add_argument("--mysql-database",
                   default=env("MYSQL_DATABASE", "face_surveillance"),
                   help="overrides MYSQL_DATABASE from .env")
    p.add_argument("--snapshots", type=Path, default=BASE / "snapshots")
    p.add_argument("--known", type=Path, default=BASE / "known_faces")
    p.add_argument("--show", action="store_true",
                   help="display annotated live windows")
    p.add_argument("--min-area", type=int, default=1500,
                   help="min moving-blob area in px to count as motion")
    p.add_argument("--vacancy", type=float, default=10.0,
                   help="seconds without motion before area is 'vacant'")
    p.add_argument("--cooldown", type=float, default=30.0,
                   help="min seconds between snapshots per camera")
    p.add_argument("--face-interval", type=float, default=0.5,
                   help="seconds between face-recognition passes")
    p.add_argument("--match-threshold", type=float, default=0.363,
                   help="SFace cosine threshold for a positive match")
    p.add_argument("--width", type=int, default=640,
                   help="processing width (frames are downscaled to this)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    for model in (DETECT_MODEL, RECOG_MODEL):
        if not model.exists():
            print(f"missing model: {model.name}\n"
                  f"run:  python download_models.py", file=sys.stderr)
            return 1

    schedule = CsvScheduleProvider(args.schedule)

    if args.storage == "mysql":
        try:
            store = MysqlEventSink(
                host=args.mysql_host, port=args.mysql_port,
                user=args.mysql_user, password=args.mysql_password,
                database=args.mysql_database)
        except Exception as exc:
            print(f"could not connect to MySQL: {exc}\n"
                  f"check that the server is running and the credentials "
                  f"in .env (MYSQL_USER / MYSQL_PASSWORD),\n"
                  f"or run with:  python main.py --storage sqlite",
                  file=sys.stderr)
            return 1
    else:
        store = SqliteEventSink(args.db)
    events = CompositeEventSink(ConsoleEventSink(), store)
    snapshots = CooldownSnapshotStore(DiskSnapshotStore(args.snapshots),
                                      cooldown_s=args.cooldown)

    workers = []
    for cam in load_cameras(args.cameras):
        detector = YuNetFaceDetector(DETECT_MODEL)
        recognizer = SFaceRecognizer(RECOG_MODEL, args.known, detector,
                                     threshold=args.match_threshold)
        workers.append(CameraWorker(
            cam,
            Mog2MotionDetector(min_area=args.min_area),
            detector, recognizer, schedule, events, snapshots,
            process_width=args.width,
            vacancy_timeout=args.vacancy,
            face_interval=args.face_interval,
        ))

    for w in workers:
        w.start()

    try:
        if args.show:
            while any(w.is_alive() for w in workers):
                for w in workers:
                    frame = w.latest_frame()
                    if frame is not None:
                        cv2.imshow(w.name, frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        else:
            while any(w.is_alive() for w in workers):
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=5.0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
