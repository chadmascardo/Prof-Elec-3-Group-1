"""Notifications and snapshot evidence.

SRP — saving evidence (SnapshotStore) and telling the admin (Notifier) are
      different jobs; the original send_notification() did both.
OCP — add EmailNotifier / SmsNotifier / PushNotifier as new subclasses and
      wire them in main.py; nothing here changes. CompositeNotifier lets
      several channels fire for one event.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Optional

import cv2
import numpy as np

from .events import DetectionEvent


class SnapshotStore:
    """Writes annotated frames to disk and returns the path (evidence)."""

    def __init__(self, directory: str):
        self._dir = directory

    def save(self, room: str, when: datetime, frame: Optional[np.ndarray]) -> Optional[str]:
        if frame is None:
            return None
        os.makedirs(self._dir, exist_ok=True)
        path = os.path.join(
            self._dir, f"{room.replace(' ', '_')}_{when.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(path, frame)
        return path


class Notifier(ABC):
    """Delivers an alert to the admin (ISP: one method)."""

    @abstractmethod
    def notify(self, event: DetectionEvent) -> None: ...


class ConsoleNotifier(Notifier):
    """Prints the alert — the prototype's stand-in for SMS/email/push."""

    def notify(self, event: DetectionEvent) -> None:
        who = "motion" if event.faces is None else f"{event.faces} person(s)"
        if event.names:
            who += f" [{event.names}]"
        print(f"[ALERT] Unscheduled {who} detected in {event.room} "
              f"at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if event.snapshot_path:
            print(f"         snapshot saved: {event.snapshot_path}")


class CompositeNotifier(Notifier):
    """Fans one alert out to several channels (console + email + SMS, ...)."""

    def __init__(self, notifiers: Iterable[Notifier]):
        self._notifiers = list(notifiers)

    def notify(self, event: DetectionEvent) -> None:
        for n in self._notifiers:
            n.notify(event)


# Example of extension without modification (OCP) — implement and wire in main.py:
#
# class EmailNotifier(Notifier):
#     def __init__(self, to_address: str): ...
#     def notify(self, event: DetectionEvent) -> None:
#         send_email(to=self._to, subject="Room anomaly", body=..., attachment=event.snapshot_path)
