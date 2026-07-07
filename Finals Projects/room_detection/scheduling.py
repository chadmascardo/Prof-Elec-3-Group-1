"""Schedules: where they come from, and whether one is active right now.

SRP  — parsing the CSV (CsvScheduleSource) is separate from the time-window
       policy (ScheduleChecker).
OCP/DIP — ScheduleChecker depends on the ScheduleSource abstraction, so the
       CSV can be swapped for a database or API source without touching it.
"""

import csv
import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ScheduleEntry:
    """One scheduled slot for a room."""
    day: str           # e.g. "Tuesday"
    start: dt.time
    end: dt.time
    course: str


class ScheduleSource(ABC):
    """Abstraction over wherever schedules live (ISP: schedule reads only)."""

    @abstractmethod
    def entries_for(self, room: str) -> Sequence[ScheduleEntry]:
        """All scheduled slots for a room (possibly empty)."""


class CsvScheduleSource(ScheduleSource):
    """Reads the combined schedule + cameras CSV used by the original script:

        room,day,start,end,course,camera1,camera2
        Lab 301,Tuesday,08:00,10:00,CS101,http://...:8080/video,http://...:8080/video
        Lab 302,,,,,http://...:8080/video,          <- camera-only row (no class)

    Any column whose name starts with "camera" is treated as a feed; cameras
    are collected across all of a room's rows and de-duplicated. Rows without
    a day contribute cameras but no schedule entry.
    """

    def __init__(self, path: str):
        self._entries: Dict[str, List[ScheduleEntry]] = {}
        self._cameras: Dict[str, List[str]] = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                room = (row.get("room") or "").strip()
                if not room:
                    continue
                for key, val in row.items():
                    if key and key.startswith("camera") and val and val.strip():
                        feeds = self._cameras.setdefault(room, [])
                        if val.strip() not in feeds:
                            feeds.append(val.strip())
                if (row.get("day") or "").strip():
                    self._entries.setdefault(room, []).append(ScheduleEntry(
                        day=row["day"].strip(),
                        start=dt.datetime.strptime(row["start"].strip(), "%H:%M").time(),
                        end=dt.datetime.strptime(row["end"].strip(), "%H:%M").time(),
                        course=(row.get("course") or "").strip(),
                    ))

    def entries_for(self, room: str) -> Sequence[ScheduleEntry]:
        return self._entries.get(room, [])

    @property
    def cameras(self) -> Dict[str, List[str]]:
        """room -> list of camera sources (used by the composition root)."""
        return self._cameras


class ScheduleChecker:
    """Decides if a room has an active slot *right now* (incl. grace period)."""

    def __init__(self, source: ScheduleSource, grace_minutes: int):
        self._source = source
        self._grace = dt.timedelta(minutes=grace_minutes)

    def current_entry(self, room: str,
                      now: Optional[dt.datetime] = None) -> Optional[ScheduleEntry]:
        """The active ScheduleEntry for `room`, or None if nothing is scheduled."""
        now = now or dt.datetime.now()
        today = now.strftime("%A")
        for entry in self._source.entries_for(room):
            if entry.day != today:
                continue
            start = dt.datetime.combine(now.date(), entry.start) - self._grace
            end = dt.datetime.combine(now.date(), entry.end) + self._grace
            if start <= now <= end:
                return entry
        return None
