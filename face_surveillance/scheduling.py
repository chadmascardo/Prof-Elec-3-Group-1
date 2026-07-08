"""Class-schedule lookup from schedule.csv.

Single responsibility: turn (room, datetime) into "which course is in session".
Open/Closed: a JSON or database-backed provider can be added by implementing
ScheduleProvider without touching any consumer.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import List, Optional

from interfaces import ScheduleProvider


@dataclass(frozen=True)
class _Slot:
    room: str   # lowercase
    day: str    # lowercase full weekday name
    start: time
    end: time
    course: str


def _parse_time(value: str) -> time:
    return datetime.strptime(value.strip(), "%H:%M").time()


class CsvScheduleProvider(ScheduleProvider):
    """Weekly timetable: room,day,start,end,course (24h HH:MM times)."""

    def __init__(self, path: Path):
        self._slots: List[_Slot] = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                self._slots.append(_Slot(
                    room=row["room"].strip().lower(),
                    day=row["day"].strip().lower(),
                    start=_parse_time(row["start"]),
                    end=_parse_time(row["end"]),
                    course=row["course"].strip(),
                ))

    def current_course(self, room: str, when: datetime) -> Optional[str]:
        room = room.strip().lower()
        day = when.strftime("%A").lower()
        now = when.time()
        for s in self._slots:
            if s.room == room and s.day == day and s.start <= now < s.end:
                return s.course
        return None
