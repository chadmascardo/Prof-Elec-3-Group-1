"""Snapshot persistence.

DiskSnapshotStore writes snapshots/<camera>/<YYYY-MM-DD>/HHMMSS_tag.jpg.
CooldownSnapshotStore is a decorator (Open/Closed) that rate-limits any
SnapshotStore without modifying it.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from interfaces import SnapshotStore


class DiskSnapshotStore(SnapshotStore):
    def __init__(self, root: Path):
        self._root = Path(root)

    def save(self, camera: str, frame: np.ndarray,
             when: datetime, tag: str) -> Optional[str]:
        folder = self._root / camera.replace(" ", "_") / f"{when:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{when:%H%M%S}_{tag}.jpg"
        return str(path) if cv2.imwrite(str(path), frame) else None


class CooldownSnapshotStore(SnapshotStore):
    """Wraps another store; allows at most one snapshot per camera
    every `cooldown_s` seconds."""

    def __init__(self, inner: SnapshotStore, cooldown_s: float = 30.0):
        self._inner = inner
        self._cooldown = cooldown_s
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def save(self, camera: str, frame: np.ndarray,
             when: datetime, tag: str) -> Optional[str]:
        now = time.monotonic()
        with self._lock:
            last = self._last.get(camera, -1e12)
            if now - last < self._cooldown:
                return None
            self._last[camera] = now
        return self._inner.save(camera, frame, when, tag)
