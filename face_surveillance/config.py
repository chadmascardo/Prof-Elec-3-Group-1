"""Camera configuration loading (single responsibility: parse cameras.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

Source = Union[int, str]  # webcam index, HTTP/MJPEG URL, or rtsp:// URL


@dataclass(frozen=True)
class CameraConfig:
    name: str       # also the room name used in schedule.csv
    source: Source


def load_cameras(path: Path) -> List[CameraConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = [CameraConfig(name=c["name"], source=c["source"])
               for c in data.get("cameras", [])]
    if not cameras:
        raise ValueError(f"no cameras defined in {path}")
    return cameras
