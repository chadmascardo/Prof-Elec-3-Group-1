"""Configuration (SRP: the only module that changes when a tunable changes).

One frozen dataclass instead of scattered module-level constants. Everything
that needs a setting receives it through its constructor, so no class reads
global state.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Schedule / alert policy
    grace_period_minutes: int = 15       # tolerance before/after a scheduled slot
    alert_cooldown_seconds: float = 120  # min seconds between alerts per camera
    schedule_file: str = "schedule.csv"

    # Motion detection
    motion_area_threshold: int = 1500    # contour pixel area that counts as motion
    process_width: int = 640             # frames are downscaled to this for motion
    process_height: int = 360

    # Outputs
    snapshot_dir: str = "snapshots"
    log_file: str = "detection_log.csv"

    # Face detection (is a person present?)
    face_detector: str = "RetinaNetMobileNetV1"
    face_confidence: float = 0.5
    face_max_resolution: int = 720

    # Face recognition (who is it?)
    known_faces_dir: str = "known_faces"
    face_recognition_model: str = "vggface2"
    face_match_threshold: float = 0.60
