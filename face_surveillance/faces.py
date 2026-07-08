"""Face detection (YuNet) and recognition (SFace), both via OpenCV DNN.

Liskov: both classes are drop-in implementations of their ABCs — a different
detector/recognizer (MediaPipe, dlib, ...) could replace them unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from interfaces import FaceDetector, FaceObservation, FaceRecognizer

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YuNetFaceDetector(FaceDetector):
    def __init__(self, model_path: Path, score_threshold: float = 0.7):
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, frame: np.ndarray) -> List[FaceObservation]:
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(frame)
        if rows is None:
            return []
        out = []
        for row in rows:
            x, y, bw, bh = (int(v) for v in row[:4])
            out.append(FaceObservation(
                box=(x, y, bw, bh), score=float(row[-1]), raw=row))
        return out


class SFaceRecognizer(FaceRecognizer):
    """Matches faces against enrollment photos in known_faces/<Name>/*.jpg."""

    def __init__(self, model_path: Path, known_dir: Path,
                 detector: FaceDetector, threshold: float = 0.363):
        self._rec = cv2.FaceRecognizerSF.create(str(model_path), "")
        self._threshold = threshold
        self._known: List[Tuple[str, np.ndarray]] = []
        self._enroll(Path(known_dir), detector)

    def _enroll(self, known_dir: Path, detector: FaceDetector) -> None:
        if not known_dir.is_dir():
            log.warning("known-faces folder %s missing - recognition will "
                        "label everyone Unknown", known_dir)
            return
        for person in sorted(p for p in known_dir.iterdir() if p.is_dir()):
            count = 0
            for img_path in sorted(person.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    log.warning("could not read %s", img_path)
                    continue
                faces = detector.detect(img)
                if not faces:
                    log.warning("no face found in %s", img_path)
                    continue
                best = max(faces, key=lambda f: f.score)
                self._known.append((person.name, self._feature(img, best)))
                count += 1
            if count:
                log.info("enrolled %s (%d photo(s))", person.name, count)
        if not self._known:
            log.warning("no usable enrollment photos in %s", known_dir)

    def _feature(self, frame: np.ndarray, face: FaceObservation) -> np.ndarray:
        aligned = self._rec.alignCrop(frame, face.raw)
        return self._rec.feature(aligned).copy()  # copy: OpenCV reuses buffer

    def identify(self, frame: np.ndarray,
                 face: FaceObservation) -> FaceObservation:
        if not self._known:
            return face
        feat = self._feature(frame, face)
        best_name, best_score = "Unknown", 0.0
        for name, known_feat in self._known:
            score = float(self._rec.match(
                feat, known_feat, cv2.FaceRecognizerSF_FR_COSINE))
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= self._threshold:
            return replace(face, name=best_name, match_score=best_score)
        return replace(face, match_score=best_score)
