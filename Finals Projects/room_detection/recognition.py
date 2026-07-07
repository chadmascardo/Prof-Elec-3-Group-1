"""Face recognition — WHO is present, matched against known_faces/.

SRP — identity is a separate concern from detection; RoomMonitor works fine
      with either recognizer.
LSP — NullFaceRecognizer is a drop-in stand-in when recognition isn't wanted
      or facenet-pytorch isn't installed: same contract, everyone "Unknown".
DIP — FacenetFaceRecognizer receives a PersonDetector abstraction to find
      faces in the reference photos; it doesn't build its own.
"""

import os
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detection import PersonDetector


class FaceRecognizer(ABC):
    """Names for detected faces (ISP: identity only)."""

    @abstractmethod
    def identify(self, frame_bgr: np.ndarray, boxes) -> List[str]:
        """One name (or 'Unknown') per detection box."""


class NullFaceRecognizer(FaceRecognizer):
    """Recognition switched off — everyone is 'Unknown' (LSP substitute)."""

    def identify(self, frame_bgr: np.ndarray, boxes) -> List[str]:
        return ["Unknown" for _ in boxes]


def list_known_face_images(directory: str) -> List[Tuple[str, str]]:
    """[(name, image_path), ...] supporting BOTH layouts (and a mix):
       known_faces/Alice/*.jpg   (folder per person) -> name = folder name
       known_faces/Bob.jpg       (one image)         -> name = file stem
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    items = []
    if not os.path.isdir(directory):
        return items
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                if f.lower().endswith(exts):
                    items.append((entry, os.path.join(path, f)))
        elif entry.lower().endswith(exts):
            items.append((os.path.splitext(entry)[0], path))
    return items


class FacenetFaceRecognizer(FaceRecognizer):
    """FaceNet (facenet-pytorch) embeddings + cosine similarity.

    The embedding model and the known-face index are built lazily, once,
    shared across all camera threads (locked). If facenet-pytorch is missing
    or weights can't download, it behaves like NullFaceRecognizer.
    """

    def __init__(self, face_finder: PersonDetector, known_faces_dir: str,
                 model_name: str = "vggface2", match_threshold: float = 0.60):
        self._face_finder = face_finder
        self._dir = known_faces_dir
        self._model_name = model_name
        self._threshold = match_threshold
        self._lock = threading.Lock()
        self._embedder = None
        self._off = False
        self._known: Optional[List[Tuple[str, np.ndarray]]] = None

    # -- model / index loading -------------------------------------------

    def _get_embedder(self):
        if self._embedder is not None or self._off:
            return self._embedder
        with self._lock:
            if self._embedder is not None or self._off:
                return self._embedder
            try:
                from facenet_pytorch import InceptionResnetV1
                print(f"[face] loading recognition model (facenet-pytorch "
                      f"'{self._model_name}'); first run downloads the weights...")
                self._embedder = InceptionResnetV1(pretrained=self._model_name).eval()
                print("[face] recognition model ready.")
            except Exception as e:
                self._off = True
                print(f"[face] Recognition disabled ({type(e).__name__}: {e}). "
                      f"Faces will be 'Unknown'.")
        return self._embedder

    def _get_known(self) -> List[Tuple[str, np.ndarray]]:
        if self._known is not None:
            return self._known
        known: List[Tuple[str, np.ndarray]] = []
        images = list_known_face_images(self._dir)
        if not images:
            print(f"[face] No known faces in '{self._dir}/' — everyone will be 'Unknown'. "
                  f"Add {self._dir}/<Name>/*.jpg or {self._dir}/<Name>.jpg to enable names.")
            self._known = known
            return known
        for name, path in images:
            img = cv2.imread(path)
            if img is None:
                print(f"[face] could not read {path}, skipping.")
                continue
            found = self._face_finder.detect(img)
            if not found.count:
                print(f"[face] no face found in {path}, skipping.")
                continue
            largest = max(found.boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            emb = self._embed(img, largest)
            if emb is not None:
                known.append((name, emb))
        people = sorted(set(n for n, _ in known))
        print(f"[face] loaded {len(known)} embedding(s) for {len(people)} people: "
              f"{', '.join(people) if people else '(none)'}")
        self._known = known
        return known

    # -- embedding + matching --------------------------------------------

    def _embed(self, frame_bgr: np.ndarray, box, margin: float = 0.2) -> Optional[np.ndarray]:
        """Crops the face box (with margin) -> 512-d L2-normalised embedding."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        import torch
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = box[:4]
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.cvtColor(cv2.resize(crop, (160, 160)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 127.5) / 128.0  # facenet fixed image standardization
        with self._lock, torch.no_grad():
            emb = embedder(tensor)[0].cpu().numpy()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _best_match(self, embedding: Optional[np.ndarray],
                    known: List[Tuple[str, np.ndarray]]) -> str:
        """Embeddings are L2-normalised, so dot product == cosine similarity."""
        if embedding is None or not known:
            return "Unknown"
        best_name, best_sim = "Unknown", -1.0
        for name, kemb in known:
            sim = float(np.dot(embedding, kemb))
            if sim > best_sim:
                best_name, best_sim = name, sim
        return best_name if best_sim >= self._threshold else "Unknown"

    def identify(self, frame_bgr: np.ndarray, boxes) -> List[str]:
        known = self._get_known()
        return [self._best_match(self._embed(frame_bgr, box), known) for box in boxes]
