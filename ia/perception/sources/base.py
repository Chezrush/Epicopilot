import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np


class SourceError(Exception): ...


@dataclass(frozen=True)
class VideoFrame:
    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: float  # time.monotonic() au moment de la capture
    index: int  # 0, 1, 2… dans l'ordre de lecture


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise SourceError(f"Format d'image non supporté : forme {image.shape}.")


class VideoSource(ABC):

    @abstractmethod
    def open(self) -> None:
        """Ouvre la source. Lève SourceError si c'est impossible."""

    @abstractmethod
    def read(self) -> VideoFrame | None:
        """Image suivante, ou None en fin de flux. Lève SourceError en cas de panne."""

    @abstractmethod
    def close(self) -> None:
        """Libère les ressources. Idempotent."""

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False  # ne masque pas une éventuelle exception


class CaptureSource(VideoSource):
    """Socle commun aux sources basées sur cv2.VideoCapture (fichier, webcam)."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    # --- à spécialiser par les sous-classes --------------------------------

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        """Construit l'objet VideoCapture (chemin de fichier ou index de webcam)."""

    @abstractmethod
    def _open_error_message(self) -> str:
        """Message d'erreur si l'ouverture échoue (doit identifier la source)."""

    @abstractmethod
    def _end_of_stream(self) -> VideoFrame | None:
        """Que faire quand read() échoue : None (fichier fini) ou lever (webcam en panne)."""

    def _pace(self, index: int) -> None:
        """Point d'extension : ralentit la lecture (mode temps réel). No-op par défaut."""

    # --- logique commune ---------------------------------------------------

    def open(self) -> None:
        cap = self._open_capture()
        if not cap.isOpened():
            cap.release()
            raise SourceError(self._open_error_message())
        self._cap = cap
        self._index = 0

    def read(self) -> VideoFrame | None:
        if self._cap is None:
            raise SourceError("Source non ouverte : appeler open() d'abord.")
        ok, image = self._cap.read()
        if not ok:
            return self._end_of_stream()
        self._pace(self._index)
        frame = VideoFrame(image=ensure_bgr(image), timestamp=time.monotonic(), index=self._index)
        self._index += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
