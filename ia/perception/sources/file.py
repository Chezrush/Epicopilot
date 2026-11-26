import time
from pathlib import Path

import cv2

from ia.perception.sources.base import CaptureSource, VideoFrame


class FileSource(CaptureSource):
    def __init__(self, path: str | Path, realtime: bool = False) -> None:
        super().__init__()
        self._path = Path(path)
        self._realtime = realtime
        self._fps = 0.0
        self._start = 0.0

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self._path))

    def _open_error_message(self) -> str:
        return f"Impossible d'ouvrir le fichier vidéo: {self._path}."

    def open(self) -> None:
        super().open()
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0  # repli si le conteneur ne déclare pas de FPS
        self._start = time.monotonic()

    def _end_of_stream(self) -> VideoFrame | None:
        return None  # fin normale du fichier

    def _pace(self, index: int) -> None:
        if not self._realtime:
            return
        # On attend jusqu'à l'instant où l'image `index` doit être montrée (t = index / fps).
        delay = (self._start + index / self._fps) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
