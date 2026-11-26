import time

import mss
import numpy as np

from ia.perception.sources.base import SourceError, VideoFrame, VideoSource, ensure_bgr


class ScreenSource(VideoSource):
    def __init__(self, region: tuple[int, int, int, int] | None = None) -> None:
        # region = (gauche, haut, largeur, hauteur); None = écran principal.
        self._region = region
        self._sct = None
        self._monitor: dict | None = None
        self._index = 0

    def open(self) -> None:
        try:
            self._sct = mss.mss()
        except Exception as exc:  # pas de serveur d'affichage, par exemple
            raise SourceError(f"Impossible d'initialiser la capture d'écran: {exc}.") from exc
        if self._region is None:
            self._monitor = self._sct.monitors[1]  # [0] = tous les écrans, [1] = principal
        else:
            left, top, width, height = self._region
            self._monitor = {"left": left, "top": top, "width": width, "height": height}
        self._index = 0

    def read(self) -> VideoFrame | None:
        if self._sct is None:
            raise SourceError("Source écran non ouverte : appeler open() d'abord.")
        shot = self._sct.grab(self._monitor)  # BGRA
        frame = VideoFrame(
            image=ensure_bgr(np.asarray(shot)),
            timestamp=time.monotonic(),
            index=self._index,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None
