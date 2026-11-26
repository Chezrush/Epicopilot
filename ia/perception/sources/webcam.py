import cv2

from ia.perception.sources.base import CaptureSource, SourceError, VideoFrame


class WebcamSource(CaptureSource):
    def __init__(self, index: int = 0) -> None:
        super().__init__()
        self._camera_index = index

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._camera_index)

    def _open_error_message(self) -> str:
        return f"Impossible d'ouvrir la webcam d'index {self._camera_index}."

    def _end_of_stream(self) -> VideoFrame | None:
        raise SourceError(f"Lecture échouée sur la webcam d'index {self._camera_index}.")
