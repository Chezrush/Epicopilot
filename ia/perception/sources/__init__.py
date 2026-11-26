from ia.perception.sources.base import (
    SourceError,
    VideoFrame,
    VideoSource,
    ensure_bgr,
)
from ia.perception.sources.file import FileSource
from ia.perception.sources.screen import ScreenSource
from ia.perception.sources.webcam import WebcamSource

_BUILDERS = {"file": FileSource, "webcam": WebcamSource, "screen": ScreenSource}


def open_source(kind: str, **kwargs) -> VideoSource:
    """Fabrique une source par son nom : "file" | "webcam" | "screen"."""
    if kind not in _BUILDERS:
        raise ValueError(f"Source inconnue : {kind!r}. Choix possibles : {', '.join(_BUILDERS)}.")
    return _BUILDERS[kind](**kwargs)


__all__ = [
    "FileSource",
    "ScreenSource",
    "SourceError",
    "VideoFrame",
    "VideoSource",
    "WebcamSource",
    "ensure_bgr",
    "open_source",
]
