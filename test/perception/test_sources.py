import time

import cv2
import numpy as np
import pytest

from ia.perception.sources import (
    FileSource,
    ScreenSource,
    SourceError,
    VideoFrame,
    WebcamSource,
    ensure_bgr,
    open_source,
)


@pytest.fixture
def tiny_video(tmp_path):
    """5 images 64x48 à 10 FPS, écrites dans un vrai fichier mp4 (codec mp4v)."""
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened(), "VideoWriter n'a pas pu ouvrir le fichier (codec mp4v absent ?)."
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


# --- FileSource ------------------------------------------------------------


def test_file_reads_all_frames_then_none(tiny_video):
    with FileSource(tiny_video) as source:
        frames = []
        while (frame := source.read()) is not None:
            frames.append(frame)

    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert all(isinstance(f, VideoFrame) for f in frames)
    assert all(f.image.shape == (48, 64, 3) for f in frames)
    assert all(f.image.dtype == np.uint8 for f in frames)
    # Les horodatages sont monotones (capturés avec time.monotonic()).
    assert [f.timestamp for f in frames] == sorted(f.timestamp for f in frames)


def test_file_missing_raises(tmp_path):
    missing = tmp_path / "absent.mp4"
    source = FileSource(missing)
    with pytest.raises(SourceError) as exc:
        source.open()
    assert str(missing) in str(exc.value)


def test_file_realtime_respects_fps(tiny_video):
    start = time.monotonic()
    with FileSource(tiny_video, realtime=True) as source:
        while source.read() is not None:
            pass
    # 5 images à 10 FPS : la dernière est rendue à t = 4/10 = 0,4 s.
    assert time.monotonic() - start >= 0.35


def test_file_fast_mode_does_not_wait(tiny_video):
    start = time.monotonic()
    with FileSource(tiny_video, realtime=False) as source:
        while source.read() is not None:
            pass
    # Sans realtime (benchmark), on lit le plus vite possible : bien en dessous de 0,4 s.
    assert time.monotonic() - start < 0.2


# --- ensure_bgr ------------------------------------------------------------


def test_ensure_bgr_from_bgra():
    bgra = np.zeros((48, 64, 4), dtype=np.uint8)
    assert ensure_bgr(bgra).shape == (48, 64, 3)


def test_ensure_bgr_from_gray():
    gray = np.zeros((48, 64), dtype=np.uint8)
    assert ensure_bgr(gray).shape == (48, 64, 3)


def test_ensure_bgr_passthrough():
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    assert ensure_bgr(bgr).shape == (48, 64, 3)


# --- open_source (factory) -------------------------------------------------


def test_open_source_unknown_kind_raises():
    with pytest.raises(ValueError):
        open_source("potato")


def test_open_source_builds_each_kind(tiny_video):
    # Les constructeurs ne touchent pas au matériel : seul open() le fait.
    assert isinstance(open_source("file", path=tiny_video), FileSource)
    assert isinstance(open_source("webcam"), WebcamSource)
    assert isinstance(open_source("screen"), ScreenSource)


# --- Matériel réel (désélectionné par défaut) ------------------------------


@pytest.mark.hardware
def test_webcam_reads_one_frame():
    with WebcamSource(0) as source:
        frame = source.read()
    assert frame is not None
    assert frame.image.shape[2] == 3


@pytest.mark.hardware
def test_screen_reads_one_frame():
    with ScreenSource() as source:
        frame = source.read()
    assert frame is not None
    assert frame.image.ndim == 3 and frame.image.shape[2] == 3
