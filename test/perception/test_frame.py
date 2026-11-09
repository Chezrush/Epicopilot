import dataclasses

import pytest

from ia.perception.frame import (
    Detection,
    ObstacleGrid,
    PerceptionFrame,
    StageLatencies,
    TargetState,
)


def make_frame() -> PerceptionFrame:
    return PerceptionFrame(
        index=0,
        timestamp=0.0,
        source_ok=True,
        error=None,
        target_state=TargetState.NONE,
        target=None,
        lost_for_ms=None,
        detections=(),
        obstacles=None,
        latencies=StageLatencies(capture_ms=1.0, detection_ms=2.0, depth_ms=None, total_ms=3.0),
    )


def test_detection_geometry():
    detection = Detection(track_id=1, class_name="person", confidence=0.9, box=(10, 20, 50, 100))
    assert detection.width == 40
    assert detection.height == 80
    assert detection.area == 3200
    assert detection.center == (30, 60)


def test_frame_is_immutable():
    frame = make_frame()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.index = 1


def test_target_state_values():
    assert TargetState.LOCKED.value == "locked"
    assert TargetState.LOST.value == "lost"
    assert TargetState.NONE.value == "none"


def make_detection(**overrides) -> Detection:
    fields = {"track_id": 1, "class_name": "person", "confidence": 0.9, "box": (10, 20, 50, 100)}
    fields.update(overrides)
    return Detection(**fields)


def test_detection_converts_list_box_to_float_tuple():
    detection = make_detection(box=[10, 20, 50, 100])
    assert detection.box == (10.0, 20.0, 50.0, 100.0)
    assert isinstance(detection.box, tuple)
    assert all(isinstance(value, float) for value in detection.box)


def test_detection_allows_degenerate_box():
    assert make_detection(box=(10, 20, 10, 20)).area == 0


@pytest.mark.parametrize(
    "box",
    [
        (50, 20, 10, 100),
        (10, 100, 50, 20),
        (10, 20, 50),
        (10, 20, 50, float("nan")),
        (10, 20, 50, float("inf")),
    ],
)
def test_detection_rejects_invalid_box(box):
    with pytest.raises(ValueError):
        make_detection(box=box)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan")])
def test_detection_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        make_detection(confidence=confidence)


def test_detection_converts_keypoints_to_tuples():
    detection = make_detection(keypoints=[[1, 2, 0.5], [3, 4, 0.9]])
    assert detection.keypoints == ((1.0, 2.0, 0.5), (3.0, 4.0, 0.9))


def test_detection_rejects_malformed_keypoint():
    with pytest.raises(ValueError):
        make_detection(keypoints=[(1, 2)])


def test_grid_converts_lists_and_checks_shape():
    grid = ObstacleGrid(
        rows=1, cols=2, proximity=[[0.0, 1.0]], critical=[[False, True]], timestamp=0.0
    )
    assert grid.proximity == ((0.0, 1.0),)
    assert grid.critical == ((False, True),)
    with pytest.raises(ValueError):
        ObstacleGrid(
            rows=2, cols=2, proximity=[[0.0, 1.0]], critical=[[False, True]], timestamp=0.0
        )


def test_grid_rejects_proximity_out_of_range():
    with pytest.raises(ValueError):
        ObstacleGrid(rows=1, cols=1, proximity=[[1.5]], critical=[[True]], timestamp=0.0)


def test_latencies_reject_negative_values():
    with pytest.raises(ValueError):
        StageLatencies(capture_ms=-1.0, detection_ms=0.0, depth_ms=None, total_ms=0.0)


def test_frame_converts_detection_list_to_tuple():
    frame = dataclasses.replace(make_frame(), detections=[make_detection()])
    assert isinstance(frame.detections, tuple)


def test_frame_rejects_non_detection_items():
    with pytest.raises(TypeError):
        dataclasses.replace(make_frame(), detections=[(10, 20, 50, 100)])


def test_locked_frame_requires_target():
    with pytest.raises(ValueError):
        dataclasses.replace(make_frame(), target_state=TargetState.LOCKED)


def test_lost_frame_requires_duration_and_no_target():
    lost = dataclasses.replace(make_frame(), target_state=TargetState.LOST, lost_for_ms=500.0)
    assert lost.lost_for_ms == 500.0
    with pytest.raises(ValueError):
        dataclasses.replace(make_frame(), target_state=TargetState.LOST)


def test_failed_source_requires_error_message():
    with pytest.raises(ValueError):
        dataclasses.replace(make_frame(), source_ok=False)
    with pytest.raises(ValueError):
        dataclasses.replace(make_frame(), error="caméra débranchée")
