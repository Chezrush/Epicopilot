import pytest

from ia.perception.geometry import (
    estimate_distance_m,
    focal_length_px,
    pixel_to_angles,
    torso_facing,
)


def test_focal():
    assert focal_length_px(640, 90) == pytest.approx(320.0)


@pytest.mark.parametrize("hfov_deg", [0, -5, 180, 270])
def test_invalid_fov_raises(hfov_deg):
    with pytest.raises(ValueError):
        focal_length_px(640, hfov_deg)


def test_center_is_zero():
    assert pixel_to_angles(320, 240, 640, 480, 90) == pytest.approx((0.0, 0.0))


def test_right_edge_is_45():
    assert pixel_to_angles(640, 240, 640, 480, 90) == pytest.approx((45.0, 0.0))


def test_left_edge_is_minus_45():
    assert pixel_to_angles(0, 240, 640, 480, 90) == pytest.approx((-45.0, 0.0))


def test_top_is_positive():
    assert pixel_to_angles(320, 0, 640, 480, 90)[1] == pytest.approx(36.8699, abs=1e-3)


def test_bottom_is_negative():
    assert pixel_to_angles(320, 480, 640, 480, 90)[1] == pytest.approx(-36.8699, abs=1e-3)


def test_distance():
    assert estimate_distance_m(160, 1.7, 640, 90) == pytest.approx(3.4)


def test_distance_zero_height_is_none():
    assert estimate_distance_m(0, 1.7, 640, 90) is None


def make_keypoints(left_x, right_x, left_conf=0.9, right_conf=0.9):
    points = [(0.0, 0.0, 0.0)] * 17
    points[5] = (float(left_x), 50.0, left_conf)
    points[6] = (float(right_x), 50.0, right_conf)
    return tuple(points)


def test_torso_front():
    assert torso_facing(make_keypoints(left_x=70, right_x=30), box_width=100) == "front"


def test_torso_back():
    assert torso_facing(make_keypoints(left_x=30, right_x=70), box_width=100) == "back"


def test_torso_side():
    assert torso_facing(make_keypoints(left_x=52, right_x=48), box_width=100) == "side"


def test_torso_low_confidence_is_none():
    assert (
        torso_facing(make_keypoints(left_x=70, right_x=30, right_conf=0.3), box_width=100) is None
    )


def test_torso_none_keypoints_is_none():
    assert torso_facing(None, box_width=100) is None


@pytest.mark.parametrize("image_width", [0, -10, float("nan"), float("inf")])
def test_invalid_width_raises(image_width):
    with pytest.raises(ValueError):
        focal_length_px(image_width, 90)


@pytest.mark.parametrize("image_height", [0, -10, float("nan")])
def test_invalid_height_raises(image_height):
    with pytest.raises(ValueError):
        pixel_to_angles(320, 240, 640, image_height, 90)


@pytest.mark.parametrize(
    ("box_height_px", "real_height_m"),
    [
        (-160, 1.7),
        (160, 0),
        (160, -1.7),
        (float("nan"), 1.7),
        (float("inf"), 1.7),
    ],
)
def test_distance_returns_none(box_height_px, real_height_m):
    assert estimate_distance_m(box_height_px, real_height_m, 640, 90) is None


def test_torso_left_low_confidence_is_none():
    kp = make_keypoints(left_x=70, right_x=30, left_conf=0.3)
    assert torso_facing(kp, box_width=100) is None


def test_torso_side_threshold_is_strict():
    assert torso_facing(make_keypoints(left_x=58, right_x=43), box_width=100) == "front"
    assert torso_facing(make_keypoints(left_x=57, right_x=43), box_width=100) == "side"


@pytest.mark.parametrize(
    ("keypoints", "box_width"),
    [
        ((), 100),
        (make_keypoints(left_x=70, right_x=30), 0),
        (make_keypoints(left_x=70, right_x=30), -5),
    ],
)
def test_torso_degenerate_input_is_none(keypoints, box_width):
    assert torso_facing(keypoints, box_width) is None
