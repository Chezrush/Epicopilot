import math
from typing import Literal

from ia.perception.frame import Keypoint


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    if not math.isfinite(image_width) or image_width <= 0:
        raise ValueError("image_width doit être une largeur finie strictement positive.")
    if not 0 < hfov_deg < 180:
        raise ValueError("hfov_deg doit être strictement compris entre 0 et 180 degrés.")
    return (image_width / 2) / math.tan(math.radians(hfov_deg) / 2)


def pixel_to_angles(x: float, y: float, image_width: int, image_height: int, hfov_deg: float, ) -> tuple[float, float]:
    if not math.isfinite(image_height) or image_height <= 0:
        raise ValueError("image_height doit être une hauteur finie strictement positive.")
    focal = focal_length_px(image_width, hfov_deg)
    dx = x - image_width / 2
    dy = image_height / 2 - y
    return math.degrees(math.atan(dx / focal)), math.degrees(math.atan(dy / focal))


def estimate_distance_m(box_height_px: float, real_height_m: float, image_width: int, hfov_deg: float,) -> float | None:
    if (not math.isfinite(box_height_px) or not math.isfinite(real_height_m)
        or box_height_px <= 0 or real_height_m <= 0):
        return None
    return focal_length_px(image_width, hfov_deg) * real_height_m / box_height_px


def torso_facing(keypoints: tuple[Keypoint, ...] | None, box_width: float, ) -> Literal["front", "back", "side"] | None:
    if keypoints is None or len(keypoints) < 7:
        return None
    if not math.isfinite(box_width) or box_width <= 0:
        return None

    left_x, _, left_conf = keypoints[5]
    right_x, _, right_conf = keypoints[6]
    if not all(math.isfinite(value) for value in (left_x, right_x, left_conf, right_conf)):
        return None
    if left_conf < 0.5 or right_conf < 0.5:
        return None

    dx = left_x - right_x
    if abs(dx) < 0.15 * box_width:
        return "side"
    return "front" if dx > 0 else "back"
