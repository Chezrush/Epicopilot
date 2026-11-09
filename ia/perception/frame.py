import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class TargetState(Enum):
    LOCKED = "locked"
    LOST = "lost"
    NONE = "none"


Box = tuple[float, float, float, float]  # x1, y1, x2, y2 en pixels
Keypoint = tuple[float, float, float]  # x, y, confiance (format COCO, 17 points)


def _finite_floats(values: Iterable, name: str, length: int | None = None) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} doit contenir {length} valeurs, reçu {len(result)}.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contient une valeur non finie (NaN ou infini) : {result}.")
    return result


def _check_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} doit être dans [0, 1], reçu {value}.")


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_name: str
    confidence: float
    box: Box
    keypoints: tuple[Keypoint, ...] | None = None

    def __post_init__(self):
        box = _finite_floats(self.box, "box", length=4)
        x1, y1, x2, y2 = box
        if x2 < x1 or y2 < y1:
            raise ValueError(f"box inversée : x2 >= x1 et y2 >= y1 attendus, reçu {box}.")
        object.__setattr__(self, "box", box)

        (confidence,) = _finite_floats([self.confidence], "confidence")
        _check_unit_interval(confidence, "confidence")
        object.__setattr__(self, "confidence", confidence)

        if self.keypoints is not None:
            keypoints = tuple(
                _finite_floats(point, "keypoint", length=3) for point in self.keypoints
            )
            object.__setattr__(self, "keypoints", keypoints)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.box[0] + self.box[2]) / 2,
            (self.box[1] + self.box[3]) / 2,
        )


@dataclass(frozen=True)
class Target:
    detection: Detection
    azimuth_deg: float
    elevation_deg: float
    distance_m: float | None
    torso_facing: Literal["front", "back", "side"] | None


@dataclass(frozen=True)
class ObstacleGrid:
    rows: int
    cols: int
    proximity: tuple[tuple[float, ...], ...]
    critical: tuple[tuple[bool, ...], ...]
    timestamp: float

    def __post_init__(self):
        proximity = tuple(
            _finite_floats(row, "proximity", length=self.cols) for row in self.proximity
        )
        critical = tuple(tuple(bool(cell) for cell in row) for row in self.critical)
        if len(proximity) != self.rows or len(critical) != self.rows:
            raise ValueError(f"La grille doit avoir {self.rows} lignes.")
        if any(len(row) != self.cols for row in critical):
            raise ValueError(f"Chaque ligne de critical doit avoir {self.cols} colonnes.")
        for row in proximity:
            for value in row:
                _check_unit_interval(value, "proximity")
        object.__setattr__(self, "proximity", proximity)
        object.__setattr__(self, "critical", critical)


@dataclass(frozen=True)
class StageLatencies:
    capture_ms: float
    detection_ms: float
    depth_ms: float | None
    total_ms: float

    def __post_init__(self):
        for name in ("capture_ms", "detection_ms", "depth_ms", "total_ms"):
            value = getattr(self, name)
            if value is not None and not value >= 0.0:
                raise ValueError(f"{name} doit être >= 0, reçu {value}.")


@dataclass(frozen=True)
class PerceptionFrame:
    index: int
    timestamp: float
    source_ok: bool
    error: str | None
    target_state: TargetState
    target: Target | None
    lost_for_ms: float | None
    detections: tuple[Detection, ...]
    obstacles: ObstacleGrid | None
    latencies: StageLatencies

    def __post_init__(self):
        detections = tuple(self.detections)
        if not all(isinstance(detection, Detection) for detection in detections):
            raise TypeError("detections ne doit contenir que des Detection.")
        object.__setattr__(self, "detections", detections)

        if self.source_ok == (self.error is not None):
            raise ValueError("error doit être renseigné si et seulement si source_ok est False.")

        if (self.target_state is TargetState.LOCKED) != (self.target is not None):
            raise ValueError("target doit être renseigné si et seulement si l'état est LOCKED.")

        if (self.target_state is TargetState.LOST) != (self.lost_for_ms is not None):
            raise ValueError("lost_for_ms doit être renseigné si et seulement si l'état est LOST.")
