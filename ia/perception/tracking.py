from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ia.perception.frame import Detection, TargetState


@dataclass(frozen=True)
class TrackingResult:
    state: TargetState
    detection: Detection | None
    lost_for_ms: float | None


class TargetTracker:
    def __init__(
        self,
        target_class: str,
        lost_timeout_ms: float = 2000.0,
        auto_reacquire: bool = False,
    ) -> None:
        self.target_class = target_class
        self.lost_timeout_ms = lost_timeout_ms
        self.auto_reacquire = auto_reacquire

        self._state: TargetState = TargetState.NONE
        self._locked_id: int | None = None
        self._last_seen: float | None = None

    def _best_candidate(self, detections: Iterable[Detection]) -> Detection | None:
        valid = [
            d
            for d in detections
            if d.class_name == self.target_class and d.track_id is not None
        ]
        if not valid:
            return None
        return max(valid, key=lambda d: d.confidence * d.area)

    def reset(self) -> None:
        self._state = TargetState.NONE
        self._locked_id = None
        self._last_seen = None

    def lock(self, track_id: int) -> None:
        self._locked_id = track_id
        self._state = TargetState.LOCKED

    def update(self, detections: Iterable[Detection], now: float) -> TrackingResult:
        det_list = list(detections)

        if self._state in (TargetState.LOCKED, TargetState.LOST):
            locked_det = next(
                (d for d in det_list if d.track_id == self._locked_id),
                None,
            )
            if locked_det is not None:
                self._last_seen = now
                self._state = TargetState.LOCKED
                return TrackingResult(
                    state=TargetState.LOCKED,
                    detection=locked_det,
                    lost_for_ms=None,
                )

            lost_for_ms = (
                (now - self._last_seen) * 1000.0
                if self._last_seen is not None
                else 0.0
            )

            if self.auto_reacquire and lost_for_ms >= self.lost_timeout_ms:
                best = self._best_candidate(det_list)
                if best is not None:
                    self._locked_id = best.track_id
                    self._last_seen = now
                    self._state = TargetState.LOCKED
                    return TrackingResult(
                        state=TargetState.LOCKED,
                        detection=best,
                        lost_for_ms=None,
                    )

            self._state = TargetState.LOST
            return TrackingResult(
                state=TargetState.LOST,
                detection=None,
                lost_for_ms=lost_for_ms,
            )

        best = self._best_candidate(det_list)
        if best is not None:
            self._locked_id = best.track_id
            self._last_seen = now
            self._state = TargetState.LOCKED
            return TrackingResult(
                state=TargetState.LOCKED,
                detection=best,
                lost_for_ms=None,
            )

        return TrackingResult(
            state=TargetState.NONE,
            detection=None,
            lost_for_ms=None,
        )
