import pytest

from ia.perception.frame import Detection, TargetState
from ia.perception.tracking import TargetTracker


def person(track_id, confidence, area, class_name="person"):
    """Détection carrée dont l'aire vaut exactement `area` (côté = racine de l'aire)."""
    side = area**0.5
    return Detection(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        box=(0.0, 0.0, side, side),
    )


def test_initial_none():
    tracker = TargetTracker("person")
    result = tracker.update((), now=0.0)
    assert result.state is TargetState.NONE
    assert result.detection is None
    assert result.lost_for_ms is None


def test_locks_best_score():
    # Score = confiance × aire. A : 0.9×100 = 90. B : 0.6×400 = 240 → B gagne.
    a = person(track_id=1, confidence=0.9, area=100)
    b = person(track_id=2, confidence=0.6, area=400)
    result = TargetTracker("person").update([a, b], now=0.0)
    assert result.state is TargetState.LOCKED
    assert result.detection.track_id == 2


def test_ignores_other_classes_and_untracked():
    # Un chien (mauvaise classe) et une personne sans track_id : aucun candidat valide.
    dog = person(track_id=5, confidence=0.99, area=1000, class_name="dog")
    untracked = person(track_id=None, confidence=0.99, area=1000)
    result = TargetTracker("person").update([dog, untracked], now=0.0)
    assert result.state is TargetState.NONE


def test_keeps_lock_even_if_better_appears():
    tracker = TargetTracker("person")
    a = person(track_id=1, confidence=0.9, area=400)  # score 360
    tracker.update([a], now=0.0)
    # B a un bien meilleur score (0.9×900 = 810), mais on ne change pas de cible seul.
    b = person(track_id=2, confidence=0.9, area=900)
    result = tracker.update([a, b], now=1.0)
    assert result.state is TargetState.LOCKED
    assert result.detection.track_id == 1


def test_lost_duration():
    tracker = TargetTracker("person")
    a = person(track_id=1, confidence=0.9, area=400)
    tracker.update([a], now=10.0)           # verrouillé, dernière fois vue : t=10.0
    result = tracker.update((), now=10.5)    # absente 0,5 s plus tard
    assert result.state is TargetState.LOST
    assert result.lost_for_ms == pytest.approx(500)


def test_no_reacquire_by_default():
    tracker = TargetTracker("person")  # auto_reacquire=False par défaut
    tracker.update([person(track_id=1, confidence=0.9, area=400)], now=0.0)
    # La cible 1 disparaît, une autre (2) est bien là, mais on ne se raccroche pas.
    result = tracker.update([person(track_id=2, confidence=0.9, area=900)], now=100.0)
    assert result.state is TargetState.LOST


def test_auto_reacquire_after_timeout():
    tracker = TargetTracker("person", lost_timeout_ms=2000.0, auto_reacquire=True)
    tracker.update([person(track_id=1, confidence=0.9, area=400)], now=0.0)

    b = person(track_id=2, confidence=0.9, area=900)
    # 1 s après la perte : sous le seuil de 2 s → toujours LOST, pas de nouvelle cible.
    assert tracker.update([b], now=1.0).state is TargetState.LOST
    # 3 s après la perte : seuil dépassé → on reverrouille sur le meilleur (ici B).
    result = tracker.update([b], now=3.0)
    assert result.state is TargetState.LOCKED
    assert result.detection.track_id == 2


def test_reset_returns_none():
    tracker = TargetTracker("person")
    tracker.update([person(track_id=1, confidence=0.9, area=400)], now=0.0)
    tracker.reset()
    # Après reset on est en NONE : une mise à jour vide reste NONE (et non LOST,
    # ce qui serait le cas si la cible était encore verrouillée).
    assert tracker.update((), now=1.0).state is TargetState.NONE


def test_manual_lock():
    tracker = TargetTracker("person")
    a = person(track_id=1, confidence=0.9, area=900)  # meilleur score
    b = person(track_id=2, confidence=0.5, area=100)  # moins bon
    tracker.lock(2)  # on impose la cible 2 malgré son score inférieur
    result = tracker.update([a, b], now=0.0)
    assert result.state is TargetState.LOCKED
    assert result.detection.track_id == 2
