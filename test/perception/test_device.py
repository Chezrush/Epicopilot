import pytest

from ia.perception.device import select_device


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert select_device("auto") == "cuda"


def test_auto_uses_mps_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert select_device("auto") == "mps"


def test_unavailable_request_falls_back_to_best_available(monkeypatch):
    # Mac : on demande cuda, mais seul mps existe -> on obtient mps, pas cpu.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    with pytest.warns(RuntimeWarning, match="cuda"):
        assert select_device("cuda") == "mps"


def test_unavailable_request_falls_back_to_cpu_when_nothing_else(monkeypatch):
    # Machine sans GPU : on demande cuda, rien n'existe -> cpu.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="cuda"):
        assert select_device("cuda") == "cpu"


def test_explicit_request_is_respected(monkeypatch):
    # PC NVIDIA : on force le cpu (benchmark) -> cpu, même si cuda existe.
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert select_device("cpu") == "cpu"


def test_invalid_preference_raises():
    with pytest.raises(ValueError, match="auto, cuda, mps, cpu"):
        select_device("gpu")
