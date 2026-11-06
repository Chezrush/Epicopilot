import warnings

import torch

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def is_available(device: str) -> bool:
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return device == "cpu"


def best_available() -> str:
    for device in ("cuda", "mps"):
        if is_available(device):
            return device
    return "cpu"


def select_device(preference: str = "auto") -> str:
    if preference not in VALID_DEVICES:
        raise ValueError(
            f"Valeur inconnue : {preference!r}. Valeurs possibles : {', '.join(VALID_DEVICES)}."
        )

    if preference == "auto":
        return best_available()

    if is_available(preference):
        return preference

    fallback = best_available()
    warnings.warn(
        f"Le matériel {preference} n'est pas disponible ; utilisation de {fallback}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return fallback
