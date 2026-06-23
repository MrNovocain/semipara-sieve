from __future__ import annotations

import numpy as np

from .registry import register_weight


@register_weight("tanh")
class TanhWeight:
    def __init__(self, b: float = 1.0):
        if b <= 0:
            raise ValueError("tanh scale b must be positive.")
        self.b = float(b)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(x, dtype=float) / self.b)


@register_weight("linear")
class LinearWeight:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)
