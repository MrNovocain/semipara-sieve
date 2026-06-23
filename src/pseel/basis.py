from __future__ import annotations

import numpy as np

from .registry import register_basis


@register_basis("polynomial")
class PolynomialBasis:
    def __init__(self, K: int, include_intercept: bool = True, standardize_w: bool = True):
        if K <= 0:
            raise ValueError("K must be positive.")
        if not include_intercept:
            raise ValueError("M1 requires an intercept in the sieve basis.")
        self.K = int(K)
        self.include_intercept = bool(include_intercept)
        self.standardize_w = bool(standardize_w)

    def make(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        if w.ndim != 1:
            raise ValueError("w must be one-dimensional.")
        z = w.copy()
        if self.standardize_w:
            scale = z.std()
            z = (z - z.mean()) / scale if scale > 0 else z - z.mean()
        cols = [np.ones_like(z)]
        for degree in range(1, self.K):
            cols.append(z**degree)
        P = np.column_stack(cols)
        if not np.all(np.isfinite(P)):
            raise ValueError("Basis contains non-finite values.")
        return P
