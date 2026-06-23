from __future__ import annotations

import numpy as np


class Residualizer:
    def __init__(self, P: np.ndarray, ridge: float = 0.0):
        P = np.asarray(P, dtype=float)
        if P.ndim != 2:
            raise ValueError("P must be a matrix.")
        if P.shape[0] <= P.shape[1]:
            raise ValueError("P must have more rows than columns.")
        if np.linalg.matrix_rank(P) < P.shape[1]:
            raise ValueError("P is rank deficient.")
        self.P = P
        self.ridge = float(ridge)
        self.gram = P.T @ P

    def coefficients(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.shape[0] != self.P.shape[0]:
            raise ValueError("z length does not match basis rows.")
        G = self.gram + self.ridge * np.eye(self.gram.shape[0])
        return np.linalg.solve(G, self.P.T @ z)

    def project(self, z: np.ndarray) -> np.ndarray:
        return self.P @ self.coefficients(z)

    def residualize(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return z - self.project(z)
