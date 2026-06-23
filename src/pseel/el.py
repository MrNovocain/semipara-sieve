from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .contracts import ELResult


def empirical_likelihood_scalar(z: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> ELResult:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return ELResult(float("inf"), None, False, "empty score")
    if np.allclose(z, 0.0, atol=tol):
        return ELResult(0.0, 0.0, True)
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_min > 0.0 or z_max < 0.0:
        return ELResult(float("inf"), None, False, "zero outside convex hull")
    if abs(float(np.mean(z))) <= tol:
        return ELResult(0.0, 0.0, True)

    lower = max((-1.0 / zi for zi in z if zi > 0.0), default=-np.inf)
    upper = min((-1.0 / zi for zi in z if zi < 0.0), default=np.inf)
    eps = max(tol, np.finfo(float).eps * 100)
    lo = lower + eps if np.isfinite(lower) else -1.0 / eps
    hi = upper - eps if np.isfinite(upper) else 1.0 / eps
    if not lo < hi:
        return ELResult(float("inf"), None, False, "empty lambda bracket")

    def score(lam: float) -> float:
        denom = 1.0 + lam * z
        if np.any(denom <= 0.0):
            return np.nan
        return float(np.sum(z / denom))

    try:
        f_lo = score(lo)
        f_hi = score(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return ELResult(float("inf"), None, False, "lambda root not bracketed")
        lam = float(brentq(score, lo, hi, xtol=tol, maxiter=max_iter))
        values = 1.0 + lam * z
        if np.any(values <= 0.0):
            return ELResult(float("inf"), None, False, "nonpositive EL weights")
        stat = float(2.0 * np.sum(np.log(values)))
    except Exception as exc:  # pragma: no cover - defensive path
        return ELResult(float("inf"), None, False, f"solver failed: {exc}")
    return ELResult(max(stat, 0.0), lam, True)
