from __future__ import annotations

from scipy.stats import chi2


def add_rejection(row: dict, alpha: float = 0.05) -> dict:
    threshold = float(chi2.ppf(1.0 - alpha, 1))
    row[f"reject_{int(alpha * 100)}"] = bool(row["el_stat"] > threshold)
    return row
