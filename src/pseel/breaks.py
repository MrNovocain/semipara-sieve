from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .el import empirical_likelihood_scalar


@dataclass(frozen=True)
class PartitionSelection:
    breaks: tuple[int, ...]
    q: int
    rss: float
    criterion: float
    by_q: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ProfileELStatistic:
    breaks: tuple[int, ...]
    el_stat: float
    p_value: float
    feasible: bool
    score: float
    score_variance: float
    rss: float
    residualized_predictor_strength: float
    orth_u: float
    orth_w: float


def validate_breaks(breaks: tuple[int, ...] | list[int], T: int) -> tuple[int, ...]:
    clean = tuple(int(b) for b in breaks)
    if any(b <= 0 or b >= T for b in clean):
        raise ValueError("break dates must lie strictly inside the sample")
    if tuple(sorted(clean)) != clean or len(set(clean)) != len(clean):
        raise ValueError("break dates must be sorted and unique")
    return clean


def segment_bounds(T: int, breaks: tuple[int, ...] | list[int]) -> tuple[tuple[int, int], ...]:
    clean = validate_breaks(breaks, T)
    points = (0, *clean, int(T))
    return tuple((points[j], points[j + 1]) for j in range(len(points) - 1))


def polynomial_segment_basis(w: np.ndarray, K: int, standardize: bool = True) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("w must be one-dimensional")
    K = int(K)
    if K <= 0:
        raise ValueError("K must be positive")
    z = w.copy()
    if standardize:
        scale = float(z.std())
        z = (z - float(z.mean())) / scale if scale > 0 else z - float(z.mean())
    cols = [np.ones_like(z)]
    for degree in range(1, K):
        cols.append(z**degree)
    return np.column_stack(cols)


def block_sieve_matrix(w: np.ndarray, breaks: tuple[int, ...] | list[int], K: int) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    T = int(w.shape[0])
    bounds = segment_bounds(T, breaks)
    K = int(K)
    Q = np.zeros((T, K * len(bounds)), dtype=float)
    for j, (start, end) in enumerate(bounds):
        if end - start <= K:
            raise ValueError("each regime must have more observations than sieve columns")
        Q[start:end, j * K : (j + 1) * K] = polynomial_segment_basis(w[start:end], K)
    if np.linalg.matrix_rank(Q) < Q.shape[1]:
        raise ValueError("block sieve matrix is rank deficient")
    return Q


def residualize_against(P: np.ndarray, z: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    z = np.asarray(z, dtype=float)
    if P.ndim != 2:
        raise ValueError("P must be a matrix")
    if z.ndim != 1 or z.shape[0] != P.shape[0]:
        raise ValueError("z must be a vector conformable with P")
    coef, *_ = np.linalg.lstsq(P, z, rcond=None)
    return z - P @ coef


def profile_rss(y: np.ndarray, x: np.ndarray, w: np.ndarray, beta0: float, K: int, breaks: tuple[int, ...] | list[int]) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    P = block_sieve_matrix(w, breaks, K)
    resid = residualize_against(P, y - float(beta0) * x)
    return float(resid @ resid)


def profile_el_statistic(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    beta0: float,
    K: int,
    breaks: tuple[int, ...] | list[int],
    weight_b: float = 1.0,
) -> ProfileELStatistic:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if not (y.shape == x.shape == w.shape and y.ndim == 1):
        raise ValueError("y, x, and w must be conformable one-dimensional arrays")
    clean_breaks = validate_breaks(breaks, len(y))
    P = block_sieve_matrix(w, clean_breaks, K)
    y_resid = residualize_against(P, y - float(beta0) * x)
    instrument = residualize_against(P, np.tanh(float(weight_b) * x))
    z = y_resid * instrument
    el = empirical_likelihood_scalar(z)
    stat = float(el.statistic)
    p_value = float(np.nan if not np.isfinite(stat) else 1.0 - _chi_square_1_cdf(stat))
    score_var = float(np.mean(z**2))
    return ProfileELStatistic(
        breaks=clean_breaks,
        el_stat=stat,
        p_value=p_value,
        feasible=bool(el.feasible),
        score=float(np.sum(z) / np.sqrt(len(z))),
        score_variance=score_var,
        rss=float(y_resid @ y_resid),
        residualized_predictor_strength=float(np.mean(instrument**2)),
        orth_u=float(np.linalg.norm(P.T @ y_resid)),
        orth_w=float(np.linalg.norm(P.T @ instrument)),
    )


def select_profile_partition(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    beta0: float,
    K: int,
    q_max: int,
    min_size: int,
    grid_step: int = 1,
    penalty_multiplier: float = 1.0,
) -> PartitionSelection:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if not (y.shape == x.shape == w.shape and y.ndim == 1):
        raise ValueError("y, x, and w must be conformable one-dimensional arrays")
    T = int(len(y))
    q_max = int(q_max)
    min_size = int(min_size)
    grid_step = int(grid_step)
    if q_max < 0:
        raise ValueError("q_max must be nonnegative")
    if min_size <= K:
        raise ValueError("min_size must exceed K")
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if T < (q_max + 1) * min_size:
        raise ValueError("sample is too short for q_max and min_size")

    candidates = list(range(min_size, T - min_size + 1, grid_step))
    points = np.array([0, *candidates, T], dtype=int)
    n_points = len(points)
    interval_cost = np.full((n_points, n_points), np.inf, dtype=float)
    y_tilde = y - float(beta0) * x
    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            start = int(points[i])
            end = int(points[j])
            if end - start < min_size:
                continue
            P_seg = polynomial_segment_basis(w[start:end], K)
            resid = residualize_against(P_seg, y_tilde[start:end])
            interval_cost[i, j] = float(resid @ resid)

    max_segments = q_max + 1
    dp = np.full((max_segments + 1, n_points), np.inf, dtype=float)
    prev = np.full((max_segments + 1, n_points), -1, dtype=int)
    dp[0, 0] = 0.0
    for segments in range(1, max_segments + 1):
        for j in range(1, n_points):
            best_value = np.inf
            best_i = -1
            for i in range(j):
                value = dp[segments - 1, i] + interval_cost[i, j]
                if value < best_value:
                    best_value = value
                    best_i = i
            dp[segments, j] = best_value
            prev[segments, j] = best_i

    by_q: list[dict[str, object]] = []
    last = n_points - 1
    for q in range(q_max + 1):
        segments = q + 1
        rss = float(dp[segments, last])
        if not np.isfinite(rss):
            continue
        breaks = _recover_breaks(points, prev, segments, last)
        effective_dim = K * segments
        criterion = float(T * log(max(rss / T, np.finfo(float).tiny)) + penalty_multiplier * effective_dim * log(T))
        by_q.append({"q": q, "breaks": breaks, "rss": rss, "criterion": criterion})
    if not by_q:
        raise ValueError("no feasible partition found")
    selected = min(by_q, key=lambda row: float(row["criterion"]))
    return PartitionSelection(
        breaks=tuple(int(b) for b in selected["breaks"]),
        q=int(selected["q"]),
        rss=float(selected["rss"]),
        criterion=float(selected["criterion"]),
        by_q=tuple(by_q),
    )


def _recover_breaks(points: np.ndarray, prev: np.ndarray, segments: int, last: int) -> tuple[int, ...]:
    indices = []
    j = int(last)
    for s in range(segments, 0, -1):
        i = int(prev[s, j])
        if i < 0:
            raise ValueError("failed to recover dynamic-programming path")
        indices.append(i)
        j = i
    indices = list(reversed(indices))
    break_indices = indices[1:]
    return tuple(int(points[idx]) for idx in break_indices)


def _chi_square_1_cdf(x: float) -> float:
    from scipy.stats import chi2

    return float(chi2.cdf(x, 1))