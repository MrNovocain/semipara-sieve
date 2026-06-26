from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .el import empirical_likelihood_scalar


@dataclass(frozen=True)
class WorkbookRates:
    T: int
    K: int
    a_K: float
    Delta_T: float | None
    kappa_T: float
    R_T: float
    r_T: float | None
    rate_check_available: bool
    rate_check_pass: bool | None
    r_T_over_sqrt_T: float | None
    kappa_R_over_T_delta_sq: float | None

    def order_penalty(self, q: int) -> float:
        return float(int(q) * self.kappa_T * self.R_T)


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
    residualized_score_weight_strength: float
    orth_u: float
    orth_score_weight: float
    residualized_predictor_strength: float
    orth_w: float


def workbook_rate_terms(T: int, K: int, a_K: float, delta_T: float | None, kappa_T: float | None = None) -> WorkbookRates:
    T = int(T)
    K = int(K)
    a_K = float(a_K)
    if T <= 0:
        raise ValueError("T must be positive")
    if K <= 0:
        raise ValueError("K must be positive")
    if kappa_T is None:
        kappa_T = log(log(T + np.e**np.e))
    kappa_T = float(kappa_T)
    if kappa_T <= 0.0 or not np.isfinite(kappa_T):
        raise ValueError("kappa_T must be positive and finite")
    R_T = float(K + log(T) + T * a_K**2)
    if delta_T is None:
        return WorkbookRates(
            T=T,
            K=K,
            a_K=a_K,
            Delta_T=None,
            kappa_T=kappa_T,
            R_T=R_T,
            r_T=None,
            rate_check_available=False,
            rate_check_pass=None,
            r_T_over_sqrt_T=None,
            kappa_R_over_T_delta_sq=None,
        )
    delta_value = float(delta_T)
    if delta_value <= 0.0 or not np.isfinite(delta_value):
        raise ValueError("delta_T must be positive and finite when supplied")
    r_T = float(R_T / delta_value**2)
    r_ratio = float(r_T / np.sqrt(T))
    penalty_ratio = float(kappa_T * R_T / (T * delta_value**2))
    return WorkbookRates(
        T=T,
        K=K,
        a_K=a_K,
        Delta_T=delta_value,
        kappa_T=kappa_T,
        R_T=R_T,
        r_T=r_T,
        rate_check_available=True,
        rate_check_pass=bool(r_ratio < 1.0 and penalty_ratio < 1.0),
        r_T_over_sqrt_T=r_ratio,
        kappa_R_over_T_delta_sq=penalty_ratio,
    )


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
    nuisance_w: np.ndarray | None = None,
    beta0: float = 0.0,
    K: int = 1,
    breaks: tuple[int, ...] | list[int] = (),
    score_weight: np.ndarray | None = None,
    weight_b: float = 1.0,
    w: np.ndarray | None = None,
) -> ProfileELStatistic:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if nuisance_w is None:
        if w is None:
            raise ValueError("nuisance_w is required")
        nuisance_w = w
    nuisance_w = np.asarray(nuisance_w, dtype=float)
    if not (y.shape == x.shape == nuisance_w.shape and y.ndim == 1):
        raise ValueError("y, x, and nuisance_w must be conformable one-dimensional arrays")
    clean_breaks = validate_breaks(breaks, len(y))
    P = block_sieve_matrix(nuisance_w, clean_breaks, K)
    y_resid = residualize_against(P, y - float(beta0) * x)
    if score_weight is None:
        raw_score_weight = np.tanh(float(weight_b) * x)
    else:
        raw_score_weight = np.asarray(score_weight, dtype=float)
        if raw_score_weight.shape != y.shape:
            raise ValueError("score_weight must be conformable with y")
    residualized_score_weight = residualize_against(P, raw_score_weight)
    z = y_resid * residualized_score_weight
    el = empirical_likelihood_scalar(z)
    stat = float(el.statistic)
    p_value = float(np.nan if not np.isfinite(stat) else 1.0 - _chi_square_1_cdf(stat))
    score_var = float(np.mean(z**2))
    strength = float(np.mean(residualized_score_weight**2))
    orth_score_weight = float(np.linalg.norm(P.T @ residualized_score_weight))
    return ProfileELStatistic(
        breaks=clean_breaks,
        el_stat=stat,
        p_value=p_value,
        feasible=bool(el.feasible),
        score=float(np.sum(z) / np.sqrt(len(z))),
        score_variance=score_var,
        rss=float(y_resid @ y_resid),
        residualized_score_weight_strength=strength,
        orth_u=float(np.linalg.norm(P.T @ y_resid)),
        orth_score_weight=orth_score_weight,
        residualized_predictor_strength=strength,
        orth_w=orth_score_weight,
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
    """Legacy BIC-style selector retained for exploratory comparisons."""
    prepared = _prepare_partition_problem(y, x, w, beta0, K, q_max, min_size, grid_step)
    by_q: list[dict[str, object]] = []
    for row in _partition_rows(prepared):
        q = int(row["q"])
        segments = q + 1
        rss = float(row["rss"])
        effective_dim = K * segments
        criterion = float(prepared["T"] * log(max(rss / prepared["T"], np.finfo(float).tiny)) + penalty_multiplier * effective_dim * log(prepared["T"]))
        by_q.append({**row, "criterion": criterion, "criterion_type": "bic_style", "effective_dim": effective_dim})
    return _selection_from_rows(by_q)


def select_workbook_partition(
    y: np.ndarray,
    x: np.ndarray,
    nuisance_w: np.ndarray,
    beta0: float,
    K: int,
    min_size: int,
    grid_step: int = 1,
    q_max: int = 1,
    a_K: float = 0.0,
    delta_T: float | None = None,
    kappa_T: float | None = None,
    allow_multiple_breaks: bool = False,
) -> PartitionSelection:
    """Paper-faithful selector using pen_T(q,K)=q*kappa_T*R_T.

    The workbook's proved estimated-partition Wilks theorem is one-break.  Calls
    with q_max > 1 must opt into exploratory mode explicitly.
    """
    q_max = int(q_max)
    if q_max > 1 and not allow_multiple_breaks:
        raise ValueError("workbook paper mode is one-break; pass allow_multiple_breaks=True for exploratory searches")
    prepared = _prepare_partition_problem(y, x, nuisance_w, beta0, K, q_max, min_size, grid_step)
    rates = workbook_rate_terms(T=prepared["T"], K=K, a_K=a_K, delta_T=delta_T, kappa_T=kappa_T)
    mode = "paper_one_break" if q_max <= 1 and not allow_multiple_breaks else "exploratory_multiple_breaks"
    by_q: list[dict[str, object]] = []
    for row in _partition_rows(prepared):
        q = int(row["q"])
        rss = float(row["rss"])
        workbook_penalty = rates.order_penalty(q)
        criterion = float(rss + workbook_penalty)
        by_q.append(
            {
                **row,
                "criterion": criterion,
                "criterion_type": "workbook_rss_plus_q_kappa_R",
                "workbook_penalty": workbook_penalty,
                "R_T": rates.R_T,
                "r_T": rates.r_T,
                "Delta_T": rates.Delta_T,
                "kappa_T": rates.kappa_T,
                "rate_check_available": rates.rate_check_available,
                "rate_check_pass": rates.rate_check_pass,
                "r_T_over_sqrt_T": rates.r_T_over_sqrt_T,
                "kappa_R_over_T_delta_sq": rates.kappa_R_over_T_delta_sq,
                "mode": mode,
            }
        )
    return _selection_from_rows(by_q)


def diagnose_workbook_conditions(
    y: np.ndarray,
    x: np.ndarray,
    nuisance_w: np.ndarray,
    score_weight: np.ndarray,
    beta0: float,
    K: int,
    breaks: tuple[int, ...] | list[int],
    a_K: float,
    delta_T: float | None,
    kappa_T: float | None = None,
    approximation_error: np.ndarray | None = None,
    gram_condition_max: float = 1e10,
    min_sign_fraction: float = 0.01,
    score_bias_tolerance: float = 1e-8,
) -> dict[str, float | bool]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    nuisance_w = np.asarray(nuisance_w, dtype=float)
    score_weight = np.asarray(score_weight, dtype=float)
    if not (y.shape == x.shape == nuisance_w.shape == score_weight.shape and y.ndim == 1):
        raise ValueError("y, x, nuisance_w, and score_weight must be conformable one-dimensional arrays")
    P = block_sieve_matrix(nuisance_w, breaks, K)
    gram_scaled = P.T @ P / len(y)
    eigenvalues = np.linalg.eigvalsh(gram_scaled)
    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))
    condition = float(max_eig / min_eig) if min_eig > 0 else np.inf
    stat = profile_el_statistic(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=beta0,
        K=K,
        breaks=breaks,
        score_weight=score_weight,
    )
    y_resid = residualize_against(P, y - float(beta0) * x)
    residualized_score_weight = residualize_against(P, score_weight)
    scores = y_resid * residualized_score_weight
    pos_frac = float(np.mean(scores > 0.0))
    neg_frac = float(np.mean(scores < 0.0))
    rates = workbook_rate_terms(T=len(y), K=K, a_K=a_K, delta_T=delta_T, kappa_T=kappa_T)
    if approximation_error is None:
        score_bias = None
        score_bias_ok = None
    else:
        approximation_error = np.asarray(approximation_error, dtype=float)
        if approximation_error.shape != y.shape:
            raise ValueError("approximation_error must be conformable with y")
        score_bias = float(abs(approximation_error @ residualized_score_weight) / np.sqrt(len(y)))
        score_bias_ok = bool(score_bias <= score_bias_tolerance)
    return {
        "gram_min_eigenvalue": min_eig,
        "gram_max_eigenvalue": max_eig,
        "gram_condition_number": condition,
        "gram_stable": bool(np.isfinite(condition) and condition <= gram_condition_max),
        "convex_hull_contains_zero": bool(pos_frac > 0.0 and neg_frac > 0.0 and stat.feasible),
        "positive_score_fraction": pos_frac,
        "negative_score_fraction": neg_frac,
        "sign_mass_ok": bool(pos_frac >= min_sign_fraction and neg_frac >= min_sign_fraction),
        "Delta_T": rates.Delta_T,
        "R_T": rates.R_T,
        "r_T": rates.r_T,
        "kappa_T": rates.kappa_T,
        "rate_check_available": rates.rate_check_available,
        "rate_check_pass": rates.rate_check_pass,
        "r_T_over_sqrt_T": rates.r_T_over_sqrt_T,
        "kappa_R_over_T_delta_sq": rates.kappa_R_over_T_delta_sq,
        "score_bias_proxy": score_bias,
        "score_bias_ok": score_bias_ok,
    }


def _prepare_partition_problem(
    y: np.ndarray,
    x: np.ndarray,
    nuisance_w: np.ndarray,
    beta0: float,
    K: int,
    q_max: int,
    min_size: int,
    grid_step: int,
) -> dict[str, object]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    nuisance_w = np.asarray(nuisance_w, dtype=float)
    if not (y.shape == x.shape == nuisance_w.shape and y.ndim == 1):
        raise ValueError("y, x, and nuisance_w must be conformable one-dimensional arrays")
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
            P_seg = polynomial_segment_basis(nuisance_w[start:end], K)
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
    return {"T": T, "points": points, "dp": dp, "prev": prev, "q_max": q_max}


def _partition_rows(prepared: dict[str, object]) -> list[dict[str, object]]:
    points = prepared["points"]
    dp = prepared["dp"]
    prev = prepared["prev"]
    q_max = int(prepared["q_max"])
    last = len(points) - 1
    rows: list[dict[str, object]] = []
    for q in range(q_max + 1):
        segments = q + 1
        rss = float(dp[segments, last])
        if not np.isfinite(rss):
            continue
        breaks = _recover_breaks(points, prev, segments, last)
        rows.append({"q": q, "breaks": breaks, "rss": rss})
    return rows


def _selection_from_rows(by_q: list[dict[str, object]]) -> PartitionSelection:
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
