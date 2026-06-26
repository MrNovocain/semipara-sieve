from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .breaks import diagnose_workbook_conditions, profile_el_statistic, select_workbook_partition
from .dgp import BrokenNuisanceAR1DGP


def workbook_break_replication(
    seed: int,
    T: int,
    K: int,
    beta0: float,
    beta: float,
    rho_design: dict[str, Any],
    dgp_params: dict[str, Any],
    min_size: int,
    grid_step: int = 1,
    a_K: float = 0.0,
    kappa_T: float | None = None,
    weight_b: float = 1.0,
) -> dict[str, Any]:
    dgp = BrokenNuisanceAR1DGP(dgp_params)
    data = dgp.simulate(seed=seed, T=T, rho_design=rho_design, beta=beta)
    true_break = int(data.meta["true_break"])
    delta_T = float(data.meta["Delta_T"])
    score_weight = np.tanh(np.asarray(data.x_lag, dtype=float) / float(weight_b))

    selected = select_workbook_partition(
        y=data.y,
        x=data.x_lag,
        nuisance_w=data.w_lag,
        beta0=beta0,
        K=K,
        min_size=min_size,
        grid_step=grid_step,
        q_max=1,
        a_K=a_K,
        delta_T=delta_T,
        kappa_T=kappa_T,
    )
    known = profile_el_statistic(
        y=data.y,
        x=data.x_lag,
        nuisance_w=data.w_lag,
        beta0=beta0,
        K=K,
        breaks=[true_break],
        score_weight=score_weight,
    )
    estimated = profile_el_statistic(
        y=data.y,
        x=data.x_lag,
        nuisance_w=data.w_lag,
        beta0=beta0,
        K=K,
        breaks=selected.breaks,
        score_weight=score_weight,
    )
    diagnostics = diagnose_workbook_conditions(
        y=data.y,
        x=data.x_lag,
        nuisance_w=data.w_lag,
        score_weight=score_weight,
        beta0=beta0,
        K=K,
        breaks=selected.breaks,
        a_K=a_K,
        delta_T=delta_T,
        kappa_T=kappa_T,
        approximation_error=np.zeros_like(data.y),
    )
    selected_break = int(selected.breaks[0]) if selected.breaks else -1
    break_error = selected_break - true_break if selected_break >= 0 else np.nan
    selected_metadata = next(row for row in selected.by_q if int(row["q"]) == int(selected.q))
    return {
        "seed": int(seed),
        "T": int(T),
        "K": int(K),
        "beta": float(beta),
        "beta0": float(beta0),
        "rho_label": data.meta["rho_label"],
        "rho_value": float(data.meta["rho_value"]),
        "q0": int(data.meta["q0"]),
        "true_break": true_break,
        "selected_q": int(selected.q),
        "selected_break": selected_break,
        "break_error": float(break_error),
        "Delta_T": delta_T,
        "R_T": float(diagnostics["R_T"]),
        "r_T": float(diagnostics["r_T"]),
        "r_T_over_sqrt_T": float(diagnostics["r_T_over_sqrt_T"]),
        "kappa_R_over_T_delta_sq": float(diagnostics["kappa_R_over_T_delta_sq"]),
        "rate_check_available": bool(diagnostics["rate_check_available"]),
        "rate_check_pass": bool(diagnostics["rate_check_pass"]),
        "workbook_penalty": float(selected_metadata["workbook_penalty"]),
        "mode": str(selected_metadata["mode"]),
        "known_el_stat": float(known.el_stat),
        "known_p_value": float(known.p_value),
        "known_reject_5": bool(known.p_value <= 0.05),
        "known_feasible": bool(known.feasible),
        "estimated_el_stat": float(estimated.el_stat),
        "estimated_p_value": float(estimated.p_value),
        "estimated_reject_5": bool(estimated.p_value <= 0.05),
        "estimated_feasible": bool(estimated.feasible),
        "gram_stable": bool(diagnostics["gram_stable"]),
        "gram_condition_number": float(diagnostics["gram_condition_number"]),
        "convex_hull_contains_zero": bool(diagnostics["convex_hull_contains_zero"]),
        "positive_score_fraction": float(diagnostics["positive_score_fraction"]),
        "negative_score_fraction": float(diagnostics["negative_score_fraction"]),
        "score_bias_proxy": float(diagnostics["score_bias_proxy"]),
        "score_bias_ok": bool(diagnostics["score_bias_ok"]),
        "criterion": float(selected.criterion),
        "criterion_type": str(selected_metadata["criterion_type"]),
    }


def run_workbook_break_monte_carlo(
    seeds: Iterable[int],
    T: int,
    K: int,
    beta0: float,
    beta: float,
    rho_design: dict[str, Any],
    dgp_params: dict[str, Any],
    min_size: int,
    grid_step: int = 1,
    a_K: float = 0.0,
    kappa_T: float | None = None,
    weight_b: float = 1.0,
) -> pd.DataFrame:
    rows = [
        workbook_break_replication(
            seed=int(seed),
            T=T,
            K=K,
            beta0=beta0,
            beta=beta,
            rho_design=rho_design,
            dgp_params=dgp_params,
            min_size=min_size,
            grid_step=grid_step,
            a_K=a_K,
            kappa_T=kappa_T,
            weight_b=weight_b,
        )
        for seed in seeds
    ]
    return pd.DataFrame(rows)

