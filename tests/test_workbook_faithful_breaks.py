import numpy as np
import pytest

from pseel.breaks import (
    diagnose_workbook_conditions,
    profile_el_statistic,
    select_workbook_partition,
    workbook_rate_terms,
)


def _broken_sample(T=140, true_break=70, seed=123):
    rng = np.random.default_rng(seed)
    nuisance_w = rng.normal(size=T)
    x = rng.normal(size=T)
    m0 = 0.25 * nuisance_w
    m1 = 0.25 * nuisance_w + 1.5
    y = np.where(np.arange(T) < true_break, m0, m1) + 0.05 * rng.normal(size=T)
    score_weight = np.tanh(x)
    delta_t = float(np.sqrt(np.mean((m1 - m0) ** 2)))
    return y, x, nuisance_w, score_weight, delta_t, true_break


def test_workbook_rate_terms_use_manuscript_definitions():
    rates = workbook_rate_terms(T=200, K=5, a_K=0.1, delta_T=2.0, kappa_T=3.0)

    assert np.isclose(rates.R_T, 5 + np.log(200) + 200 * 0.01)
    assert np.isclose(rates.r_T, rates.R_T / 4.0)
    assert np.isclose(rates.order_penalty(2), 2 * 3.0 * rates.R_T)


def test_select_workbook_partition_uses_q_kappa_R_penalty_and_selects_one_break():
    y, x, nuisance_w, _score_weight, delta_t, true_break = _broken_sample()

    selected = select_workbook_partition(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=0.0,
        K=1,
        min_size=35,
        grid_step=1,
        a_K=0.0,
        delta_T=delta_t,
        kappa_T=2.0,
    )

    assert selected.q == 1
    assert abs(selected.breaks[0] - true_break) <= 2
    assert all("workbook_penalty" in row for row in selected.by_q)
    q1 = next(row for row in selected.by_q if row["q"] == 1)
    assert np.isclose(q1["criterion"], q1["rss"] + q1["workbook_penalty"])


def test_select_workbook_partition_rejects_multiple_breaks_unless_exploratory():
    y, x, nuisance_w, _score_weight, delta_t, _true_break = _broken_sample()

    with pytest.raises(ValueError, match="one-break"):
        select_workbook_partition(
            y=y,
            x=x,
            nuisance_w=nuisance_w,
            beta0=0.0,
            K=1,
            min_size=30,
            q_max=2,
            a_K=0.0,
            delta_T=delta_t,
        )

    exploratory = select_workbook_partition(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=0.0,
        K=1,
        min_size=30,
        q_max=2,
        allow_multiple_breaks=True,
        a_K=0.0,
        delta_T=delta_t,
    )
    assert exploratory.q in {0, 1, 2}
    assert exploratory.by_q[0]["mode"] == "exploratory_multiple_breaks"


def test_profile_el_statistic_separates_nuisance_covariate_from_score_weight():
    y, x, nuisance_w, score_weight, _delta_t, true_break = _broken_sample(T=120, true_break=60)

    stat = profile_el_statistic(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=0.0,
        K=2,
        breaks=[true_break],
        score_weight=score_weight,
    )

    assert np.isfinite(stat.el_stat)
    assert stat.residualized_score_weight_strength > 0.0
    assert stat.orth_score_weight < 1e-8


def test_diagnose_workbook_conditions_reports_theorem_quantities():
    y, x, nuisance_w, score_weight, delta_t, true_break = _broken_sample(T=120, true_break=60)
    approx_error = np.zeros_like(y)

    diagnostics = diagnose_workbook_conditions(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        score_weight=score_weight,
        beta0=0.0,
        K=2,
        breaks=[true_break],
        a_K=0.0,
        delta_T=delta_t,
        kappa_T=2.0,
        approximation_error=approx_error,
    )

    expected = {
        "gram_min_eigenvalue",
        "gram_condition_number",
        "gram_stable",
        "convex_hull_contains_zero",
        "positive_score_fraction",
        "negative_score_fraction",
        "Delta_T",
        "R_T",
        "r_T",
        "score_bias_proxy",
        "score_bias_ok",
    }
    assert expected.issubset(diagnostics)
    assert diagnostics["gram_stable"] is True
    assert diagnostics["convex_hull_contains_zero"] is True
    assert diagnostics["Delta_T"] == delta_t
    assert diagnostics["R_T"] > 0.0
    assert diagnostics["r_T"] > 0.0
    assert diagnostics["score_bias_proxy"] == 0.0

def test_select_workbook_partition_with_unknown_delta_marks_rate_checks_unavailable():
    y, x, nuisance_w, _score_weight, _delta_t, _true_break = _broken_sample()

    selected = select_workbook_partition(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=0.0,
        K=1,
        min_size=35,
        grid_step=1,
        a_K=0.0,
        delta_T=None,
        kappa_T=2.0,
    )

    chosen = selected.by_q[selected.q]
    assert chosen["Delta_T"] is None
    assert chosen["r_T"] is None
    assert chosen["rate_check_available"] is False
    assert chosen["rate_check_pass"] is None
    assert chosen["R_T"] > 0.0
    assert chosen["workbook_penalty"] == selected.q * 2.0 * chosen["R_T"]


def test_diagnose_workbook_conditions_with_unknown_delta_marks_rates_unavailable():
    y, x, nuisance_w, score_weight, _delta_t, true_break = _broken_sample(T=120, true_break=60)

    diagnostics = diagnose_workbook_conditions(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        score_weight=score_weight,
        beta0=0.0,
        K=2,
        breaks=[true_break],
        a_K=0.0,
        delta_T=None,
        kappa_T=2.0,
    )

    assert diagnostics["Delta_T"] is None
    assert diagnostics["r_T"] is None
    assert diagnostics["rate_check_available"] is False
    assert diagnostics["rate_check_pass"] is None
    assert diagnostics["score_bias_ok"] is None

