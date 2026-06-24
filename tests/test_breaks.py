import numpy as np

from pseel.breaks import (
    block_sieve_matrix,
    profile_el_statistic,
    residualize_against,
    select_profile_partition,
)


def test_block_sieve_residuals_are_orthogonal_within_regimes():
    rng = np.random.default_rng(123)
    w = rng.normal(size=90)
    z = rng.normal(size=90)
    P = block_sieve_matrix(w, breaks=[45], K=3)
    resid = residualize_against(P, z)
    assert np.linalg.norm(P.T @ resid) < 1e-8


def test_profile_partition_selects_clear_level_break():
    rng = np.random.default_rng(321)
    T = 120
    true_break = 60
    w = rng.normal(size=T)
    x = np.zeros(T)
    y = np.r_[np.zeros(true_break), np.ones(T - true_break) * 2.0]
    y = y + 0.05 * rng.normal(size=T)
    selected = select_profile_partition(
        y=y,
        x=x,
        w=w,
        beta0=0.0,
        K=1,
        q_max=1,
        min_size=25,
        grid_step=1,
        penalty_multiplier=1.0,
    )
    assert selected.q == 1
    assert abs(selected.breaks[0] - true_break) <= 2


def test_profile_el_statistic_returns_finite_value_for_valid_partition():
    rng = np.random.default_rng(456)
    T = 100
    w = rng.normal(size=T)
    x = rng.normal(size=T)
    y = 0.3 * np.sin(w) + rng.normal(size=T)
    stat = profile_el_statistic(y=y, x=x, w=w, beta0=0.0, K=3, breaks=[50], weight_b=0.5)
    assert np.isfinite(stat.el_stat)
    assert 0.0 <= stat.p_value <= 1.0
    assert stat.residualized_predictor_strength > 0.0
    assert stat.orth_u < 1e-8
    assert stat.orth_w < 1e-8