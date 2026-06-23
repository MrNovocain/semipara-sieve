import numpy as np

from pseel.el import empirical_likelihood_scalar


def test_zero_mean_gives_zero_statistic():
    z = np.array([-1.0, 0.0, 1.0])
    result = empirical_likelihood_scalar(z)
    assert result.feasible
    assert abs(result.lambda_hat) < 1e-10
    assert result.statistic < 1e-10


def test_scaling_invariance():
    z = np.array([-2.0, -0.5, 0.2, 1.0, 3.0]) + 0.2
    stat1 = empirical_likelihood_scalar(z).statistic
    stat2 = empirical_likelihood_scalar(7.5 * z).statistic
    assert np.isfinite(stat1)
    assert abs(stat1 - stat2) < 1e-8


def test_convex_hull_infeasible_when_all_positive():
    result = empirical_likelihood_scalar(np.array([1.0, 2.0, 3.0]))
    assert not result.feasible
    assert np.isinf(result.statistic)


def test_quadratic_approximation_when_mean_small():
    z = np.array([-2.0, -1.0, 0.5, 1.0, 1.7])
    result = empirical_likelihood_scalar(z)
    approx = (z.sum() ** 2) / np.sum(z**2)
    assert result.feasible
    assert abs(result.statistic - approx) < 0.05
