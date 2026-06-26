import numpy as np
import pytest

from pseel.dgp import BrokenNuisanceAR1DGP, PredictiveAR1DGP


def _params(**overrides):
    params = {
        "a_w": 0.5,
        "kappa": 0.4,
        "xi": 0.2,
        "burnin": 50,
        "m": {"name": "sinus_quad", "params": {"a1": 0.5, "a2": 0.3}},
    }
    params.update(overrides)
    return params


def test_dgp_output_shapes_and_finiteness():
    dgp = PredictiveAR1DGP(_params())
    data = dgp.simulate(seed=123, T=100, rho_design={"label": "high", "formula": "fixed", "value": 0.95}, beta=0.0)
    assert data.T == 100
    assert len(data.y) == len(data.x_lag) == len(data.w_lag) == len(data.u) == len(data.m_w)
    assert np.all(np.isfinite(data.y))
    assert data.meta["rho_label"] == "high"


def test_dgp_rejects_non_positive_definite_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        PredictiveAR1DGP(_params(kappa=0.99, xi=0.99))

def test_w_process_is_stationary_ar1_with_expected_variance():
    a_w = 0.5
    dgp = PredictiveAR1DGP(_params(a_w=a_w, burnin=0))
    data = dgp.simulate(seed=789, T=5000, rho_design={"label": "stationary", "formula": "fixed", "value": 0.5}, beta=0.0)
    expected_var = 1.0 / (1.0 - a_w**2)
    assert data.meta["w_process"] == "stationary_gaussian_ar1"
    assert data.meta["w_stationary"] is True
    assert data.meta["w_alpha_mixing"] == "geometric"
    assert abs(float(np.var(data.w_lag)) - expected_var) < 0.15


def test_unit_root_predictor_starts_from_controlled_initial_condition():
    dgp = PredictiveAR1DGP(_params(burnin=50))
    data = dgp.simulate(seed=246, T=40, rho_design={"label": "unit_root", "formula": "fixed", "value": 1.0}, beta=0.0)
    assert data.meta["x_initialization"] == "zero_at_sample_start"
    assert data.x_lag[0] == 0.0

def test_broken_nuisance_dgp_records_one_break_and_jump_size():
    dgp = BrokenNuisanceAR1DGP(
        _params(
            break_fraction=0.4,
            m_left={"name": "sinus_quad", "params": {"a1": 0.2, "a2": 0.0}},
            m_right={"name": "sinus_quad", "params": {"a1": 0.2, "a2": 0.0, "level": 1.25}},
        )
    )

    data = dgp.simulate(seed=135, T=100, rho_design={"label": "stationary", "formula": "fixed", "value": 0.6}, beta=0.0)

    assert data.meta["q0"] == 1
    assert data.meta["true_break"] == 40
    assert data.meta["Delta_T"] > 1.0
    assert len(data.meta["m_left_values"]) == data.T
    assert len(data.meta["m_right_values"]) == data.T
    assert np.allclose(data.m_w[:40], np.asarray(data.meta["m_left_values"])[:40])
    assert np.allclose(data.m_w[40:], np.asarray(data.meta["m_right_values"])[40:])
