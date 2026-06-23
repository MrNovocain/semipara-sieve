import numpy as np
import pytest

from pseel.dgp import PredictiveAR1DGP


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