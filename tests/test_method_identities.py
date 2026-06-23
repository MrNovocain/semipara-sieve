import numpy as np

from pseel.basis import PolynomialBasis
from pseel.dgp import PredictiveAR1DGP
from pseel.methods import InterceptOnlyBoundedEL, ProfileBoundedEL, ProfileEfficientEL
from pseel.residualize import Residualizer


BASIS_CONFIG = {"basis": "polynomial", "include_intercept": True, "standardize_w": True, "K": 1}
WEIGHT_CONFIG = {"name": "tanh", "params": {"b": 1.0}}
EL_CONFIG = {"solver": {"tolerance": 1e-10, "max_iter": 100}}


def test_profile_components_are_orthogonal_to_basis():
    dgp = PredictiveAR1DGP({"a_w": 0.5, "kappa": 0.2, "xi": 0.1, "burnin": 40, "m": {"name": "zero"}})
    data = dgp.simulate(seed=321, T=80, rho_design={"label": "rho", "formula": "fixed", "value": 0.9}, beta=0.0)
    P = PolynomialBasis(K=4).make(data.w_lag)
    R = Residualizer(P)
    uhat = R.residualize(data.y)
    wc = R.residualize(np.tanh(data.x_lag))
    assert np.linalg.norm(P.T @ uhat) < 1e-8
    assert np.linalg.norm(P.T @ wc) < 1e-8


def test_profile_with_k1_matches_intercept_only():
    dgp = PredictiveAR1DGP({"a_w": 0.5, "kappa": 0.2, "xi": 0.1, "burnin": 40, "m": {"name": "zero"}})
    data = dgp.simulate(seed=456, T=80, rho_design={"label": "rho", "formula": "fixed", "value": 0.9}, beta=0.0)
    profile = ProfileBoundedEL("profile_bounded", BASIS_CONFIG, WEIGHT_CONFIG, EL_CONFIG).evaluate(data, beta0=0.0, K=1)
    intercept = InterceptOnlyBoundedEL("intercept_only_bounded", BASIS_CONFIG, WEIGHT_CONFIG, EL_CONFIG).evaluate(data, beta0=0.0, K=1)
    assert abs(profile.el_stat - intercept.el_stat) < 1e-9


def test_profile_relative_efficiency_is_squared_angle():
    dgp = PredictiveAR1DGP({"a_w": 0.5, "kappa": 0.2, "xi": 0.1, "burnin": 40, "m": {"name": "sinus_quad"}})
    data = dgp.simulate(seed=654, T=120, rho_design={"label": "rho", "formula": "fixed", "value": 0.9}, beta=0.0)
    basis_config = {"basis": "polynomial", "include_intercept": True, "standardize_w": True, "K": 4}
    profile = ProfileBoundedEL("profile_bounded", basis_config, WEIGHT_CONFIG, EL_CONFIG).evaluate(data, beta0=0.0, K=4)
    efficient = ProfileEfficientEL("profile_efficient", basis_config, WEIGHT_CONFIG, EL_CONFIG).evaluate(data, beta0=0.0, K=4)
    assert 0.0 <= profile.diagnostics["RE"] <= 1.0 + 1e-12
    assert efficient.diagnostics["RE"] == 1.0