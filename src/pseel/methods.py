from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .basis import PolynomialBasis
from .contracts import Dataset, MethodResult
from .el import empirical_likelihood_scalar
from .registry import build_weight, register_method
from .residualize import Residualizer
from .weights import LinearWeight, TanhWeight  # noqa: F401 - ensure registration


def _reject_5(stat: float) -> bool:
    return bool(np.isfinite(stat) and stat > chi2.ppf(0.95, 1))


def _basis_from_config(config: dict, K: int | None = None) -> PolynomialBasis:
    if config.get("basis", "polynomial") != "polynomial":
        raise ValueError("M1 currently supports only polynomial basis.")
    return PolynomialBasis(
        K=int(K if K is not None else config["K"]),
        include_intercept=bool(config.get("include_intercept", True)),
        standardize_w=bool(config.get("standardize_w", True)),
    )


def _full_basis_residualizer(data: Dataset, basis_config: dict, K: int | None) -> tuple[np.ndarray, Residualizer]:
    P = _basis_from_config(basis_config, K).make(data.w_lag)
    return P, Residualizer(P)


def _projected_instrument(data: Dataset, weight, basis_config: dict, K: int | None) -> tuple[np.ndarray, np.ndarray]:
    P, R = _full_basis_residualizer(data, basis_config, K)
    raw = weight.transform(data.x_lag)
    return R.residualize(raw), P


def _intercept_instrument(data: Dataset, weight) -> tuple[np.ndarray, np.ndarray]:
    P = np.ones((data.T, 1))
    R = Residualizer(P)
    return R.residualize(weight.transform(data.x_lag)), P


def _ds_dv(z: np.ndarray, oracle_z: np.ndarray) -> dict[str, float]:
    denom = float(np.mean(oracle_z**2))
    if denom <= 0 or not np.isfinite(denom):
        return {"DS": np.nan, "DV": np.nan, "oracle_var": denom}
    ds = float(np.sum(z - oracle_z) / np.sqrt(len(z)) / np.sqrt(denom))
    dv = float(np.mean(z**2) / denom - 1.0)
    return {"DS": ds, "DV": dv, "oracle_var": denom}


def _relative_efficiency(instrument: np.ndarray, efficient: np.ndarray) -> float:
    numerator = float(instrument @ efficient) ** 2
    denominator = float((instrument @ instrument) * (efficient @ efficient))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator)


class BaseMethod:
    def __init__(self, name: str, basis_config: dict, weight_config: dict, el_config: dict):
        self.name = name
        self.basis_config = basis_config
        self.weight_config = weight_config
        self.weight = build_weight(weight_config)
        solver = el_config.get("solver", {})
        self.tol = float(solver.get("tolerance", 1e-10))
        self.max_iter = int(solver.get("max_iter", 100))

    def _finish(self, z: np.ndarray, beta0: float, diagnostics: dict[str, float]) -> MethodResult:
        el = empirical_likelihood_scalar(z, tol=self.tol, max_iter=self.max_iter)
        return MethodResult(
            method_name=self.name,
            el_stat=el.statistic,
            reject_5=_reject_5(el.statistic),
            beta0=float(beta0),
            feasible=el.feasible,
            lambda_hat=el.lambda_hat,
            diagnostics=diagnostics,
        )


@register_method("oracle_bounded")
class OracleBoundedEL(BaseMethod):
    def evaluate(self, data: Dataset, beta0: float, K: int | None = None) -> MethodResult:
        instrument, P = _projected_instrument(data, self.weight, self.basis_config, K)
        g_eff = Residualizer(P).residualize(data.x_lag)
        z = data.u * instrument
        diagnostics = {
            "DS": 0.0,
            "DV": 0.0,
            "oracle_var": float(np.mean(z**2)),
            "orth_u": np.nan,
            "orth_w": float(np.linalg.norm(P.T @ instrument)),
            "RE": _relative_efficiency(instrument, g_eff),
            "instrument_var": float(np.mean(instrument**2)),
            "efficient_var": float(np.mean(g_eff**2)),
        }
        return self._finish(z, beta0, diagnostics)


@register_method("profile_bounded")
class ProfileBoundedEL(BaseMethod):
    def evaluate(self, data: Dataset, beta0: float, K: int | None = None) -> MethodResult:
        P, R = _full_basis_residualizer(data, self.basis_config, K)
        uhat = R.residualize(data.y - beta0 * data.x_lag)
        instrument = R.residualize(self.weight.transform(data.x_lag))
        g_eff = R.residualize(data.x_lag)
        z = uhat * instrument
        oracle_z = data.u * instrument
        diagnostics = {
            **_ds_dv(z, oracle_z),
            "orth_u": float(np.linalg.norm(P.T @ uhat)),
            "orth_w": float(np.linalg.norm(P.T @ instrument)),
            "RE": _relative_efficiency(instrument, g_eff),
            "instrument_var": float(np.mean(instrument**2)),
            "efficient_var": float(np.mean(g_eff**2)),
        }
        return self._finish(z, beta0, diagnostics)


@register_method("profile_bounded_frontier")
class ProfileBoundedFrontierEL(ProfileBoundedEL):
    pass


@register_method("intercept_only_bounded")
class InterceptOnlyBoundedEL(BaseMethod):
    def evaluate(self, data: Dataset, beta0: float, K: int | None = None) -> MethodResult:
        instrument, P = _intercept_instrument(data, self.weight)
        R = Residualizer(P)
        uhat = R.residualize(data.y - beta0 * data.x_lag)
        g_eff = R.residualize(data.x_lag)
        z = uhat * instrument
        oracle_z = data.u * instrument
        diagnostics = {
            **_ds_dv(z, oracle_z),
            "orth_u": float(np.linalg.norm(P.T @ uhat)),
            "orth_w": float(np.linalg.norm(P.T @ instrument)),
            "RE": _relative_efficiency(instrument, g_eff),
            "instrument_var": float(np.mean(instrument**2)),
            "efficient_var": float(np.mean(g_eff**2)),
        }
        return self._finish(z, beta0, diagnostics)


@register_method("profile_efficient")
class ProfileEfficientEL(BaseMethod):
    def evaluate(self, data: Dataset, beta0: float, K: int | None = None) -> MethodResult:
        P, R = _full_basis_residualizer(data, self.basis_config, K)
        uhat = R.residualize(data.y - beta0 * data.x_lag)
        g_eff = R.residualize(data.x_lag)
        z = uhat * g_eff
        denom = float((g_eff @ g_eff) * (g_eff @ g_eff))
        re = 1.0 if denom > 0 else np.nan
        diagnostics = {
            "DS": np.nan,
            "DV": np.nan,
            "oracle_var": np.nan,
            "orth_u": float(np.linalg.norm(P.T @ uhat)),
            "orth_w": float(np.linalg.norm(P.T @ g_eff)),
            "RE": re,
            "instrument_var": float(np.mean(g_eff**2)),
            "efficient_var": float(np.mean(g_eff**2)),
        }
        return self._finish(z, beta0, diagnostics)