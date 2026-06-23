from __future__ import annotations

import numpy as np

from .contracts import Dataset
from .registry import register_dgp


def _rho_from_design(design: dict, T: int) -> float:
    formula = design.get("formula", "fixed")
    if formula == "fixed":
        return float(design["value"])
    if formula == "local":
        return float(1.0 - float(design["c"]) / T)
    raise ValueError(f"Unknown rho formula: {formula}")


def _m_function(w: np.ndarray, config: dict) -> np.ndarray:
    name = config.get("name", "sinus_quad")
    params = config.get("params", {})
    if name == "zero":
        return np.zeros_like(w, dtype=float)
    if name == "sinus_quad":
        a1 = float(params.get("a1", 0.5))
        a2 = float(params.get("a2", 0.3))
        return a1 * np.sin(w) + a2 * (w**2 - 1.0)
    if name == "strong_nonlinear":
        a1 = float(params.get("a1", 0.5))
        a2 = float(params.get("a2", 0.4))
        return a1 * np.sin(2.0 * w) + a2 * w * np.exp(-0.5 * w**2)
    raise ValueError(f"Unknown nuisance m function: {name}")


@register_dgp("predictive_ar1")
class PredictiveAR1DGP:
    """Predictive-regression DGP with stationary alpha-mixing nuisance covariate.

    The simulated nuisance driver is

        W_t = a_W W_{t-1} + eta_t,  |a_W| < 1,

    initialized from its stationary Gaussian distribution. This makes W_t
    strictly stationary and geometrically alpha-mixing. The predictor X_t may
    be stationary, local-to-unity, or unit-root depending on rho_design. The
    predictor initialization is explicit because random-walk burn-in is not a
    stationary device.
    """

    def __init__(self, params: dict):
        self.params = dict(params)
        self.a_w = float(self.params.get("a_w", 0.5))
        self.kappa = float(self.params.get("kappa", 0.0))
        self.xi = float(self.params.get("xi", 0.0))
        self.burnin = int(self.params.get("burnin", 300))
        self.x_initialization = str(self.params.get("x_initialization", "zero_at_sample_start"))
        self.m_config = self.params.get("m", {"name": "sinus_quad", "params": {}})
        self._check_covariance()
        if abs(self.a_w) >= 1.0:
            raise ValueError("a_w must satisfy |a_w| < 1 for stationary W_t.")
        if self.burnin < 0:
            raise ValueError("burnin must be nonnegative.")
        valid_x_initialization = {"zero_at_sample_start", "burnin"}
        if self.x_initialization not in valid_x_initialization:
            raise ValueError(f"x_initialization must be one of {sorted(valid_x_initialization)}.")

    @property
    def covariance(self) -> np.ndarray:
        return np.array(
            [
                [1.0, self.kappa, 0.0],
                [self.kappa, 1.0, self.xi],
                [0.0, self.xi, 1.0],
            ],
            dtype=float,
        )

    def _check_covariance(self) -> None:
        eig = np.linalg.eigvalsh(self.covariance)
        if not np.all(eig > 1e-10):
            raise ValueError(f"Innovation covariance is not positive definite: eigenvalues={eig}")

    def simulate(self, seed: int, T: int, rho_design: dict, beta: float) -> Dataset:
        if T <= 0:
            raise ValueError("T must be positive.")
        rho = _rho_from_design(rho_design, T)
        rng = np.random.default_rng(seed)
        n = T + self.burnin + 1
        innovations = rng.multivariate_normal(np.zeros(3), self.covariance, size=n)
        u_full = innovations[:, 0]
        v_full = innovations[:, 1]
        eta_full = innovations[:, 2]

        x = np.zeros(n, dtype=float)
        w = np.zeros(n, dtype=float)
        w[0] = rng.normal(0.0, np.sqrt(1.0 / (1.0 - self.a_w**2)))
        for t in range(1, n):
            w[t] = self.a_w * w[t - 1] + eta_full[t]

        if self.x_initialization == "burnin":
            for t in range(1, n):
                x[t] = rho * x[t - 1] + v_full[t]
        else:
            sample_start = self.burnin
            x[sample_start] = 0.0
            for t in range(sample_start + 1, n):
                x[t] = rho * x[t - 1] + v_full[t]

        idx = np.arange(self.burnin + 1, self.burnin + T + 1)
        x_lag = x[idx - 1]
        w_lag = w[idx - 1]
        u = u_full[idx]
        m_w = _m_function(w_lag, self.m_config)
        y = m_w + float(beta) * x_lag + u
        meta = {
            "T": int(T),
            "rho_label": rho_design.get("label", str(rho)),
            "rho_value": float(rho),
            "rho_formula": rho_design.get("formula", "fixed"),
            "beta": float(beta),
            "a_w": self.a_w,
            "x_initialization": self.x_initialization,
            "w_process": "stationary_gaussian_ar1",
            "w_stationary": True,
            "w_alpha_mixing": "geometric",
            "kappa": self.kappa,
            "xi": self.xi,
            "m_type": self.m_config.get("name", "unknown"),
        }
        return Dataset(y=y, x_lag=x_lag, w_lag=w_lag, u=u, m_w=m_w, meta=meta)
