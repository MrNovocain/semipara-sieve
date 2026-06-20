from __future__ import annotations

import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import BSpline
from scipy.special import eval_legendre

# =============================================================================
# 1. Data Interfaces and DGP (SOLID Architecture)
# =============================================================================


class NuisanceFunction(ABC):
    """Strategy interface for the structural nuisance function m(W)."""

    name: str = "abstract"

    @abstractmethod
    def __call__(self, W: np.ndarray) -> np.ndarray:
        """Evaluate m(W) elementwise."""
        pass


@dataclass(frozen=True)
class SmoothLogisticImpact(NuisanceFunction):
    """Smooth threshold-like m(W), useful when profile-sieve smoothness is plausible."""

    amplitude: float = 2.0
    slope: float = 5.0
    threshold: float = 1.5
    name: str = "smooth_logistic"

    def __call__(self, W: np.ndarray) -> np.ndarray:
        return self.amplitude * (1.0 / (1.0 + np.exp(-self.slope * (W - self.threshold))) - 0.5)


@dataclass(frozen=True)
class StepImpact(NuisanceFunction):
    """Non-smooth jump m(W), included to stress-test future non-smooth sieve designs."""

    low: float = -1.0
    high: float = 1.0
    threshold: float = 1.5
    name: str = "step"

    def __call__(self, W: np.ndarray) -> np.ndarray:
        return np.where(W >= self.threshold, self.high, self.low)


@dataclass(frozen=True)
class PiecewiseLinearImpact(NuisanceFunction):
    """Continuous but kinked m(W), between the smooth and discontinuous designs."""

    left_slope: float = 0.25
    right_slope: float = 1.25
    kink: float = 0.0
    name: str = "piecewise_linear"

    def __call__(self, W: np.ndarray) -> np.ndarray:
        return np.where(
            W < self.kink,
            self.left_slope * (W - self.kink),
            self.right_slope * (W - self.kink),
        )


class NuisanceFunctionFactory:
    """Small switcher/factory so Monte Carlo designs can swap m(W) without touching the DGP."""

    _REGISTRY: Dict[str, Callable[..., NuisanceFunction]] = {
        "smooth_logistic": SmoothLogisticImpact,
        "step": StepImpact,
        "piecewise_linear": PiecewiseLinearImpact,
    }

    @classmethod
    def create(cls, name: str = "smooth_logistic", **kwargs) -> NuisanceFunction:
        try:
            return cls._REGISTRY[name](**kwargs)
        except KeyError as exc:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(f"Unknown nuisance function '{name}'. Available choices: {available}") from exc

    @classmethod
    def register(cls, name: str, builder: Callable[..., NuisanceFunction]) -> None:
        """Register a new m(W) strategy (for example, future non-smooth alternatives)."""
        cls._REGISTRY[name] = builder


class DataProvider(ABC):
    """Abstract base class for providing data to the estimator."""

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Y (np.ndarray): Dependent variable (T,)
            X (np.ndarray): Persistent predictor (T,)
            X_lag (np.ndarray): Lagged predictor (T,)
            W_lag (np.ndarray): Lagged exogenous covariate (T,)
            eps (np.ndarray): True innovations of X (T,) [NaN for real data]
        """
        pass


class SimulatedDGP(DataProvider):
    """Monte Carlo DGP with modular m(W), alpha-mixing W, and persistent X."""

    def __init__(
        self,
        T: int,
        rho: float,
        beta: float,
        phi: float,
        theta: float = 0.0,
        rho_w: float = 0.5,
        nuisance_function: Optional[NuisanceFunction] = None,
        seed: Optional[int] = None,
    ):
        if abs(rho_w) >= 1:
            raise ValueError("rho_w must satisfy |rho_w| < 1 so W is geometrically alpha-mixing.")
        self.T = T
        self.rho = rho
        self.beta = beta
        self.phi = phi
        self.theta = theta
        self.rho_w = rho_w
        self.m_function = nuisance_function or NuisanceFunctionFactory.create("smooth_logistic")
        self.rng = np.random.default_rng(seed)
        self._cached_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    def _generate_alpha_mixing_W(self, size: int) -> np.ndarray:
        W = np.zeros(size)
        for t in range(1, size):
            W[t] = self.rho_w * W[t - 1] + self.rng.normal(0, 1)
        return W

    def _m(self, W: np.ndarray) -> np.ndarray:
        return self.m_function(W)

    def _generate_X(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        X = np.zeros(size)
        eps = self.rng.normal(0, 1, size)
        for t in range(1, size):
            X[t] = self.theta + self.rho * X[t - 1] + eps[t]
        return X, eps

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_data is not None:
            return self._cached_data

        W_full = self._generate_alpha_mixing_W(self.T + 1)
        X_full, eps_full = self._generate_X(self.T + 1)

        W_lag = W_full[:-1]
        X_lag = X_full[:-1]
        eps = eps_full[1:]
        X = X_full[1:]

        z = self.rng.normal(0, 1, self.T)
        U = self.phi * eps + z
        Y = self._m(W_lag) + self.beta * X_lag + U

        self._cached_data = (Y, X, X_lag, W_lag, eps)
        return self._cached_data


class RealDataProvider(DataProvider):
    """Adapter for empirical data; innovations are intentionally not used in the EL statistic."""

    def __init__(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray):
        self.Y = Y[1:]
        self.X = X[1:]
        self.X_lag = X[:-1]
        self.W_lag = W[:-1]
        self.eps = np.full_like(self.Y, np.nan)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.Y, self.X, self.X_lag, self.W_lag, self.eps

# =============================================================================
# 2. Sieve Basis Strategy Pattern
# =============================================================================


class SieveBasis(ABC):
    """Strategy interface for constructing the Sieve Basis."""

    def __init__(self, K: int):
        self.K = K

    @abstractmethod
    def construct(self, W: np.ndarray) -> np.ndarray:
        """Returns the basis matrix P_K of shape (T, K)."""
        pass


class LegendreSieve(SieveBasis):
    def construct(self, W: np.ndarray) -> np.ndarray:
        W_min, W_max = np.min(W), np.max(W)
        if W_max > W_min:
            W_norm = 2 * (W - W_min) / (W_max - W_min) - 1
        else:
            W_norm = W

        P_K = np.zeros((len(W), self.K))
        for k in range(self.K):
            P_K[:, k] = eval_legendre(k, W_norm)
        return P_K


class BSplineSieve(SieveBasis):
    def __init__(self, K: int, degree: int = 3):
        super().__init__(K)
        self.degree = degree

    def construct(self, W: np.ndarray) -> np.ndarray:
        T = len(W)
        K = self.K
        degree = min(self.degree, K - 1)
        n_internal = K - degree - 1

        if n_internal > 0:
            qs = np.linspace(0, 100, n_internal + 2)[1:-1]
            knots_internal = np.percentile(W, qs)
        else:
            knots_internal = []

        W_min, W_max = np.min(W), np.max(W)
        knots = np.concatenate([
            [W_min - 1e-4] * (degree + 1),
            knots_internal,
            [W_max + 1e-4] * (degree + 1),
        ])

        P_K = np.zeros((T, K))
        for i in range(K):
            coefs = np.zeros(K)
            coefs[i] = 1.0
            spl = BSpline(knots, coefs, degree, extrapolate=True)
            P_K[:, i] = spl(W)

        return P_K

# =============================================================================
# 3. Estimation Core Algorithm
# =============================================================================


class SieveELEstimator:
    """Implements the null-imposed profile Sieve-EL algorithm in profile sieve.tex."""

    def __init__(self, basis_strategy: SieveBasis):
        self.basis_strategy = basis_strategy

    def profile_nuisance(self, Y: np.ndarray, X_lag: np.ndarray, P_K: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute c_hat_K(beta) and u_hat(beta) = M_K(Y - beta X_lag)."""
        y_minus_beta_x = Y - beta * X_lag
        c_hat, _, _, _ = np.linalg.lstsq(P_K, y_minus_beta_x, rcond=None)
        u_hat = y_minus_beta_x - P_K @ c_hat
        return c_hat, u_hat

    def compute_orthogonalized_weights(self, X_lag: np.ndarray, P_K: np.ndarray) -> np.ndarray:
        """Compute w^c = M_K w by residualizing the raw predictor weight on the same sieve basis."""
        std_x = np.std(X_lag)
        cw = std_x if std_x > 0 else 1.0
        w_raw = np.tanh(X_lag / cw)

        gamma_hat, _, _, _ = np.linalg.lstsq(P_K, w_raw, rcond=None)
        return w_raw - P_K @ gamma_hat

    def compute_el_statistic_from_scores(self, Z: np.ndarray) -> float:
        """Scalar empirical likelihood ratio statistic for mean-zero scores Z_t."""
        max_Z, min_Z = np.max(Z), np.min(Z)

        if max_Z == 0 and min_Z == 0:
            return 0.0
        if max_Z * min_Z > 0:
            return 1e6

        lb = -1.0 / max_Z + 1e-8
        ub = -1.0 / min_Z - 1e-8

        def el_gradient(lam):
            return np.sum(Z / (1.0 + lam * Z))

        try:
            res = opt.root_scalar(el_gradient, bracket=[lb, ub], method="brentq")
            return 2.0 * np.sum(np.log(1.0 + res.root * Z))
        except ValueError:
            return 1e6

    def fit(self, data: DataProvider, beta_0: float) -> float:
        """Return ell_T(beta_0) using profile residuals and sieve-orthogonalized weights."""
        Y, _, X_lag, W_lag, _ = data.get_data()

        P_K = self.basis_strategy.construct(W_lag)
        _, u_hat = self.profile_nuisance(Y, X_lag, P_K, beta_0)
        w_c = self.compute_orthogonalized_weights(X_lag, P_K)
        Z = u_hat * w_c
        return self.compute_el_statistic_from_scores(Z)

    def fit_robust_score(self, data: DataProvider, beta_0: float) -> float:
        """Return a self-normalized profile score statistic using the same profile score Z_t."""
        Y, _, X_lag, W_lag, _ = data.get_data()

        P_K = self.basis_strategy.construct(W_lag)
        _, u_hat = self.profile_nuisance(Y, X_lag, P_K, beta_0)
        w_c = self.compute_orthogonalized_weights(X_lag, P_K)
        Z = u_hat * w_c

        den = np.sum(Z ** 2)
        if den == 0:
            return 0.0
        return (np.sum(Z) ** 2) / den

# =============================================================================
# 4. Monte Carlo Orchestration Module
# =============================================================================


class MonteCarloRunner:
    """Manages simulation experiments while keeping DGP, basis, and m(W) choices modular."""

    def __init__(self, iterations: int, T: int, K: int, m_name: str = "smooth_logistic", m_kwargs: Optional[Dict] = None):
        self.iterations = iterations
        self.T = T
        self.K = K
        self.m_name = m_name
        self.m_kwargs = m_kwargs or {}

    @staticmethod
    def _run_single_iteration(
        seed: int,
        T: int,
        rho: float,
        beta_true: float,
        beta_0: float,
        phi: float,
        K: int,
        m_name: str,
        m_kwargs: Dict,
        theta: float = 0.0,
    ) -> Tuple[float, float, float]:
        m_function = NuisanceFunctionFactory.create(m_name, **m_kwargs)
        dgp = SimulatedDGP(T, rho, beta_true, phi, theta=theta, nuisance_function=m_function, seed=seed)
        basis = LegendreSieve(K)
        estimator = SieveELEstimator(basis)

        l_T = estimator.fit(dgp, beta_0)
        s_T = estimator.fit_robust_score(dgp, beta_0)

        Y, _, X_lag, W_lag, _ = dgp.get_data()
        X_mat = np.column_stack((np.ones(T), X_lag, W_lag))
        coefs, residuals, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
        var_res = np.var(Y - X_mat @ coefs) if len(residuals) == 0 else residuals[0] / max(T - X_mat.shape[1], 1)
        inv_xx = np.linalg.pinv(X_mat.T @ X_mat)
        var_beta_ols = var_res * inv_xx[1, 1]
        t_stat_ols = (coefs[1] - beta_0) / np.sqrt(var_beta_ols)

        return l_T, s_T, t_stat_ols

    def simulate(self, rho: float, beta_true: float, beta_0: float, phi: float, theta: float = 0.0) -> Tuple[float, float, float]:
        seeds = np.random.default_rng().integers(0, 1_000_000, size=self.iterations)
        func = partial(
            self._run_single_iteration,
            T=self.T,
            rho=rho,
            beta_true=beta_true,
            beta_0=beta_0,
            phi=phi,
            K=self.K,
            m_name=self.m_name,
            m_kwargs=self.m_kwargs,
            theta=theta,
        )

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(func, seeds)

        el_stats = np.array([r[0] for r in results])
        score_stats = np.array([r[1] for r in results])
        ols_t_stats = np.array([r[2] for r in results])

        el_rejections = np.mean(el_stats > 3.841)
        score_rejections = np.mean(score_stats > 3.841)
        ols_rejections = np.mean(np.abs(ols_t_stats) > 1.96)

        return el_rejections, score_rejections, ols_rejections

# =============================================================================
# 5. Visualization & Reporting Module
# =============================================================================


class AcademicPlotter:
    @staticmethod
    def set_style():
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
        })

    @classmethod
    def plot_size(cls, runner: MonteCarloRunner):
        cls.set_style()
        rhos = [0.90, 0.95, 0.99, 0.999]
        el_sizes, score_sizes, ols_sizes = [], [], []

        print(f"Running Size simulations (iterations={runner.iterations}, T={runner.T})...")
        for rho in rhos:
            el_rej, score_rej, ols_rej = runner.simulate(rho=rho, beta_true=0.0, beta_0=0.0, phi=0.5)
            el_sizes.append(el_rej)
            score_sizes.append(score_rej)
            ols_sizes.append(ols_rej)
            print(f"  rho={rho}: EL Size = {el_rej:.3f}, Score Size = {score_rej:.3f}, OLS Size = {ols_rej:.3f}")

        plt.figure(figsize=(8, 6))
        plt.plot(rhos, el_sizes, "k-o", label="Orthogonalized Sieve-EL")
        plt.plot(rhos, score_sizes, "k--d", label="Robust Sieve Score")
        plt.plot(rhos, ols_sizes, "gray", linestyle=":", marker="s", label="Standard OLS")
        plt.axhline(y=0.05, color="k", linestyle=":", label="Nominal 5% Level")
        plt.xlabel("Predictor Persistence ($\\rho$)")
        plt.ylabel("Empirical Size (Rejection Rate)")
        plt.title("Size Distortion under Persistent Predictors")
        plt.legend()
        plt.tight_layout()
        plt.savefig("size_distortion.png", dpi=300)
        print("Saved plot to size_distortion.png")

    @classmethod
    def plot_power(cls, runner: MonteCarloRunner, rho: float = 0.99):
        cls.set_style()
        betas = np.linspace(0.0, 0.5, 11)
        el_powers, score_powers, ols_powers = [], [], []

        print(f"\nRunning Power simulations (iterations={runner.iterations}, T={runner.T}, rho={rho})...")
        for beta in betas:
            el_rej, score_rej, ols_rej = runner.simulate(rho=rho, beta_true=beta, beta_0=0.0, phi=0.5)
            el_powers.append(el_rej)
            score_powers.append(score_rej)
            ols_powers.append(ols_rej)
            print(f"  beta={beta:.3f}: EL Power = {el_rej:.3f}, Score Power = {score_rej:.3f}")

        plt.figure(figsize=(8, 6))
        plt.plot(betas, el_powers, "k-o", label="Orthogonalized Sieve-EL")
        plt.plot(betas, score_powers, "k--d", label="Robust Sieve Score")
        plt.plot(betas, ols_powers, "gray", linestyle=":", label="Standard OLS")
        plt.axhline(y=0.05, color="k", linestyle=":", label="Nominal 5% Level")
        plt.xlabel("True Predictability ($\\beta$)")
        plt.ylabel("Empirical Power (Rejection Rate)")
        plt.title(f"Power Curve ($\\rho = {rho}$)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("power_curve.png", dpi=300)
        print("Saved plot to power_curve.png")


if __name__ == "__main__":
    mp.freeze_support()

    # Configuration
    T_sim = 500
    K_sim = int(np.round(T_sim ** (1 / 3)))
    iterations_sim = 1000  # Set lower for quick testing, increase to 1000+ for paper

    # 1. Verification of Orthogonalization
    m_design = NuisanceFunctionFactory.create("smooth_logistic")  # switch to "step" for a non-smooth m(W) stress test
    dgp = SimulatedDGP(T_sim, rho=0.99, beta=0.0, phi=0.5, nuisance_function=m_design, seed=42)
    Y, X, X_lag, W_lag, _ = dgp.get_data()
    basis = LegendreSieve(K_sim)
    P_K = basis.construct(W_lag)
    estimator = SieveELEstimator(basis)
    w_c = estimator.compute_orthogonalized_weights(X_lag, P_K)
    ortho_check = np.sum(P_K.T * w_c, axis=1)
    print("Verification of Step 4 Orthogonalization (sum P_K * w_c):")
    print(ortho_check)
    assert np.allclose(ortho_check, 0, atol=1e-10), "Orthogonalization mathematically failed!"
    print("Orthogonalization structurally verified via Sieve weight projection.\n")

    # 2. Run Monte Carlo
    t0 = time.time()
    runner = MonteCarloRunner(iterations=iterations_sim, T=T_sim, K=K_sim, m_name=m_design.name)
    AcademicPlotter.plot_size(runner)
    AcademicPlotter.plot_power(runner, rho=0.99)
    print(f"Total time elapsed: {time.time() - t0:.2f} seconds")
