import numpy as np
import scipy.optimize as opt
from scipy.special import eval_legendre
from scipy.interpolate import BSpline
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from functools import partial
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict


# =============================================================================
# 1. Data Interfaces and DGP (SOLID Architecture)
# =============================================================================

class DataProvider(ABC):
    """Interface for simulated or empirical data used by the sieve EL estimator."""

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Y: response vector (T,)
            X: current persistent predictor X_t (T,)
            X_lag: lagged persistent predictor X_{t-1} (T,)
            W_lag: lagged stationary exogenous covariate W_{t-1} (T,)
            eps: true predictor innovations eps_t for simulation diagnostics (T,)
        """
        pass


class SimulatedDGP(DataProvider):
    """DGP from profile sieve.tex: Y_t=m(W_{t-1})+beta X_{t-1}+U_t."""

    def __init__(
        self,
        T: int,
        rho: float,
        beta: float,
        phi: float,
        theta: float = 0.0,
        rho_w: float = 0.5,
        m_type: str = "smooth",
        x_w_loading: float = 0.0,
        seed: Optional[int] = None,
    ):
        if T <= 0:
            raise ValueError("T must be positive.")
        if abs(rho_w) >= 1:
            raise ValueError("rho_w must satisfy |rho_w| < 1 for stationarity.")

        self.T = T
        self.rho = rho
        self.beta = beta
        self.phi = phi
        self.theta = theta
        self.rho_w = rho_w
        self.m_type = m_type
        self.x_w_loading = x_w_loading
        self.rng = np.random.default_rng(seed)
        self._data = None

    def _generate_alpha_mixing_W(self, size: int) -> np.ndarray:
        """Stationary AR(1) exogenous covariate, geometrically alpha-mixing."""
        W = np.zeros(size)
        for t in range(1, size):
            W[t] = self.rho_w * W[t - 1] + self.rng.normal(0.0, 1.0)
        return W

    def _m(self, W: np.ndarray) -> np.ndarray:
        """Nuisance component m(W). Use rough variants for K-sensitivity stress tests."""
        if self.m_type == "smooth":
            return 2.0 * (1.0 / (1.0 + np.exp(-5.0 * (W - 1.5))) - 0.5)
        if self.m_type == "rough":
            return 1.2 * np.sin(5.0 * W) + 0.9 * np.sin(11.0 * W) + 0.35 * W**2
        if self.m_type == "mixed":
            smooth = 2.0 * (1.0 / (1.0 + np.exp(-5.0 * (W - 1.5))) - 0.5)
            return smooth + 0.8 * np.sin(9.0 * W)
        raise ValueError(f"Unknown m_type: {self.m_type}")

    def _generate_X(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Persistent predictor X_t = theta + rho X_{t-1} + eps_t."""
        X = np.zeros(size)
        eps = self.rng.normal(0.0, 1.0, size)
        for t in range(1, size):
            X[t] = self.theta + self.rho * X[t - 1] + eps[t]
        return X, eps

    def _simulate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        W_full = self._generate_alpha_mixing_W(self.T + 1)
        X_full, eps_full = self._generate_X(self.T + 1)

        W_lag = W_full[:-1]
        X_lag = X_full[:-1] + self.x_w_loading * W_lag
        X = X_full[1:] + self.x_w_loading * W_full[1:]
        eps = eps_full[1:]

        z = self.rng.normal(0.0, 1.0, self.T)
        U = self.phi * eps + z
        Y = self._m(W_lag) + self.beta * X_lag + U
        return Y, X, X_lag, W_lag, eps

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._data is None:
            self._data = self._simulate_data()
        return self._data


class RealDataProvider(DataProvider):
    """Adapter for empirical vectors Y_t, X_t, and W_t."""

    def __init__(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray):
        if not (len(Y) == len(X) == len(W)):
            raise ValueError("Y, X, and W must have the same length.")
        if len(Y) < 2:
            raise ValueError("At least two observations are required to form lags.")

        self.Y = np.asarray(Y[1:], dtype=float)
        self.X = np.asarray(X[1:], dtype=float)
        self.X_lag = np.asarray(X[:-1], dtype=float)
        self.W_lag = np.asarray(W[:-1], dtype=float)
        self.eps = np.full_like(self.Y, np.nan, dtype=float)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.Y, self.X, self.X_lag, self.W_lag, self.eps


# =============================================================================
# 2. Sieve Basis Strategy Pattern
# =============================================================================

class SieveBasis(ABC):
    """Strategy interface for constructing P_K(W)."""

    def __init__(self, K: int):
        if K <= 0:
            raise ValueError("K must be positive.")
        self.K = K

    @abstractmethod
    def construct(self, W: np.ndarray) -> np.ndarray:
        """Return a T x K basis matrix whose first column is a constant."""
        pass


class LegendreSieve(SieveBasis):
    def construct(self, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        W_min, W_max = np.min(W), np.max(W)
        if W_max > W_min:
            W_norm = 2.0 * (W - W_min) / (W_max - W_min) - 1.0
        else:
            W_norm = np.zeros_like(W)

        P_K = np.zeros((len(W), self.K))
        for k in range(self.K):
            P_K[:, k] = eval_legendre(k, W_norm)
        return P_K


class BSplineSieve(SieveBasis):
    def __init__(self, K: int, degree: int = 3):
        super().__init__(K)
        self.degree = degree

    def construct(self, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        P_K = np.ones((len(W), self.K))
        if self.K == 1:
            return P_K

        n_splines = self.K - 1
        degree = min(self.degree, n_splines - 1)
        n_internal = n_splines - degree - 1

        if n_internal > 0:
            qs = np.linspace(0, 100, n_internal + 2)[1:-1]
            knots_internal = np.percentile(W, qs)
        else:
            knots_internal = []

        W_min, W_max = np.min(W), np.max(W)
        knots = np.concatenate([
            [W_min - 1e-8] * (degree + 1),
            knots_internal,
            [W_max + 1e-8] * (degree + 1),
        ])

        for i in range(n_splines):
            coefs = np.zeros(n_splines)
            coefs[i] = 1.0
            P_K[:, i + 1] = BSpline(knots, coefs, degree, extrapolate=True)(W)
        return P_K


# =============================================================================
# 3. Estimation Core Algorithm
# =============================================================================

class SieveProjection:
    """Linear projection helper for repeated residualization on P_K."""

    @staticmethod
    def coefficients(P_K: np.ndarray, values: np.ndarray) -> np.ndarray:
        coefs, _, _, _ = np.linalg.lstsq(P_K, values, rcond=None)
        return coefs

    @classmethod
    def residualize(cls, P_K: np.ndarray, values: np.ndarray) -> np.ndarray:
        coefs = cls.coefficients(P_K, values)
        return values - P_K @ coefs


class SieveELEstimator:
    """Null-imposed profile sieve empirical likelihood estimator."""

    def __init__(self, basis_strategy: SieveBasis):
        self.basis_strategy = basis_strategy

    def profile_coefficients(
        self,
        Y: np.ndarray,
        X_lag: np.ndarray,
        P_K: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        """c_hat_K(beta) = (P'P)^{-1}P'(Y - beta X_lag)."""
        return SieveProjection.coefficients(P_K, Y - beta * X_lag)

    def profile_residual(
        self,
        Y: np.ndarray,
        X_lag: np.ndarray,
        P_K: np.ndarray,
        beta: float,
    ) -> np.ndarray:
        """u_hat(beta) = M_K(Y - beta X_lag)."""
        return SieveProjection.residualize(P_K, Y - beta * X_lag)

    def raw_weight(self, X_lag: np.ndarray) -> np.ndarray:
        """Bounded saturated weight w(X_{t-1})."""
        std_x = np.std(X_lag)
        c_w = std_x if std_x > 0 else 1.0
        return np.tanh(X_lag / c_w)

    def orthogonalized_weight(self, X_lag: np.ndarray, P_K: np.ndarray) -> np.ndarray:
        """w_c = M_K w, the same sieve projection used for the nuisance."""
        return SieveProjection.residualize(P_K, self.raw_weight(X_lag))

    def moment_values(
        self,
        Y: np.ndarray,
        X_lag: np.ndarray,
        W_lag: np.ndarray,
        beta: float,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        P_K = self.basis_strategy.construct(W_lag)
        u_hat = self.profile_residual(Y, X_lag, P_K, beta)
        w_c = self.orthogonalized_weight(X_lag, P_K)
        Z = u_hat * w_c
        diagnostics = {"P_K": P_K, "u_hat": u_hat, "w_c": w_c}
        return Z, diagnostics

    @staticmethod
    def empirical_likelihood_statistic(Z: np.ndarray) -> float:
        Z = np.asarray(Z, dtype=float)
        max_Z, min_Z = np.max(Z), np.min(Z)

        if np.allclose(Z, 0.0):
            return 0.0
        if not (min_Z < 0.0 < max_Z):
            return 1e6

        eps = 1e-10
        lb = -1.0 / max_Z + eps
        ub = -1.0 / min_Z - eps

        def lambda_equation(lam: float) -> float:
            return np.sum(Z / (1.0 + lam * Z))

        try:
            res = opt.root_scalar(lambda_equation, bracket=[lb, ub], method="brentq")
        except ValueError:
            return 1e6

        return float(2.0 * np.sum(np.log1p(res.root * Z)))

    def fit(self, data: DataProvider, beta_0: float, return_diagnostics: bool = False):
        """Evaluate ell_T(beta_0) using the null-imposed profile score."""
        Y, _, X_lag, W_lag, _ = data.get_data()
        Z, diagnostics = self.moment_values(Y, X_lag, W_lag, beta_0)
        ell_T = self.empirical_likelihood_statistic(Z)

        if return_diagnostics:
            diagnostics = {**diagnostics, "Z": Z, "ell_T": ell_T}
            return ell_T, diagnostics
        return ell_T


# =============================================================================
# 4. Monte Carlo Orchestration Module
# =============================================================================

class MonteCarloRunner:
    """Manages simulation experiments and keeps estimator construction isolated."""

    def __init__(self, iterations: int, T: int, K: int, n_jobs: Optional[int] = 1):
        self.iterations = iterations
        self.T = T
        self.K = K
        self.n_jobs = n_jobs

    @staticmethod
    def _run_single_iteration(
        seed: int,
        T: int,
        rho: float,
        beta_true: float,
        beta_0: float,
        phi: float,
        K: int,
        theta: float = 0.0,
    ) -> Tuple[float, float]:
        dgp = SimulatedDGP(T, rho, beta_true, phi, theta=theta, seed=seed)
        estimator = SieveELEstimator(LegendreSieve(K))
        ell_T = estimator.fit(dgp, beta_0)

        Y, _, X_lag, W_lag, _ = dgp.get_data()
        X_mat = np.column_stack((np.ones(T), X_lag, W_lag))
        coefs, _, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
        residual = Y - X_mat @ coefs
        dof = max(T - X_mat.shape[1], 1)
        sigma2 = np.sum(residual**2) / dof
        var_beta = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)[1, 1]
        t_stat_ols = (coefs[1] - beta_0) / np.sqrt(var_beta) if var_beta > 0 else np.nan
        return float(ell_T), float(t_stat_ols)

    def simulate(
        self,
        rho: float,
        beta_true: float,
        beta_0: float,
        phi: float,
        theta: float = 0.0,
    ) -> Tuple[float, float]:
        seeds = np.random.default_rng().integers(0, 1_000_000, size=self.iterations)
        func = partial(
            self._run_single_iteration,
            T=self.T,
            rho=rho,
            beta_true=beta_true,
            beta_0=beta_0,
            phi=phi,
            K=self.K,
            theta=theta,
        )

        if self.n_jobs == 1:
            results = list(map(func, seeds))
        else:
            process_count = self.n_jobs or mp.cpu_count()
            with mp.Pool(process_count) as pool:
                results = pool.map(func, seeds)

        el_stats = np.array([r[0] for r in results])
        ols_t_stats = np.array([r[1] for r in results])

        el_rejections = np.mean(el_stats > 3.841)
        ols_rejections = np.mean(np.abs(ols_t_stats) > 1.96)
        return float(el_rejections), float(ols_rejections)


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
        el_sizes, ols_sizes = [], []

        print(f"Running size simulations (iterations={runner.iterations}, T={runner.T})...")
        for rho in rhos:
            el_rej, ols_rej = runner.simulate(rho=rho, beta_true=0.0, beta_0=0.0, phi=0.5)
            el_sizes.append(el_rej)
            ols_sizes.append(ols_rej)
            print(f"  rho={rho}: EL Size = {el_rej:.3f}, OLS Size = {ols_rej:.3f}")

        plt.figure(figsize=(8, 6))
        plt.plot(rhos, el_sizes, "k-o", label="Profile Sieve-EL")
        plt.plot(rhos, ols_sizes, color="gray", linestyle="--", marker="s", label="Standard OLS")
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
        betas = np.linspace(0.0, 0.1, 11)
        el_powers, ols_powers = [], []

        print(f"\nRunning power simulations (iterations={runner.iterations}, T={runner.T}, rho={rho})...")
        for beta in betas:
            el_rej, ols_rej = runner.simulate(rho=rho, beta_true=beta, beta_0=0.0, phi=0.5)
            el_powers.append(el_rej)
            ols_powers.append(ols_rej)
            print(f"  beta={beta:.3f}: EL Power = {el_rej:.3f}, OLS Power = {ols_rej:.3f}")

        plt.figure(figsize=(8, 6))
        plt.plot(betas, el_powers, "k-o", label="Profile Sieve-EL")
        plt.plot(betas, ols_powers, color="gray", linestyle="--", marker="s", label="Standard OLS")
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

    T_sim = 500
    K_sim = int(np.round(T_sim ** (1 / 3)))
    iterations_sim = 200
    run_full_monte_carlo = False

    dgp = SimulatedDGP(T_sim, rho=0.99, beta=0.0, phi=0.5, seed=42)
    estimator = SieveELEstimator(LegendreSieve(K_sim))
    ell_T, diagnostics = estimator.fit(dgp, beta_0=0.0, return_diagnostics=True)
    P_K = diagnostics["P_K"]
    w_c = diagnostics["w_c"]
    u_hat = diagnostics["u_hat"]
    Y, _, X_lag, _, _ = dgp.get_data()

    weight_ortho_check = P_K.T @ w_c
    profile_ortho_check = P_K.T @ u_hat
    direct_profile_check = SieveProjection.residualize(P_K, Y - 0.0 * X_lag)

    print("K_sim:", K_sim)
    print("EL statistic at beta_0=0:", ell_T)
    print("max |P_K' w_c|:", np.max(np.abs(weight_ortho_check)))
    print("max |P_K' u_hat|:", np.max(np.abs(profile_ortho_check)))
    assert np.allclose(weight_ortho_check, 0.0, atol=1e-10)
    assert np.allclose(profile_ortho_check, 0.0, atol=1e-10)
    assert np.allclose(u_hat, direct_profile_check)

    if run_full_monte_carlo:
        t0 = time.time()
        runner = MonteCarloRunner(iterations=iterations_sim, T=T_sim, K=K_sim)
        AcademicPlotter.plot_size(runner)
        AcademicPlotter.plot_power(runner, rho=0.99)
        print(f"Total time elapsed: {time.time() - t0:.2f} seconds")
