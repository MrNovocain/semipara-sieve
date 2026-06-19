import numpy as np
import scipy.optimize as opt
from scipy.special import eval_legendre
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
    """
    Abstract base class for providing data to the estimator.
    Wrapped to allow easy replacement with real empirical data.
    """
    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Y (np.ndarray): Dependent variable (T,)
            X (np.ndarray): Persistent predictor (T,)
            X_lag (np.ndarray): Lagged predictor (T,)
            W_lag (np.ndarray): Lagged exogenous weather covariate (T,)
            eps (np.ndarray): True innovations of X (T,) [None for real data]
        """
        pass

class SimulatedDGP(DataProvider):
    """
    Data Generating Process for Monte Carlo simulations.
    Generates alpha-mixing weather covariates and highly persistent predictors.
    """
    def __init__(self, T: int, rho: float, beta: float, phi: float, theta: float = 0.0, seed: Optional[int] = None):
        self.T = T
        self.rho = rho
        self.beta = beta
        self.phi = phi
        self.theta = theta
        if seed is not None:
            np.random.seed(seed)

    def _generate_alpha_mixing_W(self, size: int) -> np.ndarray:
        """
        Generate alpha-mixing exogenous weather covariate.
        A stationary AR(1) process with |rho_w| < 1 is geometrically alpha-mixing.
        """
        W = np.zeros(size)
        rho_w = 0.5
        for t in range(1, size):
            W[t] = rho_w * W[t-1] + np.random.normal(0, 1)
        return W

    def _m(self, W: np.ndarray) -> np.ndarray:
        """Nonlinear climate impact function (threshold-like)."""
        return 2.0 * (1 / (1 + np.exp(-5 * (W - 1.5))) - 0.5)

    def _generate_X(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate highly persistent financial predictor."""
        X = np.zeros(size)
        eps = np.random.normal(0, 1, size)
        for t in range(1, size):
            X[t] = self.theta + self.rho * X[t-1] + eps[t]
        return X, eps

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Generate T+1 periods to extract t and t-1
        W_full = self._generate_alpha_mixing_W(self.T + 1)
        X_full, eps_full = self._generate_X(self.T + 1)
        
        W_lag = W_full[:-1]
        X_lag = X_full[:-1]
        eps = eps_full[1:] # eps_t
        X = X_full[1:]     # X_t
        
        z = np.random.normal(0, 1, self.T)
        U = self.phi * eps + z
        Y = self._m(W_lag) + self.beta * X_lag + U
        
        return Y, X, X_lag, W_lag, eps

class RealDataProvider(DataProvider):
    """
    Placeholder class for when real empirical data is injected.
    """
    def __init__(self, Y: np.ndarray, X: np.ndarray, W: np.ndarray):
        self.Y = Y[1:]
        self.X = X[1:]
        self.X_lag = X[:-1]
        self.W_lag = W[:-1]
        self.eps = np.full_like(self.Y, np.nan) # True innovations unknown

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
        # Normalize W to [-1, 1] for Legendre numerical stability
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
    def construct(self, W: np.ndarray) -> np.ndarray:
        """Placeholder for B-Spline implementation to expand modularity."""
        raise NotImplementedError("B-Spline basis to be implemented.")

# =============================================================================
# 3. Estimation Core Algorithm
# =============================================================================

class SieveELEstimator:
    """
    Encapsulates the 5-step estimation algorithm for the Orthogonalized Sieve-EL.
    """
    def __init__(self, basis_strategy: SieveBasis):
        self.basis_strategy = basis_strategy

    def step1_estimate_ar1(self, X: np.ndarray, X_lag: np.ndarray) -> np.ndarray:
        T = len(X)
        X_mat = np.column_stack((np.ones(T), X_lag))
        coefs, _, _, _ = np.linalg.lstsq(X_mat, X, rcond=None)
        return X - X_mat @ coefs

    def step2_joint_sieve_ols(self, Y: np.ndarray, P_K: np.ndarray, X_lag: np.ndarray, eps_hat: np.ndarray) -> np.ndarray:
        Lambda = np.column_stack((P_K, X_lag, eps_hat))
        coefs, _, _, _ = np.linalg.lstsq(Lambda, Y, rcond=None)
        return coefs[:P_K.shape[1]]

    def step3_compute_adjusted_returns(self, Y: np.ndarray, P_K: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
        return Y - P_K @ c_hat

    def step4_compute_orthogonalized_weights(self, X_lag: np.ndarray, P_K: np.ndarray) -> np.ndarray:
        std_x = np.std(X_lag)
        cw = std_x if std_x > 0 else 1.0
        w_raw = np.tanh(X_lag / cw)
        
        gamma_hat, _, _, _ = np.linalg.lstsq(P_K, w_raw, rcond=None)
        return w_raw - P_K @ gamma_hat

    def step5_compute_el_statistic(self, Y_adj: np.ndarray, X_lag: np.ndarray, w_c: np.ndarray, beta_0: float) -> float:
        Z = (Y_adj - beta_0 * X_lag) * w_c
        max_Z, min_Z = np.max(Z), np.min(Z)
        
        # Infinite test statistic if 0 not in convex hull
        if max_Z * min_Z > 0:
            return 1e6 
            
        lb = -1.0 / max_Z + 1e-8
        ub = -1.0 / min_Z - 1e-8
        
        def el_gradient(lam):
            return np.sum(Z / (1 + lam * Z))
            
        try:
            res = opt.root_scalar(el_gradient, bracket=[lb, ub], method='brentq')
            return 2 * np.sum(np.log(1 + res.root * Z))
        except ValueError:
            return 1e6

    def fit(self, data: DataProvider, beta_0: float) -> float:
        """Executes the full pipeline and returns the EL statistic."""
        Y, X, X_lag, W_lag, _ = data.get_data()
        
        eps_hat = self.step1_estimate_ar1(X, X_lag)
        P_K = self.basis_strategy.construct(W_lag)
        c_hat = self.step2_joint_sieve_ols(Y, P_K, X_lag, eps_hat)
        Y_adj = self.step3_compute_adjusted_returns(Y, P_K, c_hat)
        w_c = self.step4_compute_orthogonalized_weights(X_lag, P_K)
        return self.step5_compute_el_statistic(Y_adj, X_lag, w_c, beta_0)

    def fit_robust_score(self, data: DataProvider, beta_0: float) -> float:
        """Executes the pipeline and returns the Robust Sieve Score statistic."""
        Y, X, X_lag, W_lag, _ = data.get_data()
        
        eps_hat = self.step1_estimate_ar1(X, X_lag)
        P_K = self.basis_strategy.construct(W_lag)
        
        # Step 2: Unconstrained estimation to get residuals
        Lambda = np.column_stack((P_K, X_lag, eps_hat))
        coefs, _, _, _ = np.linalg.lstsq(Lambda, Y, rcond=None)
        c_hat = coefs[:P_K.shape[1]]
        beta_hat = coefs[P_K.shape[1]]
        phi_hat = coefs[P_K.shape[1] + 1]
        
        # Step 3: Compute unconstrained residuals (strictly locked to innovation variance)
        U_hat = Y - P_K @ c_hat - beta_hat * X_lag
        
        # Step 4: Compute adjusted returns and weights
        Y_adj = Y - P_K @ c_hat
        w_c = self.step4_compute_orthogonalized_weights(X_lag, P_K)
        
        # Step 5: Compute Robust Score
        Z = (Y_adj - beta_0 * X_lag) * w_c
        num = np.sum(Z)
        den = np.sum((U_hat * w_c) ** 2)
        
        if den == 0:
            return 0.0
            
        return (num ** 2) / den

# =============================================================================
# 4. Monte Carlo Orchestration Module
# =============================================================================

class MonteCarloRunner:
    """Manages the execution of simulation experiments in parallel."""
    def __init__(self, iterations: int, T: int, K: int):
        self.iterations = iterations
        self.T = T
        self.K = K

    @staticmethod
    def _run_single_iteration(seed: int, T: int, rho: float, beta_true: float, beta_0: float, phi: float, K: int, theta: float = 0.0) -> Tuple[float, float, float]:
        dgp = SimulatedDGP(T, rho, beta_true, phi, theta=theta, seed=seed)
        basis = LegendreSieve(K)
        estimator = SieveELEstimator(basis)
        
        # Compute Sieve-EL stat
        l_T = estimator.fit(dgp, beta_0)
        
        # Compute Robust Sieve Score stat
        s_T = estimator.fit_robust_score(dgp, beta_0)
        
        # Compute Standard OLS t-stat for comparison
        Y, _, X_lag, W_lag, _ = dgp.get_data()
        X_mat = np.column_stack((np.ones(T), X_lag, W_lag))
        coefs, residuals, _, _ = np.linalg.lstsq(X_mat, Y, rcond=None)
        var_res = np.var(residuals)
        inv_xx = np.linalg.inv(X_mat.T @ X_mat)
        var_beta_ols = var_res * inv_xx[1, 1]
        t_stat_ols = (coefs[1] - beta_0) / np.sqrt(var_beta_ols)
        
        return l_T, s_T, t_stat_ols

    def simulate(self, rho: float, beta_true: float, beta_0: float, phi: float, theta: float = 0.0) -> Tuple[float, float, float]:
        seeds = np.random.randint(0, 1000000, size=self.iterations)
        func = partial(self._run_single_iteration, T=self.T, rho=rho, beta_true=beta_true, beta_0=beta_0, phi=phi, K=self.K, theta=theta)
        
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(func, seeds)
            
        el_stats = np.array([r[0] for r in results])
        score_stats = np.array([r[1] for r in results])
        ols_t_stats = np.array([r[2] for r in results])
        
        el_rejections = np.mean(el_stats > 3.841) # Chi^2_1 at 5%
        score_rejections = np.mean(score_stats > 3.841) # Chi^2_1 at 5%
        ols_rejections = np.mean(np.abs(ols_t_stats) > 1.96) # Standard Normal at 5%
        
        return el_rejections, score_rejections, ols_rejections

# =============================================================================
# 5. Visualization & Reporting Module
# =============================================================================

class AcademicPlotter:
    @staticmethod
    def set_style():
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
            'axes.grid': False,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'lines.linewidth': 2.0,
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'legend.fontsize': 12
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
        plt.plot(rhos, el_sizes, 'k-o', label='Orthogonalized Sieve-EL')
        plt.plot(rhos, score_sizes, 'k--d', label='Robust Sieve Score')
        plt.plot(rhos, ols_sizes, 'gray', linestyle=':', marker='s', label='Standard OLS')
        plt.axhline(y=0.05, color='k', linestyle=':', label='Nominal 5% Level')
        plt.xlabel('Predictor Persistence ($\\rho$)')
        plt.ylabel('Empirical Size (Rejection Rate)')
        plt.title('Size Distortion under Persistent Predictors')
        plt.legend()
        plt.tight_layout()
        plt.savefig('size_distortion.png', dpi=300)
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
        plt.plot(betas, el_powers, 'k-o', label='Orthogonalized Sieve-EL')
        plt.plot(betas, score_powers, 'k--d', label='Robust Sieve Score')
        plt.plot(betas, ols_powers, 'gray', linestyle=':', label='Standard OLS')
        plt.axhline(y=0.05, color='k', linestyle=':', label='Nominal 5% Level')
        plt.xlabel('True Predictability ($\\beta$)')
        plt.ylabel('Empirical Power (Rejection Rate)')
        plt.title(f'Power Curve ($\\rho = {rho}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('power_curve.png', dpi=300)
        print("Saved plot to power_curve.png")

if __name__ == "__main__":
    mp.freeze_support()
    
    # Configuration
    T_sim = 500
    K_sim = int(np.round(T_sim**(1/3)))
    iterations_sim = 1000 # Set lower for quick testing, increase to 1000+ for paper
    
    # 1. Verification of Orthogonalization
    dgp = SimulatedDGP(T_sim, rho=0.99, beta=0.0, phi=0.5, seed=42)
    Y, X, X_lag, W_lag, _ = dgp.get_data()
    basis = LegendreSieve(K_sim)
    P_K = basis.construct(W_lag)
    estimator = SieveELEstimator(basis)
    w_c = estimator.step4_compute_orthogonalized_weights(X_lag, P_K)
    ortho_check = np.sum(P_K.T * w_c, axis=1)
    print("Verification of Step 4 Orthogonalization (sum P_K * w_c):")
    print(ortho_check)
    assert np.allclose(ortho_check, 0, atol=1e-10), "Orthogonalization mathematically failed!"
    print("Orthogonalization structurally verified via Sieve weight projection.\n")
    
    # 2. Run Monte Carlo
    t0 = time.time()
    runner = MonteCarloRunner(iterations=iterations_sim, T=T_sim, K=K_sim)
    AcademicPlotter.plot_size(runner)
    AcademicPlotter.plot_power(runner, rho=0.99)
    print(f"Total time elapsed: {time.time() - t0:.2f} seconds")

