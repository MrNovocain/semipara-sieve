import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from functools import partial
from mc_sieve_el import SimulatedDGP, BSplineSieve, SieveELEstimator

def run_single_null_replication(seed: int, T: int, rho: float, phi: float, K: int) -> tuple:
    dgp = SimulatedDGP(T=T, rho=rho, beta=0.0, phi=phi, seed=seed)
    basis = BSplineSieve(K, degree=3)
    estimator = SieveELEstimator(basis)
    l_T = estimator.fit(dgp, beta_0=0.0)
    s_T = estimator.fit_robust_score(dgp, beta_0=0.0)
    return l_T, s_T

def run_simulations_for_T(T: int, K: int, iterations: int):
    rho = 0.99
    phi = 0.5
    seeds = np.random.randint(0, 10000000, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K)
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(func, seeds)
        
    el_stats = np.array([r[0] for r in results])
    score_stats = np.array([r[1] for r in results])
    
    # Filter valid
    valid_el = el_stats[el_stats < 1e5]
    valid_score = score_stats[score_stats < 1e5]
    
    return valid_el, valid_score

def main():
    Ts = [1000, 2000, 3000, 4000, 5000]
    iterations = 1000
    results = {}
    
    print(f"Running Q-Q convergence simulations with B-Splines (T={Ts}, iterations={iterations})...")
    
    for T in Ts:
        K = int(np.round(T**(1/3)))
        t0 = time.time()
        valid_el, valid_score = run_simulations_for_T(T, K, iterations)
        t1 = time.time()
        print(f"T = {T} (K={K}) completed in {t1 - t0:.2f} seconds. Valid EL: {len(valid_el)}, Valid Score: {len(valid_score)}")
        results[T] = (valid_el, valid_score)
        
    # Set academic plotting style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.2,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9
    })
    
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    
    for idx, T in enumerate(Ts):
        valid_el, valid_score = results[T]
        K = int(np.round(T**(1/3)))
        
        # 1. Sieve-EL Q-Q Plot
        ax_el = axes[0, idx]
        n_el = len(valid_el)
        empirical_el = np.sort(valid_el)
        theoretical_el = stats.chi2.ppf((np.arange(1, n_el + 1) - 0.5) / n_el, df=1)
        
        ax_el.scatter(theoretical_el, empirical_el, facecolors='none', edgecolors='k', s=15, label=f'T = {T}')
        max_val_el = max(np.max(valid_el), stats.chi2.ppf(0.99, df=1))
        ax_el.plot([0, max_val_el], [0, max_val_el], 'k--', alpha=0.7)
        ax_el.set_xlabel('Theoretical $\\chi^2_1$ Quantiles')
        if idx == 0:
            ax_el.set_ylabel('Empirical Quantiles (Sieve-EL)')
        ax_el.set_title(f'Sieve-EL ($T = {T}$, $K = {K}$)')
        ax_el.legend(frameon=False, loc='upper left')
        
        # 2. Robust Sieve Score Q-Q Plot
        ax_score = axes[1, idx]
        n_score = len(valid_score)
        empirical_score = np.sort(valid_score)
        theoretical_score = stats.chi2.ppf((np.arange(1, n_score + 1) - 0.5) / n_score, df=1)
        
        ax_score.scatter(theoretical_score, empirical_score, facecolors='none', edgecolors='k', s=15, label=f'T = {T}')
        max_val_score = max(np.max(valid_score), stats.chi2.ppf(0.99, df=1))
        ax_score.plot([0, max_val_score], [0, max_val_score], 'k--', alpha=0.7)
        ax_score.set_xlabel('Theoretical $\\chi^2_1$ Quantiles')
        if idx == 0:
            ax_score.set_ylabel('Empirical Quantiles (Score)')
        ax_score.set_title(f'Robust Sieve Score ($T = {T}$, $K = {K}$)')
        ax_score.legend(frameon=False, loc='upper left')
        
    plt.tight_layout()
    plt.savefig('qq_convergence_plot.png', dpi=300)
    print("Academic Q-Q convergence plot with B-Splines saved as 'qq_convergence_plot.png'.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
