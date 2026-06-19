import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
from functools import partial
from mc_sieve_el import SimulatedDGP, LegendreSieve, SieveELEstimator

def run_single_null_replication(seed: int, T: int, rho: float, phi: float, K: int) -> tuple:
    # Under H0: beta = 0
    dgp = SimulatedDGP(T=T, rho=rho, beta=0.0, phi=phi, seed=seed)
    basis = LegendreSieve(K)
    estimator = SieveELEstimator(basis)
    l_T = estimator.fit(dgp, beta_0=0.0)
    s_T = estimator.fit_robust_score(dgp, beta_0=0.0)
    return l_T, s_T

def main():
    # Simulation Parameters
    T = 500
    K = int(np.round(T**(1/3))) # K = 8
    rho = 0.99
    phi = 0.5
    iterations = 2000
    
    print(f"Starting null distribution convergence test...")
    print(f"Parameters: T={T}, K={K}, rho={rho}, phi={phi}, replications={iterations}")
    
    seeds = np.random.randint(0, 10000000, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K)
    
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(func, seeds)
    t1 = time.time()
    print(f"Simulation completed in {t1 - t0:.2f} seconds.")
    
    el_stats = np.array([r[0] for r in results])
    score_stats = np.array([r[1] for r in results])
    
    # Filter out any numerical failures (1e6 values) for EL
    valid_mask_el = el_stats < 1e5
    valid_el = el_stats[valid_mask_el]
    num_failures_el = np.sum(~valid_mask_el)
    
    # Filter out any numerical failures for Score (should be 0)
    valid_mask_score = score_stats < 1e5
    valid_score = score_stats[valid_mask_score]
    num_failures_score = np.sum(~valid_mask_score)
    
    print("\nNumerical failures (convex hull violations):")
    print(f"  Sieve-EL:           {num_failures_el} / {iterations}")
    print(f"  Robust Sieve Score: {num_failures_score} / {iterations}")
    
    if len(valid_el) == 0 or len(valid_score) == 0:
        print("Error: No valid statistics computed.")
        return
        
    # Calculate Moments
    print("\nMoments Comparison (Theoretical Chi-sq(1): Mean = 1.0, Var = 2.0):")
    print(f"  Sieve-EL Mean:           {np.mean(valid_el):.4f}")
    print(f"  Sieve-EL Var:            {np.var(valid_el):.4f}")
    print(f"  Robust Sieve Score Mean: {np.mean(valid_score):.4f}")
    print(f"  Robust Sieve Score Var:  {np.var(valid_score):.4f}")
    
    # Empirical Rejection Rates at Nominal Levels
    alpha_levels = [0.10, 0.05, 0.01]
    print("\nRejection Rates at Nominal Levels:")
    print(f"  {'Nominal':<8} | {'Sieve-EL':<10} | {'Robust Sieve Score':<18}")
    print("-" * 45)
    for alpha in alpha_levels:
        cv = stats.chi2.ppf(1 - alpha, df=1)
        rate_el = np.mean(valid_el > cv)
        rate_score = np.mean(valid_score > cv)
        print(f"  {alpha*100:2.0f}%      | {rate_el*100:8.2f}%   | {rate_score*100:16.2f}%")
        
    # Kolmogorov-Smirnov Test against Chi-sq(1)
    ks_stat_el, p_val_el = stats.kstest(valid_el, 'chi2', args=(1,))
    ks_stat_score, p_val_score = stats.kstest(valid_score, 'chi2', args=(1,))
    print("\nKolmogorov-Smirnov Test for Chi-sq(1) Fit:")
    print(f"  Sieve-EL:           KS Stat = {ks_stat_el:.4f}, p-value = {p_val_el:.4f}")
    print(f"  Robust Sieve Score: KS Stat = {ks_stat_score:.4f}, p-value = {p_val_score:.4f}")
    
    # Generate Academic Q-Q Plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    
    # 1. Sieve-EL Q-Q Plot
    n_el = len(valid_el)
    empirical_quantiles_el = np.sort(valid_el)
    theoretical_quantiles_el = stats.chi2.ppf((np.arange(1, n_el + 1) - 0.5) / n_el, df=1)
    ax1.scatter(theoretical_quantiles_el, empirical_quantiles_el, facecolors='none', edgecolors='k', s=10, label='Simulated Sieve-EL')
    max_val_el = max(np.max(valid_el), stats.chi2.ppf(0.999, df=1))
    ax1.plot([0, max_val_el], [0, max_val_el], 'k--', label='45-degree Reference Line')
    ax1.set_xlabel('Theoretical $\\chi^2_1$ Quantiles')
    ax1.set_ylabel('Empirical Quantiles')
    ax1.set_title('Q-Q Plot: Empirical Likelihood Statistic vs. $\\chi^2_1$')
    ax1.legend(frameon=False)
    
    # 2. Robust Sieve Score Q-Q Plot
    n_score = len(valid_score)
    empirical_quantiles_score = np.sort(valid_score)
    theoretical_quantiles_score = stats.chi2.ppf((np.arange(1, n_score + 1) - 0.5) / n_score, df=1)
    ax2.scatter(theoretical_quantiles_score, empirical_quantiles_score, facecolors='none', edgecolors='k', s=10, label='Simulated Robust Sieve Score')
    max_val_score = max(np.max(valid_score), stats.chi2.ppf(0.999, df=1))
    ax2.plot([0, max_val_score], [0, max_val_score], 'k--', label='45-degree Reference Line')
    ax2.set_xlabel('Theoretical $\\chi^2_1$ Quantiles')
    ax2.set_ylabel('Empirical Quantiles')
    ax2.set_title('Q-Q Plot: Robust Sieve Score Statistic vs. $\\chi^2_1$')
    ax2.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig('null_qq_plot.png', dpi=300)
    print("\nAcademic Q-Q plots saved as 'null_qq_plot.png'.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
