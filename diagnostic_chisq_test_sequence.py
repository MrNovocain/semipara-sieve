import numpy as np
import scipy.stats as stats
import multiprocessing as mp
import time
from functools import partial
from mc_sieve_el import SimulatedDGP, LegendreSieve, SieveELEstimator

def run_single_null_replication(seed: int, T: int, rho: float, phi: float, K: int) -> tuple:
    dgp = SimulatedDGP(T=T, rho=rho, beta=0.0, phi=phi, seed=seed)
    basis = LegendreSieve(K)
    estimator = SieveELEstimator(basis)
    l_T = estimator.fit(dgp, beta_0=0.0)
    s_T = estimator.fit_robust_score(dgp, beta_0=0.0)
    return l_T, s_T

def test_size_for_T(T: int, iterations: int = 2000) -> dict:
    K = int(np.round(T**(1/3)))
    rho = 0.99
    phi = 0.5
    
    seeds = np.random.randint(0, 10000000, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K)
    
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(func, seeds)
    t1 = time.time()
    
    el_stats = np.array([r[0] for r in results])
    score_stats = np.array([r[1] for r in results])
    
    # Filter valid
    valid_el = el_stats[el_stats < 1e5]
    valid_score = score_stats[score_stats < 1e5]
    
    alpha_levels = [0.10, 0.05, 0.01]
    
    rates_el = {a: np.mean(valid_el > stats.chi2.ppf(1 - a, df=1)) for a in alpha_levels}
    rates_score = {a: np.mean(valid_score > stats.chi2.ppf(1 - a, df=1)) for a in alpha_levels}
    
    _, p_val_el = stats.kstest(valid_el, 'chi2', args=(1,))
    _, p_val_score = stats.kstest(valid_score, 'chi2', args=(1,))
    
    return {
        'T': T,
        'K': K,
        'el': {
            'mean': np.mean(valid_el),
            'var': np.var(valid_el),
            'rates': rates_el,
            'ks_p': p_val_el
        },
        'score': {
            'mean': np.mean(valid_score),
            'var': np.var(valid_score),
            'rates': rates_score,
            'ks_p': p_val_score
        },
        'time': t1 - t0
    }

def main():
    Ts = [500, 1000, 2000, 4000]
    iterations = 2000
    
    print(f"Running asymptotic convergence diagnostics for Sieve-EL vs. Robust Sieve Score (iterations={iterations} per T)...")
    print(f"{'T':>6} | {'Stat':>5} | {'K':>2} | {'Mean':>7} | {'Var':>7} | {'Size 10%':>8} | {'Size 5%':>7} | {'Size 1%':>7} | {'KS p-val':>8}")
    print("-" * 87)
    
    for T in Ts:
        res = test_size_for_T(T, iterations)
        el = res['el']
        sc = res['score']
        print(f"{res['T']:6d} | {'EL':5s} | {res['K']:2d} | {el['mean']:7.4f} | {el['var']:7.4f} | {el['rates'][0.10]*100:7.2f}% | {el['rates'][0.05]*100:6.2f}% | {el['rates'][0.01]*100:6.2f}% | {el['ks_p']:8.4f}")
        print(f"{'':6s} | {'Score':5s} | {res['K']:2d} | {sc['mean']:7.4f} | {sc['var']:7.4f} | {sc['rates'][0.10]*100:7.2f}% | {sc['rates'][0.05]*100:6.2f}% | {sc['rates'][0.01]*100:6.2f}% | {sc['ks_p']:8.4f}")
        print("-" * 87)


if __name__ == "__main__":
    mp.freeze_support()
    main()
