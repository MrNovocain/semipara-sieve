import argparse
import multiprocessing as mp
import time
from functools import partial

import numpy as np
import scipy.stats as stats

from mc_sieve_el import BSplineSieve, LegendreSieve, SimulatedDGP, SieveELEstimator


def make_basis(name: str, K: int):
    if name == "legendre":
        return LegendreSieve(K)
    if name == "bspline":
        return BSplineSieve(K, degree=3)
    raise ValueError(f"Unknown basis: {name}")


def run_single_null_replication(seed: int, T: int, rho: float, phi: float, K: int, basis_name: str, m_type: str = "smooth", x_w_loading: float = 0.0) -> float:
    dgp = SimulatedDGP(T=T, rho=rho, beta=0.0, phi=phi, m_type=m_type, x_w_loading=x_w_loading, seed=int(seed))
    estimator = SieveELEstimator(make_basis(basis_name, K))
    return estimator.fit(dgp, beta_0=0.0)


def test_size_for_T(T: int, iterations: int, rho: float, phi: float, basis_name: str, jobs: int, seed: int) -> dict:
    K = int(np.round(T ** (1 / 3)))
    rng = np.random.default_rng(seed + T)
    seeds = rng.integers(0, 2**31 - 1, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K, basis_name=basis_name, m_type=m_type, x_w_loading=x_w_loading)

    t0 = time.time()
    if jobs == 1:
        raw_stats = np.array(list(map(func, seeds)), dtype=float)
    else:
        with mp.Pool(jobs) as pool:
            raw_stats = np.array(pool.map(func, seeds), dtype=float)
    elapsed = time.time() - t0

    valid = raw_stats[np.isfinite(raw_stats) & (raw_stats < 1e5)]
    if len(valid) == 0:
        raise RuntimeError(f"No valid statistics for T={T}")

    alpha_levels = [0.10, 0.05, 0.01]
    rates = {a: np.mean(valid > stats.chi2.ppf(1 - a, df=1)) for a in alpha_levels}
    ks_stat, ks_p = stats.kstest(valid, "chi2", args=(1,))

    return {
        "T": T,
        "K": K,
        "mean": np.mean(valid),
        "var": np.var(valid, ddof=1),
        "rates": rates,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "failures": len(raw_stats) - len(valid),
        "time": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Profile sieve EL chi-square convergence sequence.")
    parser.add_argument("--Ts", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--basis", choices=["legendre", "bspline"], default="legendre")
    parser.add_argument("--m-type", choices=["smooth", "mixed", "rough"], default="smooth")
    parser.add_argument("--x-w-loading", type=float, default=0.0, help="Stress parameter: add this multiple of W_t to the observed predictor X_t.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Profile sieve EL chi-square convergence diagnostics "
        f"(iterations={args.iterations} per T, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g})"
    )
    print(f"{'T':>6} | {'K':>3} | {'Mean':>7} | {'Var':>7} | {'Size 10%':>8} | {'Size 5%':>7} | {'Size 1%':>7} | {'KS stat':>7} | {'KS p':>7} | {'Fail':>4} | {'Sec':>6}")
    print("-" * 108)

    for T in args.Ts:
        res = test_size_for_T(T, args.iterations, args.rho, args.phi, args.basis, args.jobs, args.seed)
        rates = res["rates"]
        print(
            f"{res['T']:6d} | {res['K']:3d} | {res['mean']:7.4f} | {res['var']:7.4f} | "
            f"{rates[0.10] * 100:7.2f}% | {rates[0.05] * 100:6.2f}% | {rates[0.01] * 100:6.2f}% | "
            f"{res['ks_stat']:7.4f} | {res['ks_p']:7.4f} | {res['failures']:4d} | {res['time']:6.2f}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
