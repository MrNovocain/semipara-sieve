import argparse
import time
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import scipy.stats as stats

from mc_sieve_el import BSplineSieve, LegendreSieve, SimulatedDGP, SieveELEstimator


CRITICAL_VALUES = {
    0.10: stats.chi2.ppf(0.90, df=1),
    0.05: stats.chi2.ppf(0.95, df=1),
    0.01: stats.chi2.ppf(0.99, df=1),
}


def make_basis(name: str, K: int):
    if name == "legendre":
        return LegendreSieve(K)
    if name == "bspline":
        return BSplineSieve(K, degree=3)
    raise ValueError(f"Unknown basis: {name}")


def run_single_null_replication(
    seed: int,
    T: int,
    rho: float,
    phi: float,
    K: int,
    basis_name: str,
) -> float:
    dgp = SimulatedDGP(T=T, rho=rho, beta=0.0, phi=phi, m_type=m_type, x_w_loading=x_w_loading, seed=int(seed))
    estimator = SieveELEstimator(make_basis(basis_name, K))
    return estimator.fit(dgp, beta_0=0.0)


def run_null_distribution(
    T: int,
    K: int,
    rho: float,
    phi: float,
    iterations: int,
    basis_name: str,
    jobs: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K, basis_name=basis_name, m_type=m_type, x_w_loading=x_w_loading)

    t0 = time.time()
    if jobs == 1:
        stats_out = np.array(list(map(func, seeds)), dtype=float)
    else:
        with mp.Pool(jobs) as pool:
            stats_out = np.array(pool.map(func, seeds), dtype=float)
    return stats_out, time.time() - t0


def summarize_chisq_fit(raw_stats: np.ndarray) -> Dict[str, object]:
    finite_mask = np.isfinite(raw_stats) & (raw_stats < 1e5)
    valid = raw_stats[finite_mask]
    failures = int(raw_stats.size - valid.size)
    if valid.size == 0:
        raise RuntimeError("No finite profile EL statistics were computed.")

    ks_stat, ks_p = stats.kstest(valid, "chi2", args=(1,))
    alpha_rates = {
        alpha: float(np.mean(valid > cv))
        for alpha, cv in CRITICAL_VALUES.items()
    }
    probs = np.array([0.50, 0.90, 0.95, 0.99])
    empirical_q = np.quantile(valid, probs)
    theoretical_q = stats.chi2.ppf(probs, df=1)

    return {
        "valid": valid,
        "failures": failures,
        "mean": float(np.mean(valid)),
        "var": float(np.var(valid, ddof=1)),
        "median": float(np.median(valid)),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "rates": alpha_rates,
        "probs": probs,
        "empirical_q": empirical_q,
        "theoretical_q": theoretical_q,
    }


def print_summary(summary: Dict[str, object], iterations: int, elapsed: float, T: int, K: int, rho: float, phi: float, basis: str):
    print("Null chi-square diagnostic for profile sieve EL")
    print(f"Parameters: T={T}, K={K}, rho={rho}, phi={phi}, basis={basis}, replications={iterations}")
    print(f"Simulation time: {elapsed:.2f} seconds")
    print(f"Convex-hull/root failures: {summary['failures']} / {iterations}")

    print("\nMoment comparison against chi-square(1): mean=1, variance=2")
    print(f"  mean:   {summary['mean']:.4f}")
    print(f"  var:    {summary['var']:.4f}")
    print(f"  median: {summary['median']:.4f}  (chi-square(1) median={stats.chi2.ppf(0.5, df=1):.4f})")

    print("\nEmpirical rejection rates:")
    print(f"  {'Nominal':<8} | {'Empirical':<9} | {'Chi-square CV':<13}")
    print("-" * 38)
    for alpha, cv in CRITICAL_VALUES.items():
        print(f"  {alpha * 100:5.1f}%   | {summary['rates'][alpha] * 100:7.2f}%   | {cv:11.4f}")

    print("\nQuantile comparison:")
    print(f"  {'p':<6} | {'Empirical':<10} | {'Chi-square(1)':<12} | {'Diff':<10}")
    print("-" * 48)
    for p, emp, th in zip(summary['probs'], summary['empirical_q'], summary['theoretical_q']):
        print(f"  {p:4.2f}  | {emp:10.4f} | {th:12.4f} | {emp - th:10.4f}")

    print("\nKolmogorov-Smirnov test against chi-square(1):")
    print(f"  KS statistic: {summary['ks_stat']:.4f}")
    print(f"  p-value:      {summary['ks_p']:.4f}")


def timestamped_result_path(stem: str, suffix: str = ".png") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{stem}_{timestamp}{suffix}"


def plot_qq(valid_stats: np.ndarray, output: str):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
    })

    n = len(valid_stats)
    empirical = np.sort(valid_stats)
    theoretical = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=1)
    limit = max(float(np.max(empirical)), float(stats.chi2.ppf(0.999, df=1)))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(theoretical, empirical, facecolors="none", edgecolors="k", s=12, label="Simulated profile EL")
    ax.plot([0, limit], [0, limit], "k--", label="45-degree reference")
    ax.set_xlabel("Theoretical $\\chi^2_1$ Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("Profile Sieve-EL Null Distribution vs. $\\chi^2_1$")
    ax.legend(frameon=False)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"\nQ-Q plot saved as '{output}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile sieve EL chi-square(1) null diagnostic.")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--basis", choices=["legendre", "bspline"], default="legendre")
    parser.add_argument("--m-type", choices=["smooth", "mixed", "rough"], default="smooth")
    parser.add_argument("--x-w-loading", type=float, default=0.0, help="Stress parameter: add this multiple of W_t to the observed predictor X_t.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    K = args.K if args.K is not None else int(np.round(args.T ** (1 / 3)))
    raw_stats, elapsed = run_null_distribution(
        T=args.T,
        K=K,
        rho=args.rho,
        phi=args.phi,
        iterations=args.iterations,
        basis_name=args.basis,
        jobs=args.jobs,
        seed=args.seed,
    )
    summary = summarize_chisq_fit(raw_stats)
    print_summary(summary, args.iterations, elapsed, args.T, K, args.rho, args.phi, args.basis)
    if not args.no_plot:
        output = args.output or timestamped_result_path("null_qq_plot")
        plot_qq(summary["valid"], output)


if __name__ == "__main__":
    mp.freeze_support()
    main()
