import argparse
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mc_sieve_el import BSplineSieve, LegendreSieve, SimulatedDGP, SieveELEstimator


CHI2_5PCT_CV = stats.chi2.ppf(0.95, df=1)


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


def run_simulations_for_T(T: int, K: int, iterations: int, rho: float, phi: float, basis_name: str, jobs: int, seed: int):
    rng = np.random.default_rng(seed + T)
    seeds = rng.integers(0, 2**31 - 1, size=iterations)
    func = partial(run_single_null_replication, T=T, rho=rho, phi=phi, K=K, basis_name=basis_name, m_type=m_type, x_w_loading=x_w_loading)

    if jobs == 1:
        raw_stats = np.array(list(map(func, seeds)), dtype=float)
    else:
        with mp.Pool(jobs) as pool:
            raw_stats = np.array(pool.map(func, seeds), dtype=float)
    return raw_stats[np.isfinite(raw_stats) & (raw_stats < 1e5)]


def timestamped_result_path(stem: str, suffix: str = ".png") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{stem}_{timestamp}{suffix}"


def parse_args():
    parser = argparse.ArgumentParser(description="Five-panel Q-Q convergence plot for profile sieve EL.")
    parser.add_argument("--Ts", type=int, nargs="+", default=[500, 1000, 2000, 3000, 4000])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--basis", choices=["legendre", "bspline"], default="legendre")
    parser.add_argument("--m-type", choices=["smooth", "mixed", "rough"], default="smooth")
    parser.add_argument("--x-w-loading", type=float, default=0.0, help="Stress parameter: add this multiple of W_t to the observed predictor X_t.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_panel_data(results: dict):
    panels = []
    for T, valid_stats in results.items():
        empirical = np.sort(valid_stats)
        n = len(empirical)
        theoretical = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=1)
        ks_stat, ks_p = stats.kstest(valid_stats, "chi2", args=(1,))
        panels.append({
            "T": T,
            "K": int(np.round(T ** (1 / 3))),
            "empirical": empirical,
            "theoretical": theoretical,
            "size_5": float(np.mean(valid_stats > CHI2_5PCT_CV)),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "q995": float(np.quantile(valid_stats, 0.995)),
        })
    return panels


def plot_convergence(panels: list, args, output: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
    })

    common_limit = max(
        stats.chi2.ppf(0.995, df=1),
        max(panel["q995"] for panel in panels),
        CHI2_5PCT_CV * 1.12,
    )
    common_limit = float(np.ceil(common_limit * 10) / 10)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.15 * len(panels), 4.35), squeeze=False)
    fig.suptitle(
        "Profile Sieve-EL Q-Q Convergence to $\\chi^2_1$",
        fontsize=13,
        y=1.02,
    )

    for idx, panel in enumerate(panels):
        ax = axes[0, idx]
        ax.scatter(
            panel["theoretical"],
            panel["empirical"],
            facecolors="none",
            edgecolors="k",
            linewidths=0.7,
            s=13,
        )
        ax.plot([0, common_limit], [0, common_limit], "k--", alpha=0.7, label="45-degree line")
        ax.axvline(CHI2_5PCT_CV, color="0.35", linestyle=":", linewidth=1.15, label="5% cutoff" if idx == 0 else None)
        ax.axhline(CHI2_5PCT_CV, color="0.35", linestyle=":", linewidth=1.15)
        ax.plot(CHI2_5PCT_CV, CHI2_5PCT_CV, marker="o", color="0.15", markersize=3.5)

        ax.set_xlim(0, common_limit)
        ax.set_ylim(0, common_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Theoretical $\\chi^2_1$ Quantiles")
        if idx == 0:
            ax.set_ylabel("Empirical Quantiles")
        ax.set_title(f"$T={panel['T']}$, $K={panel['K']}$")
        ax.text(
            0.05,
            0.95,
            f"5% size: {panel['size_5'] * 100:.2f}%\nKS: {panel['ks_stat']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        if idx == 0:
            ax.legend(frameon=False, loc="lower right")

    fig.text(
        0.5,
        -0.015,
        f"Null simulations: reps={args.iterations}, rho={args.rho}, phi={args.phi}, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g}. Dotted lines mark $\\chi^2_{{1,0.95}}={CHI2_5PCT_CV:.4f}$.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results = {}
    if len(args.Ts) != 5:
        print(f"Warning: requested {len(args.Ts)} panels. Default is five panels.")
    print(f"Running Q-Q convergence simulations (Ts={args.Ts}, iterations={args.iterations}, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g})")

    for T in args.Ts:
        K = int(np.round(T ** (1 / 3)))
        t0 = time.time()
        valid_stats = run_simulations_for_T(T, K, args.iterations, args.rho, args.phi, args.basis, args.jobs, args.seed)
        elapsed = time.time() - t0
        size_5 = np.mean(valid_stats > CHI2_5PCT_CV)
        print(f"T={T} (K={K}) completed in {elapsed:.2f}s. Valid={len(valid_stats)}, 5% size={size_5 * 100:.2f}%")
        results[T] = valid_stats

    panels = build_panel_data(results)
    output = Path(args.output) if args.output else timestamped_result_path("qq_convergence_5panel")
    plot_convergence(panels, args, output)
    print(f"Five-panel Q-Q convergence plot saved as '{output}'.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
