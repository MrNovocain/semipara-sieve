import argparse
import csv
import multiprocessing as mp
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from mc_sieve_el import BSplineSieve, LegendreSieve, SimulatedDGP, SieveELEstimator


CHI2_5PCT_CV = stats.chi2.ppf(0.95, df=1)
CHI2_MEAN = 1.0
CHI2_VAR = 2.0


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


def timestamped_result_path(stem: str, timestamp: str, suffix: str) -> Path:
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{stem}_{timestamp}{suffix}"


def simulate_for_k(K: int, args, seed_offset: int) -> np.ndarray:
    total = args.batches * args.iterations_per_batch
    rng = np.random.default_rng(args.seed + seed_offset)
    seeds = rng.integers(0, 2**31 - 1, size=total)
    func = partial(run_single_null_replication, T=args.T, rho=args.rho, phi=args.phi, K=K, basis_name=args.basis, m_type=args.m_type, x_w_loading=args.x_w_loading)

    if args.jobs == 1:
        raw = np.array(list(map(func, seeds)), dtype=float)
    else:
        with mp.Pool(args.jobs) as pool:
            raw = np.array(pool.map(func, seeds), dtype=float)

    raw = raw[np.isfinite(raw) & (raw < 1e5)]
    if len(raw) == 0:
        raise RuntimeError(f"No valid profile EL statistics for K={K}")
    return raw


def summarize_stats(K: int, values: np.ndarray) -> dict:
    ks_stat, ks_p = stats.kstest(values, "chi2", args=(1,))
    size_5 = float(np.mean(values > CHI2_5PCT_CV))
    q95 = float(np.quantile(values, 0.95))
    mean = float(np.mean(values))
    var = float(np.var(values, ddof=1))
    return {
        "K": K,
        "valid_replications": int(len(values)),
        "size_5": size_5,
        "size_5_error_abs": abs(size_5 - 0.05),
        "mean": mean,
        "mean_error_abs": abs(mean - CHI2_MEAN),
        "var": var,
        "var_error_abs": abs(var - CHI2_VAR),
        "q95": q95,
        "q95_error_abs": abs(q95 - CHI2_5PCT_CV),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


def write_summary_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "basis",
                "m_type",
                "x_w_loading",
                "rho",
                "phi",
                "batches",
                "iterations_per_batch",
                "K",
                "valid_replications",
                "size_5",
                "size_5_error_abs",
                "mean",
                "mean_error_abs",
                "var",
                "var_error_abs",
                "q95",
                "q95_error_abs",
                "ks_stat",
                "ks_p",
                "smoothness",
                "dimension",
                "lower_bound",
                "selected_K",
                "upper_bound",
                "bound_region",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def lower_bound(T: int, dimension: float, smoothness: float) -> float:
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if smoothness <= 0:
        raise ValueError("smoothness must be positive.")
    return T ** (dimension / (2.0 * smoothness))


def upper_bound(T: int) -> float:
    return T ** 0.5


def bound_region(K: int, lower: float, upper: float) -> str:
    if K <= lower:
        return "below_lower_bound"
    if K >= upper:
        return "above_upper_bound"
    return "inside_window"


def plot_k_sensitivity(rows: list, args, output: Path):
    lower = lower_bound(args.T, args.dimension, args.smoothness)
    selected = int(np.round(args.T ** (1 / 3)))
    upper = upper_bound(args.T)

    rows = sorted(rows, key=lambda row: row["K"])
    Ks = np.array([row["K"] for row in rows])
    size_5 = np.array([row["size_5"] for row in rows])
    q95 = np.array([row["q95"] for row in rows])
    ks_stat = np.array([row["ks_stat"] for row in rows])

    colors = []
    for K in Ks:
        if K <= lower or K >= upper:
            colors.append("0.70")
        elif K == selected:
            colors.append("k")
        else:
            colors.append("0.25")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharex=True)
    fig.suptitle(
        f"Profile Sieve-EL Sensitivity to Sieve Dimension K (T={args.T})",
        fontsize=13,
        y=1.02,
    )

    for ax in axes:
        ax.axvspan(0, lower, color="0.92", zorder=0)
        ax.axvspan(upper, max(Ks) * 1.06, color="0.92", zorder=0)
        ax.axvline(lower, color="0.45", linestyle=":", linewidth=1.1, label="$T^{d/(2s)}$ lower" if ax is axes[0] else None)
        ax.axvline(selected, color="k", linestyle="-", linewidth=1.2, label="$round(T^{1/3})$" if ax is axes[0] else None)
        ax.axvline(upper, color="0.45", linestyle=":", linewidth=1.1, label="$T^{1/2}$ upper" if ax is axes[0] else None)
        ax.set_xlabel("Sieve dimension K")

    axes[0].plot(Ks, 100 * size_5, color="k", marker="o", markersize=4)
    axes[0].scatter(Ks, 100 * size_5, c=colors, s=28, zorder=3)
    axes[0].axhline(5.0, color="0.25", linestyle="--", linewidth=1.1, label="Nominal 5%")
    axes[0].set_ylabel("Empirical 5% size (%)")
    axes[0].set_title("Size Calibration")

    axes[1].plot(Ks, q95, color="k", marker="o", markersize=4)
    axes[1].scatter(Ks, q95, c=colors, s=28, zorder=3)
    axes[1].axhline(CHI2_5PCT_CV, color="0.25", linestyle="--", linewidth=1.1, label="$\\chi^2_{1,0.95}$")
    axes[1].set_ylabel("Empirical 95% quantile")
    axes[1].set_title("Upper Tail")

    axes[2].plot(Ks, ks_stat, color="k", marker="o", markersize=4)
    axes[2].scatter(Ks, ks_stat, c=colors, s=28, zorder=3)
    axes[2].set_ylabel("KS distance to $\\chi^2_1$")
    axes[2].set_title("Distributional Distance")

    axes[0].legend(frameon=False, loc="best")
    axes[1].legend(frameon=False, loc="best")
    fig.text(
        0.5,
        -0.015,
        f"Shaded regions mark K outside the theoretical window: K <= T^(d/(2s))={lower:.2f} with d={args.dimension:g}, s={args.smoothness:g}, or K >= T^(1/2)={upper:.2f}. "
        f"DGP: rho={args.rho}, phi={args.phi}, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g}, reps per K={args.batches * args.iterations_per_batch}.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_k_grid(value: str) -> list:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-T K sensitivity diagnostic for profile sieve EL.")
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--K-grid", type=parse_k_grid, default=None)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--iterations-per-batch", type=int, default=40)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--basis", choices=["legendre", "bspline"], default="legendre")
    parser.add_argument("--m-type", choices=["smooth", "mixed", "rough"], default="smooth")
    parser.add_argument("--x-w-loading", type=float, default=0.0, help="Stress parameter: add this multiple of W_t to the observed predictor X_t.")
    parser.add_argument("--smoothness", type=float, default=2.0, help="Smoothness s in the lower bound T^(d/(2s)).")
    parser.add_argument("--dimension", type=float, default=1.0, help="Dimension d of the nonparametric covariate W.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def default_k_grid(T: int) -> list:
    selected = int(np.round(T ** (1 / 3)))
    candidates = [2, 4, 6, 8, 10, selected, 16, 20, 28, 36, 44, 60, 80]
    return sorted(set(k for k in candidates if k < T // 2))


def main():
    args = parse_args()
    K_grid = args.K_grid or default_k_grid(args.T)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else timestamped_result_path("k_sensitivity_chisq", timestamp, ".png")
    summary_output = Path(args.summary_output) if args.summary_output else timestamped_result_path("k_sensitivity_chisq_summary", timestamp, ".csv")

    lower = lower_bound(args.T, args.dimension, args.smoothness)
    selected = int(np.round(args.T ** (1 / 3)))
    upper = upper_bound(args.T)
    print(
        f"Running fixed-T K sensitivity: T={args.T}, K_grid={K_grid}, "
        f"lower={lower:.2f}, selected={selected}, upper={upper:.2f}, "
        f"reps_per_K={args.batches * args.iterations_per_batch}"
    )

    rows = []
    total_t0 = time.time()
    for idx, K in enumerate(K_grid):
        t0 = time.time()
        values = simulate_for_k(K, args, seed_offset=10_000 * idx + K)
        row = summarize_stats(K, values)
        row.update({
            "T": args.T,
            "basis": args.basis,
            "m_type": args.m_type,
            "x_w_loading": args.x_w_loading,
            "rho": args.rho,
            "phi": args.phi,
            "batches": args.batches,
            "iterations_per_batch": args.iterations_per_batch,
            "smoothness": args.smoothness,
            "dimension": args.dimension,
            "lower_bound": lower,
            "selected_K": selected,
            "upper_bound": upper,
            "bound_region": bound_region(K, lower, upper),
        })
        rows.append(row)
        print(
            f"K={K:3d} [{row['bound_region']}]: size5={row['size_5'] * 100:5.2f}%, "
            f"q95={row['q95']:.4f}, KS={row['ks_stat']:.4f}, elapsed={time.time() - t0:.2f}s"
        )

    write_summary_csv(summary_output, rows)
    plot_k_sensitivity(rows, args, output)
    print(f"Summary CSV saved as '{summary_output}'.")
    print(f"K sensitivity plot saved as '{output}'.")
    print(f"Total elapsed: {time.time() - total_t0:.2f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
