import argparse
import csv
import math
import multiprocessing as mp
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from profile_sieve.mc_sieve_el import BSplineSieve, LegendreSieve, SimulatedDGP, SieveELEstimator


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


def timestamped_result_path(stem: str, timestamp: str, suffix: str) -> Path:
    result_dir = ROOT / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{stem}_{timestamp}{suffix}"


def quantile_grid(iterations_per_batch: int, q_points: int, q_min: float, q_max: float) -> np.ndarray:
    if not (0.0 < q_min < q_max < 1.0):
        raise ValueError("Require 0 < q_min < q_max < 1.")
    if iterations_per_batch <= 1:
        raise ValueError("iterations_per_batch must exceed 1.")

    # Per-batch quantile bands are unstable beyond the order-statistic support.
    lower = max(q_min, 1.0 / (iterations_per_batch + 1.0))
    upper = min(q_max, iterations_per_batch / (iterations_per_batch + 1.0))
    if lower >= upper:
        raise ValueError("Batch size is too small for the requested quantile range.")

    probs = np.linspace(lower, upper, q_points)
    if lower <= 0.95 <= upper:
        probs = np.unique(np.r_[probs, 0.95])
    return probs


def simulate_batched_stats(T: int, K: int, args, base_seed: int) -> np.ndarray:
    total = args.batches * args.iterations_per_batch
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(0, 2**31 - 1, size=total)
    func = partial(run_single_null_replication, T=T, rho=args.rho, phi=args.phi, K=K, basis_name=args.basis, m_type=args.m_type, x_w_loading=args.x_w_loading)

    if args.jobs == 1:
        raw = np.array(list(map(func, seeds)), dtype=float)
    else:
        with mp.Pool(args.jobs) as pool:
            raw = np.array(pool.map(func, seeds), dtype=float)

    raw = raw.reshape(args.batches, args.iterations_per_batch)
    valid_mask = np.isfinite(raw) & (raw < 1e5)
    if not np.all(valid_mask):
        cleaned = []
        for row, mask in zip(raw, valid_mask):
            cleaned.append(row[mask])
        min_len = min(len(row) for row in cleaned)
        if min_len == 0:
            raise RuntimeError(f"At least one batch for T={T} had no valid statistics.")
        return np.vstack([row[:min_len] for row in cleaned])
    return raw


def summarize_batches(stats_by_batch: np.ndarray, q_probs: np.ndarray) -> dict:
    batches = stats_by_batch.shape[0]
    pooled = stats_by_batch.reshape(-1)
    empirical_q = np.vstack([np.quantile(row, q_probs) for row in stats_by_batch])
    size_5 = np.mean(stats_by_batch > CHI2_5PCT_CV, axis=1)
    mean_stat = np.mean(stats_by_batch, axis=1)
    var_stat = np.var(stats_by_batch, axis=1, ddof=1)
    batch_q95 = np.quantile(stats_by_batch, 0.95, axis=1)
    ks_stats = np.array([stats.kstest(row, "chi2", args=(1,)).statistic for row in stats_by_batch])

    return {
        "pooled_empirical_q": np.quantile(pooled, q_probs),
        "mean_batch_empirical_q": np.mean(empirical_q, axis=0),
        "empirical_q_p05": np.quantile(empirical_q, 0.05, axis=0),
        "empirical_q_p95": np.quantile(empirical_q, 0.95, axis=0),
        "size_5_mean": float(np.mean(size_5)),
        "size_5_sd": float(np.std(size_5, ddof=1)) if batches > 1 else 0.0,
        "size_5_se": float(np.std(size_5, ddof=1) / np.sqrt(batches)) if batches > 1 else 0.0,
        "mean_stat_mean": float(np.mean(mean_stat)),
        "var_stat_mean": float(np.mean(var_stat)),
        "batch_q95_mean": float(np.mean(batch_q95)),
        "pooled_q95": float(np.quantile(pooled, 0.95)),
        "ks_stat_mean": float(np.mean(ks_stats)),
    }


def write_summary_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "K",
                "batches",
                "iterations_per_batch",
                "total_replications",
                "avg_size_5",
                "sd_size_5",
                "se_size_5",
                "avg_mean_stat",
                "avg_var_stat",
                "avg_batch_q95",
                "pooled_q95",
                "chi2_q95",
                "avg_ks_stat",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_batched_convergence(panel_data: list, q_probs: np.ndarray, args, output: Path):
    theoretical = stats.chi2.ppf(q_probs, df=1)
    common_limit = max(
        float(np.max(theoretical)),
        max(float(np.nanmax(panel["summary"]["empirical_q_p95"])) for panel in panel_data),
        CHI2_5PCT_CV * 1.12,
    )
    common_limit = float(np.ceil(common_limit * 10) / 10)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.25,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
    })

    ncols = min(args.ncols, len(panel_data))
    nrows = int(math.ceil(len(panel_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.1 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    fig.suptitle("Batched Profile Sieve-EL Q-Q Convergence to $\\chi^2_1$", fontsize=13, y=0.995)

    handles = []
    labels = []
    for idx, panel in enumerate(panel_data):
        summary = panel["summary"]
        ax = flat_axes[idx]
        band = ax.fill_between(
            theoretical,
            summary["empirical_q_p05"],
            summary["empirical_q_p95"],
            color="0.70",
            alpha=0.22,
            linewidth=0,
            label="5%-95% batch band",
        )
        pooled_line, = ax.plot(theoretical, summary["pooled_empirical_q"], color="k", linewidth=1.8, label="Pooled empirical Q-Q")
        if args.show_batch_mean:
            batch_line, = ax.plot(theoretical, summary["mean_batch_empirical_q"], color="0.45", linewidth=1.0, linestyle="-.", label="Mean batch Q-Q")
        ref_line, = ax.plot([0, common_limit], [0, common_limit], "k--", alpha=0.7, label="45-degree line")
        cutoff_line = ax.axvline(CHI2_5PCT_CV, color="0.35", linestyle=":", linewidth=1.1, label="5% cutoff")
        ax.axhline(CHI2_5PCT_CV, color="0.35", linestyle=":", linewidth=1.1)
        ax.plot(CHI2_5PCT_CV, CHI2_5PCT_CV, marker="o", color="0.15", markersize=3.4)

        if idx == 0:
            handles = [pooled_line, band, ref_line, cutoff_line]
            labels = [h.get_label() for h in handles]
            if args.show_batch_mean:
                handles.insert(1, batch_line)
                labels.insert(1, batch_line.get_label())

        ax.set_xlim(0, common_limit)
        ax.set_ylim(0, common_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Theoretical $\\chi^2_1$ Quantiles")
        ax.set_ylabel("Empirical Quantiles")
        ax.set_title(
            f"$T={panel['T']}$, $K={panel['K']}$\n"
            f"5% size: {summary['size_5_mean'] * 100:.2f}%  (SE {summary['size_5_se'] * 100:.2f} pp)"
        )

    for ax in flat_axes[len(panel_data):]:
        ax.axis("off")

    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.012,
        f"Batches={args.batches}, reps per batch={args.iterations_per_batch}, rho={args.rho}, phi={args.phi}, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g}. Quantile grid {q_probs[0]:.3f}-{q_probs[-1]:.3f}; dotted lines mark $\\chi^2_{{1,0.95}}={CHI2_5PCT_CV:.4f}$.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Batched five-panel convergence diagnostic for profile sieve EL.")
    parser.add_argument("--Ts", type=int, nargs="+", default=[500, 1000, 2000, 3000, 4000])
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--iterations-per-batch", type=int, default=100)
    parser.add_argument("--q-points", type=int, default=121)
    parser.add_argument("--q-min", type=float, default=0.01)
    parser.add_argument("--q-max", type=float, default=0.99)
    parser.add_argument("--rho", type=float, default=0.99)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--basis", choices=["legendre", "bspline"], default="legendre")
    parser.add_argument("--m-type", choices=["smooth", "mixed", "rough"], default="smooth")
    parser.add_argument("--x-w-loading", type=float, default=0.0, help="Stress parameter: add this multiple of W_t to the observed predictor X_t.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument("--show-batch-mean", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else timestamped_result_path("batched_qq_convergence_5panel", timestamp, ".png")
    summary_output = Path(args.summary_output) if args.summary_output else timestamped_result_path("batched_qq_convergence_summary", timestamp, ".csv")
    q_probs = quantile_grid(args.iterations_per_batch, args.q_points, args.q_min, args.q_max)

    print(
        f"Running batched Q-Q convergence: Ts={args.Ts}, batches={args.batches}, "
        f"iterations_per_batch={args.iterations_per_batch}, basis={args.basis}, m_type={args.m_type}, x_w_loading={args.x_w_loading:g}, "
        f"quantile_grid=[{q_probs[0]:.3f}, {q_probs[-1]:.3f}]"
    )

    panel_data = []
    summary_rows = []
    total_t0 = time.time()
    for idx, T in enumerate(args.Ts):
        K = int(np.round(T ** (1 / 3)))
        t0 = time.time()
        stats_by_batch = simulate_batched_stats(T, K, args, base_seed=args.seed + 100_000 * idx + T)
        summary = summarize_batches(stats_by_batch, q_probs)
        elapsed = time.time() - t0
        panel_data.append({"T": T, "K": K, "summary": summary})
        summary_rows.append({
            "T": T,
            "K": K,
            "batches": args.batches,
            "iterations_per_batch": int(stats_by_batch.shape[1]),
            "total_replications": int(stats_by_batch.size),
            "avg_size_5": summary["size_5_mean"],
            "sd_size_5": summary["size_5_sd"],
            "se_size_5": summary["size_5_se"],
            "avg_mean_stat": summary["mean_stat_mean"],
            "avg_var_stat": summary["var_stat_mean"],
            "avg_batch_q95": summary["batch_q95_mean"],
            "pooled_q95": summary["pooled_q95"],
            "chi2_q95": CHI2_5PCT_CV,
            "avg_ks_stat": summary["ks_stat_mean"],
        })
        print(
            f"T={T} K={K}: avg 5% size={summary['size_5_mean'] * 100:.2f}% "
            f"(SE {summary['size_5_se'] * 100:.2f} pp), pooled q95={summary['pooled_q95']:.4f}, "
            f"avg batch q95={summary['batch_q95_mean']:.4f}, elapsed={elapsed:.2f}s"
        )

    write_summary_csv(summary_output, summary_rows)
    print(f"Summary CSV saved as '{summary_output}'.")
    if not args.no_plot:
        plot_batched_convergence(panel_data, q_probs, args, output)
        print(f"Batched five-panel Q-Q plot saved as '{output}'.")
    print(f"Total elapsed: {time.time() - total_t0:.2f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
