from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


def plot_chisq_qq(run_dir: Path, method: str = "profile_bounded", T: int | None = None) -> Path:
    raw_path = run_dir / "raw_replications.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_parquet(raw_path)
    data = raw[(raw["method"] == method) & np.isfinite(raw["el_stat"])].copy()
    if T is not None:
        data = data[data["T"].eq(int(T))].copy()
    if data.empty:
        raise ValueError(f"No finite EL statistics found for method={method!r} in {raw_path}")

    panels = list(data.groupby("rho_label", sort=False))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2), squeeze=False)
    for ax, (rho_label, frame) in zip(axes[0], panels):
        observed = np.sort(frame["el_stat"].to_numpy(dtype=float))
        probs = (np.arange(1, len(observed) + 1) - 0.5) / len(observed)
        expected = chi2.ppf(probs, df=1)
        max_val = max(float(np.max(observed)), float(np.max(expected)), chi2.ppf(0.995, 1))
        ax.scatter(expected, observed, s=18, color="black", alpha=0.72)
        ax.plot([0, max_val], [0, max_val], color="#b22222", linewidth=1.5)
        t_label = f", T={int(T)}" if T is not None else ""
        ax.set_title(f"{method}: {rho_label}{t_label}\nR={len(observed)} finite draws")
        ax.set_xlabel(r"Theoretical $\chi^2_1$ quantile")
        ax.set_ylabel("Empirical EL quantile")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.2)
    fig.suptitle(r"Profile-sieve EL QQ check against $\chi^2_1$", y=1.02)
    fig.tight_layout()
    suffix = f"_T{int(T)}" if T is not None else ""
    output = run_dir / f"qq_chisq1_{method}{suffix}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_frontier(run_dir: Path, method: str = "profile_bounded_frontier") -> Path:
    summary_path = run_dir / "summary_frontier.csv"
    if summary_path.exists():
        data = pd.read_csv(summary_path)
    else:
        raw_path = run_dir / "raw_replications.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw = pd.read_parquet(raw_path)
        data = (
            raw[raw["method"].eq(method)]
            .groupby(["T", "rho_label", "rho_value", "K", "weight_b", "method"], dropna=False)
            .agg(
                mean_RE=("RE", "mean"),
                rejection_rate_5=("reject_5", "mean"),
                feasible_rate=("feasible", "mean"),
            )
            .reset_index()
        )
        data["size_distortion_5"] = data["rejection_rate_5"] - 0.05
    data = data[data["method"].eq(method)].copy()
    if data.empty:
        raise ValueError(f"No frontier rows for method={method!r}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (rho_label, frame) in enumerate(data.groupby("rho_label", sort=False)):
        frame = frame.sort_values("weight_b")
        color = colors[idx % len(colors)]
        axes[0].plot(frame["weight_b"], frame["mean_RE"], marker="o", label=rho_label, color=color)
        axes[1].plot(frame["weight_b"], frame["rejection_rate_5"], marker="o", label=rho_label, color=color)
    axes[0].set_ylabel(r"Mean $\widehat{RE}(b)$")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].axhline(0.05, color="black", linewidth=1.0, linestyle="--", label="nominal 5%")
    axes[1].set_ylabel("5% rejection frequency")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"Saturation scale $b$ in $\tanh(X/b)$")
        ax.grid(alpha=0.25)
    axes[0].set_title("Efficiency angle")
    axes[1].set_title("Calibration along the frontier")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Robustness-efficiency frontier for bounded profile-sieve EL", y=1.03)
    fig.tight_layout()
    output = run_dir / "figure_frontier.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Make figures for a pseel run.")
    parser.add_argument("--run-dir", required=True, help="Path to results/<run_id>")
    parser.add_argument("--method", default="profile_bounded")
    parser.add_argument("--fig", choices=["qq", "frontier"], default="qq")
    parser.add_argument("--T", type=int, default=None, help="Optional sample-size filter for QQ plots.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if args.fig == "qq":
        output = plot_chisq_qq(run_dir, args.method, T=args.T)
    else:
        output = plot_frontier(run_dir, args.method if args.method != "profile_bounded" else "profile_bounded_frontier")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())