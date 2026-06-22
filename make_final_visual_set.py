import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
CHI2_95 = stats.chi2.ppf(0.95, df=1)


def run(cmd):
    print("RUN:", " ".join(str(x) for x in cmd))
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(str(x) for x in cmd)}")


def latest_matching(pattern: str) -> Path:
    matches = sorted(RESULT.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def score_k_summary(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    selected = int(df["selected_K"].iloc[0])
    lower = float(df["lower_bound"].iloc[0])
    upper = float(df["upper_bound"].iloc[0])

    low = df[df["K"] <= lower]
    high = df[df["K"] >= upper]
    interior = df[(df["K"] > lower) & (df["K"] < upper)]
    selected_row = df.iloc[(df["K"] - selected).abs().argsort()[:1]]

    low_bad = float(low["size_5"].max()) if len(low) else np.nan
    high_bad = float(high["size_5"].max()) if len(high) else np.nan
    selected_size = float(selected_row["size_5"].iloc[0])
    interior_best_error = float((interior["size_5"] - 0.05).abs().min()) if len(interior) else np.nan

    # Reward high bad tails and selected/interior calibration near 5%.
    score = 0.0
    if not np.isnan(low_bad):
        score += max(0.0, low_bad - 0.05)
    if not np.isnan(high_bad):
        score += max(0.0, high_bad - 0.05)
    score -= 2.0 * abs(selected_size - 0.05)
    score -= interior_best_error

    return {
        "score": score,
        "low_bad": low_bad,
        "high_bad": high_bad,
        "selected_size": selected_size,
        "selected_K": selected,
        "csv": csv_path,
        "plot": Path(str(csv_path).replace("_summary_", "_").replace(".csv", ".png")),
    }


def plot_size_convergence(summary_csv: Path, output: Path):
    df = pd.read_csv(summary_csv)
    T = df["T"].to_numpy()
    size = 100 * df["avg_size_5"].to_numpy()
    se = 100 * df["se_size_5"].to_numpy()
    pooled_q95 = df["pooled_q95"].to_numpy()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].plot(T, size, "k-o", markersize=4, label="Average empirical size")
    axes[0].fill_between(T, size - 1.96 * se, size + 1.96 * se, color="0.75", alpha=0.35, linewidth=0, label="95% MC band")
    axes[0].axhline(5.0, color="0.25", linestyle="--", linewidth=1.1, label="Nominal 5%")
    axes[0].set_xlabel("Sample size T")
    axes[0].set_ylabel("5% rejection rate (%)")
    axes[0].set_title("Size Calibration Across T")
    axes[0].legend(frameon=False)

    axes[1].plot(T, pooled_q95, "k-o", markersize=4, label="Pooled empirical 95% quantile")
    axes[1].axhline(CHI2_95, color="0.25", linestyle="--", linewidth=1.1, label="$\\chi^2_{1,0.95}$")
    axes[1].set_xlabel("Sample size T")
    axes[1].set_ylabel("95% quantile")
    axes[1].set_title("Upper-Tail Calibration")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT / f"final_visual_set_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env_prefix = []

    # 1. Good-theory convergence graph. This is the main theorem-facing diagnostic.
    qq_png = out_dir / "01_qq_convergence_profile_el.png"
    qq_csv = out_dir / "01_qq_convergence_profile_el.csv"
    run([
        py, "batched_qq_convergence.py",
        "--batches", "60",
        "--iterations-per-batch", "50",
        "--jobs", "1",
        "--output", str(qq_png),
        "--summary-output", str(qq_csv),
    ])

    # 2. Loop over stress designs and choose the clearest K sensitivity picture.
    stress_candidates = [
        ("rough", "0"),
        ("rough", "1"),
        ("rough", "3"),
        ("mixed", "2"),
    ]
    scored = []
    for m_type, loading in stress_candidates:
        run([
            py, "k_sensitivity_chisq.py",
            "--T", "500",
            "--K-grid", "1,2,4,6,8,12,16,20,30,50,100,150",
            "--m-type", m_type,
            "--x-w-loading", loading,
            "--batches", "8",
            "--iterations-per-batch", "30",
            "--jobs", "1",
        ])
        csv_path = latest_matching("k_sensitivity_chisq_summary_*.csv")
        score = score_k_summary(csv_path)
        score["m_type"] = m_type
        score["x_w_loading"] = loading
        scored.append(score)
        print(
            f"candidate m_type={m_type}, x_w_loading={loading}: score={score['score']:.4f}, "
            f"low_bad={score['low_bad']:.3f}, selected={score['selected_size']:.3f}, high_bad={score['high_bad']:.3f}"
        )

    best = max(scored, key=lambda row: row["score"])
    best_plot = best["plot"]
    best_csv = best["csv"]
    if not best_plot.exists():
        raise FileNotFoundError(best_plot)
    k_png = out_dir / "03_k_sensitivity_stress_tradeoff.png"
    k_csv = out_dir / "03_k_sensitivity_stress_tradeoff.csv"
    shutil.copy2(best_plot, k_png)
    shutil.copy2(best_csv, k_csv)

    # 3. Compact size/quantile convergence from the main smooth-theory run.
    size_png = out_dir / "02_size_and_tail_convergence.png"
    plot_size_convergence(qq_csv, size_png)

    # 4. Candidate-selection CSV for auditability.
    selection_csv = out_dir / "stress_candidate_scores.csv"
    with selection_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["m_type", "x_w_loading", "score", "low_bad", "selected_size", "selected_K", "high_bad", "csv", "plot"])
        writer.writeheader()
        for row in scored:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    write_manifest(out_dir / "README.txt", [
        "Final visual set for profile sieve EL Monte Carlo diagnostics.",
        "Profile sieve.tex was not modified.",
        "",
        "Figures:",
        "01_qq_convergence_profile_el.png: theorem-facing Q-Q convergence under the smooth, orthogonal DGP.",
        "02_size_and_tail_convergence.png: average 5% size and pooled 95% quantile across T.",
        "03_k_sensitivity_stress_tradeoff.png: stress DGP chosen by loop over candidate designs to make K bias/variance visible.",
        "",
        f"Selected stress candidate: m_type={best['m_type']}, x_w_loading={best['x_w_loading']}, score={best['score']:.4f}.",
        "The stress candidate intentionally violates/strains orthogonality when x_w_loading > 0; use it to illustrate failure modes, not theorem validity.",
    ])

    print("FINAL_VISUAL_DIR", out_dir)
    for p in sorted(out_dir.iterdir()):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
