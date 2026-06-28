from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pseel.breaks import (  # noqa: E402
    polynomial_segment_basis,
    profile_el_statistic,
    residualize_against,
    select_profile_partition,
)


@dataclass(frozen=True)
class SimData:
    y: np.ndarray
    x: np.ndarray
    w: np.ndarray
    true_breaks: tuple[int, ...]
    rho: float
    rho_label: str
    q0: int
    delta: float


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.nanstd(values))
    centered = values - float(np.nanmean(values))
    return centered / scale if scale > 0 else centered


def simulate_nuisance_break(seed: int, T: int, rho: float, rho_label: str, q0: int, delta: float) -> SimData:
    rng = np.random.default_rng(seed)
    burnin = 300
    a_w = 0.45
    cov = np.array([[1.0, 0.35, 0.0], [0.35, 1.0, 0.15], [0.0, 0.15, 1.0]], dtype=float)
    n = T + burnin + 1
    innov = rng.multivariate_normal(np.zeros(3), cov, size=n)
    u_full = innov[:, 0]
    v_full = innov[:, 1]
    eta_full = innov[:, 2]
    x_full = np.zeros(n, dtype=float)
    w_full = np.zeros(n, dtype=float)
    w_full[0] = rng.normal(0.0, np.sqrt(1.0 / (1.0 - a_w**2)))
    for t in range(1, n):
        w_full[t] = a_w * w_full[t - 1] + eta_full[t]
    sample_start = burnin
    x_full[sample_start] = 0.0
    for t in range(sample_start + 1, n):
        x_full[t] = rho * x_full[t - 1] + v_full[t]

    idx = np.arange(burnin + 1, burnin + T + 1)
    x = x_full[idx - 1]
    w = w_full[idx - 1]
    u = u_full[idx]
    base = 0.45 * np.sin(w) + 0.25 * (w**2 - 1.0)
    if q0 == 0:
        true_breaks: tuple[int, ...] = ()
        shift = np.zeros(T)
    elif q0 == 1:
        br = T // 2
        true_breaks = (br,)
        shift = np.zeros(T)
        shift[br:] = delta
    else:
        raise ValueError("This compact paper run supports q0 in {0, 1}.")
    y = base + shift + u
    return SimData(y=y, x=x, w=w, true_breaks=true_breaks, rho=rho, rho_label=rho_label, q0=q0, delta=delta)


def evaluate_methods(data: SimData, K: int, min_size: int, grid_step: int, penalty: float) -> list[dict[str, object]]:
    selected = select_profile_partition(
        y=data.y,
        x=data.x,
        w=data.w,
        beta0=0.0,
        K=K,
        q_max=1,
        min_size=min_size,
        grid_step=grid_step,
        penalty_multiplier=penalty,
    )
    method_breaks = {
        "stable": (),
        "oracle": data.true_breaks,
        "estimated": selected.breaks,
    }
    stats = {name: profile_el_statistic(data.y, data.x, data.w, 0.0, K, breaks, weight_b=0.8) for name, breaks in method_breaks.items()}
    break_error = np.nan
    if data.q0 == 1 and selected.q == 1:
        break_error = abs(selected.breaks[0] - data.true_breaks[0])
    rows: list[dict[str, object]] = []
    for name, stat in stats.items():
        rows.append(
            {
                "method": name,
                "rho_label": data.rho_label,
                "rho": data.rho,
                "q0": data.q0,
                "delta": data.delta,
                "true_break": data.true_breaks[0] if data.true_breaks else np.nan,
                "qhat": selected.q,
                "estimated_break": selected.breaks[0] if selected.breaks else np.nan,
                "break_abs_error": break_error,
                "el_stat": stat.el_stat,
                "p_value": stat.p_value,
                "reject_5": bool(np.isfinite(stat.el_stat) and stat.el_stat > chi2.ppf(0.95, 1)),
                "feasible": stat.feasible,
                "score": stat.score,
                "score_variance": stat.score_variance,
                "rss": stat.rss,
                "strength": stat.residualized_predictor_strength,
                "score_gap_stable_minus_estimated": stats["stable"].score - stats["estimated"].score,
            }
        )
    return rows


def run_monte_carlo(reps: int, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    K = 4
    T = 240
    min_size = 60
    grid_step = 4
    penalty = 0.75
    rho_grid = [("0.50", 0.50), ("0.90", 0.90), ("0.98", 0.98), ("1.00", 1.00)]
    designs = [(0, 0.0), (1, 1.25)]
    rows: list[dict[str, object]] = []
    seed0 = 20260624
    for q0, delta in designs:
        for rho_label, rho in rho_grid:
            for rep in range(reps):
                seed = seed0 + 100000 * q0 + 1000 * len(rows) + rep
                data = simulate_nuisance_break(seed=seed, T=T, rho=rho, rho_label=rho_label, q0=q0, delta=delta)
                rep_rows = evaluate_methods(data, K=K, min_size=min_size, grid_step=grid_step, penalty=penalty)
                for row in rep_rows:
                    row.update({"rep": rep, "T": T, "K": K, "min_size": min_size, "grid_step": grid_step, "penalty": penalty})
                rows.extend(rep_rows)
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["q0", "delta", "rho_label", "rho", "method"], dropna=False)
        .agg(
            n=("rep", "nunique"),
            rejection_rate_5=("reject_5", "mean"),
            feasible_rate=("feasible", "mean"),
            mean_el_stat=("el_stat", lambda s: float(np.nanmean(pd.Series(s).replace([np.inf, -np.inf], np.nan)))),
            median_p_value=("p_value", "median"),
            qhat_one_rate=("qhat", lambda s: float(np.mean(np.asarray(s) == 1))),
            mean_break_abs_error=("break_abs_error", "mean"),
            mean_score_gap=("score_gap_stable_minus_estimated", "mean"),
            mean_strength=("strength", "mean"),
        )
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "break_mc_raw.csv", index=False)
    summary.to_csv(output_dir / "break_mc_summary.csv", index=False)
    return raw, summary


def make_mc_table(summary: pd.DataFrame, path: Path) -> None:
    pivot = summary.pivot_table(index=["q0", "rho_label"], columns="method", values="rejection_rate_5", aggfunc="first")
    meta = summary.groupby(["q0", "rho_label"], dropna=False).agg(
        qhat_one_rate=("qhat_one_rate", "first"),
        mean_break_abs_error=("mean_break_abs_error", "first"),
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Monte Carlo rejection rates under nuisance breaks}",
        r"\label{tab:break-mc}",
        r"\begin{tabular}{ccccccc}",
        r"\hline",
        "$q_0$ & $\\rho$ & Stable & Oracle & Estimated & $P(\\widehat q=1)$ & Break error " + r"\\",
        r"\hline",
    ]
    for (q0, rho_label), row in pivot.sort_index().iterrows():
        qhat = float(meta.loc[(q0, rho_label), "qhat_one_rate"])
        err = meta.loc[(q0, rho_label), "mean_break_abs_error"]
        err_text = "--" if not np.isfinite(err) else f"{float(err):.1f}"
        lines.append(
            f"{int(q0)} & {rho_label} & {float(row['stable']):.3f} & {float(row['oracle']):.3f} & "
            f"{float(row['estimated']):.3f} & {qhat:.3f} & {err_text} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\begin{minipage}{0.94\linewidth}",
            r"\footnotesize Notes: $T=240$, $K=4$, 300 replications in the submission run unless the script is called with a different replication count. The break design has a level shift of 1.25 at mid-sample. The estimated procedure selects between zero and one nuisance break by the null-imposed profile-sieve information criterion.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def make_mc_plot(summary: pd.DataFrame, path: Path) -> None:
    focus = summary[summary["q0"] == 1].copy()
    methods = ["stable", "oracle", "estimated"]
    labels = {"stable": "Stable", "oracle": "Oracle break", "estimated": "Estimated break"}
    x_labels = list(focus["rho_label"].drop_duplicates())
    x = np.arange(len(x_labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for offset, method in zip([-width, 0.0, width], methods):
        vals = []
        for rho_label in x_labels:
            value = focus[(focus["rho_label"] == rho_label) & (focus["method"] == method)]["rejection_rate_5"].iloc[0]
            vals.append(float(value))
        ax.bar(x + offset, vals, width=width, label=labels[method])
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="5% nominal")
    ax.set_xlabel(r"Predictor persistence $\rho$")
    ax.set_ylabel("Rejection rate at 5%")
    ax.set_ylim(0.0, max(0.25, float(focus["rejection_rate_5"].max()) + 0.06))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    ax.set_title("Omitted nuisance break and spurious predictability")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def empirical_weather_index(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    weather = np.column_stack([zscore(df["PRCP_anom_mean"].to_numpy()), zscore(df["TAVG_anom_mean"].to_numpy())])
    _, _, vt = np.linalg.svd(weather, full_matrices=False)
    loading = vt[0].copy()
    if loading.sum() < 0:
        loading *= -1.0
    index = weather @ loading
    index = zscore(index)
    return index, {"PRCP_loading": float(loading[0]), "TAVG_loading": float(loading[1])}


def run_empirical(output_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(ROOT / "data" / "processed" / "cocoa_ghana.csv", parse_dates=["date"])
    keep = ["date", "log_return", "log_price_lagt", "PRCP_anom_mean", "TAVG_anom_mean"]
    df = df[keep].dropna().sort_values("date").reset_index(drop=True)
    y = 100.0 * df["log_return"].to_numpy(dtype=float)
    x = zscore(df["log_price_lagt"].to_numpy(dtype=float))
    w, loadings = empirical_weather_index(df)
    K = 4
    selected = select_profile_partition(
        y=y,
        x=x,
        w=w,
        beta0=0.0,
        K=K,
        q_max=3,
        min_size=504,
        grid_step=126,
        penalty_multiplier=1.25,
    )
    stable = profile_el_statistic(y, x, w, 0.0, K, (), weight_b=0.8)
    adaptive = profile_el_statistic(y, x, w, 0.0, K, selected.breaks, weight_b=0.8)
    break_dates = [df.loc[b, "date"].strftime("%Y-%m-%d") for b in selected.breaks]
    rows = [
        {
            "method": "stable",
            "el_stat": stable.el_stat,
            "p_value": stable.p_value,
            "qhat": 0,
            "break_dates": "",
            "score": stable.score,
            "strength": stable.residualized_predictor_strength,
            "rss": stable.rss,
        },
        {
            "method": "break_adaptive",
            "el_stat": adaptive.el_stat,
            "p_value": adaptive.p_value,
            "qhat": selected.q,
            "break_dates": ", ".join(break_dates),
            "score": adaptive.score,
            "strength": adaptive.residualized_predictor_strength,
            "rss": adaptive.rss,
        },
    ]
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "ghana_break_empirical.csv", index=False)
    meta: dict[str, object] = {
        "n": int(len(df)),
        "start": df["date"].min().strftime("%Y-%m-%d"),
        "end": df["date"].max().strftime("%Y-%m-%d"),
        "K": K,
        "qhat": selected.q,
        "break_indices": list(selected.breaks),
        "break_dates": break_dates,
        **loadings,
        "score_gap_stable_minus_adaptive": float(stable.score - adaptive.score),
    }
    (output_dir / "ghana_break_empirical_meta.json").write_text(json.dumps(meta, indent=2), encoding="ascii")
    make_empirical_table(result, meta, ROOT / "paper" / "tables" / "ghana_break_empirical.tex")
    make_weather_response_plot(df, y, x, w, selected.breaks, K, ROOT / "paper" / "figures" / "ghana_weather_response.png")
    return result, meta


def make_empirical_table(result: pd.DataFrame, meta: dict[str, object], path: Path) -> None:
    stable = result[result["method"] == "stable"].iloc[0]
    adaptive = result[result["method"] == "break_adaptive"].iloc[0]
    break_text = str(adaptive["break_dates"]) if str(adaptive["break_dates"]) else "None"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Commodity-return predictability with Ghana weather controls}",
        r"\label{tab:ghana-empirical}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r" & Stable nuisance & Break-adaptive nuisance \\",
        r"\hline",
        "EL ratio at $\\beta=0$ & {:.3f} & {:.3f} \\\\".format(float(stable["el_stat"]), float(adaptive["el_stat"])),
        "$p$-value & {:.3f} & {:.3f} \\\\".format(float(stable["p_value"]), float(adaptive["p_value"])),
        "Score & {:.3f} & {:.3f} \\\\".format(float(stable["score"]), float(adaptive["score"])),
        "Pred. strength & {:.3f} & {:.3f} \\\\".format(float(stable["strength"]), float(adaptive["strength"])),
        "$\\widehat q$ & 0 & {} \\\\".format(int(adaptive["qhat"])),
        r"\hline",
        r"\end{tabular}",
        r"\begin{minipage}{0.94\linewidth}",
        "\\footnotesize Notes: The sample has {} daily observations from {} to {}. The weather index is the first principal component of precipitation and temperature anomalies. Estimated nuisance break dates: {}.".format(int(meta["n"]), meta["start"], meta["end"], break_text),
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def make_weather_response_plot(df: pd.DataFrame, y: np.ndarray, x: np.ndarray, w: np.ndarray, breaks: tuple[int, ...], K: int, path: Path) -> None:
    bounds = [0, *breaks, len(y)]
    grid = np.linspace(float(np.quantile(w, 0.02)), float(np.quantile(w, 0.98)), 120)
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for j in range(len(bounds) - 1):
        start, end = int(bounds[j]), int(bounds[j + 1])
        w_seg = w[start:end]
        target = y[start:end]
        mean = float(w_seg.mean())
        scale = float(w_seg.std())
        z_seg = (w_seg - mean) / scale if scale > 0 else w_seg - mean
        P_seg = np.column_stack([z_seg**degree if degree else np.ones_like(z_seg) for degree in range(K)])
        coef, *_ = np.linalg.lstsq(P_seg, target, rcond=None)
        z_grid = (grid - mean) / scale if scale > 0 else grid - mean
        P_grid = np.column_stack([z_grid**degree if degree else np.ones_like(z_grid) for degree in range(K)])
        label_start = df.loc[start, "date"].strftime("%Y")
        label_end = df.loc[end - 1, "date"].strftime("%Y")
        ax.plot(grid, P_grid @ coef, linewidth=1.8, label=f"{label_start}-{label_end}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Weather index")
    ax.set_ylabel("Fitted return component")
    ax.set_title("Regime-specific weather response")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate break-adaptive Monte Carlo and empirical artifacts.")
    parser.add_argument("--mc-reps", type=int, default=300)
    parser.add_argument("--output-dir", default="results/break_adaptive_sinica")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    raw, summary = run_monte_carlo(reps=args.mc_reps, output_dir=output_dir)
    make_mc_table(summary, ROOT / "paper" / "tables" / "break_mc_summary.tex")
    make_mc_plot(summary, ROOT / "paper" / "figures" / "break_mc_false_rejection.png")
    empirical, meta = run_empirical(output_dir=output_dir)
    print(f"mc_rows={len(raw)}")
    print(f"mc_summary={output_dir / 'break_mc_summary.csv'}")
    print(f"empirical_rows={len(empirical)} qhat={meta['qhat']} breaks={meta['break_dates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
