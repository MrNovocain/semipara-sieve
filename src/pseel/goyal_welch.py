from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, t as student_t

from .breaks import block_sieve_matrix, profile_el_statistic, residualize_against, select_workbook_partition

SOURCE_PAGE_URL = "https://sites.google.com/view/agoyal145/"
DEFAULT_DOWNLOAD_URL = "https://docs.google.com/spreadsheets/d/1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ/export?format=xlsx"

REQUIRED_COMPARISON_METHODS = {
    "standard_predictive_regression",
    "linear_break_benchmark",
    "no_break_sieve_el",
    "one_break_profile_sieve_el",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_raw_dataset(url: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp.write_bytes(response.read())
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"downloaded empty file from {url}")
    tmp.replace(output)
    return output


def load_monthly_workbook(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    workbook = Path(path)
    if sheet_name is not None:
        return pd.read_excel(workbook, sheet_name=sheet_name)

    excel = pd.ExcelFile(workbook)
    errors: list[str] = []
    for candidate in excel.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=candidate)
        try:
            clean_goyal_welch_monthly(frame)
            return frame
        except Exception as exc:  # pragma: no cover - used only for workbook auto-detection.
            errors.append(f"{candidate}: {exc}")
    raise ValueError("no sheet looked like monthly Goyal-Welch data; " + "; ".join(errors))


def clean_goyal_welch_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work.columns = [str(c).strip() for c in work.columns]
    date_col = _find_date_column(work)
    dates = _parse_monthly_dates(work[date_col])
    work = work.loc[dates.notna()].copy()
    dates = dates.loc[dates.notna()]
    work["date"] = dates.to_numpy()

    lookup = _column_lookup(work)
    index = _numeric_column(work, lookup, ["Index", "index"])
    d12 = _numeric_column(work, lookup, ["D12", "d12"])
    e12 = _numeric_column(work, lookup, ["E12", "e12"])
    market_return = _numeric_column(work, lookup, ["CRSP_SPvw", "CRSP_SPvwx", "mktrf", "market_return"])
    risk_free = _numeric_column(work, lookup, ["Rfree", "R_f", "rf", "risk_free"])

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(work["date"]),
            "equity_premium": market_return - risk_free,
            "market_return": market_return,
            "risk_free": risk_free,
            "dp": _safe_log(d12) - _safe_log(index),
            "dy": _safe_log(d12) - _safe_log(index.shift(1)),
            "ep": _safe_log(e12) - _safe_log(index),
            "de": _safe_log(d12) - _safe_log(e12),
        }
    )
    optional_passthrough = {
        "bm": ["b/m", "bm", "book_to_market"],
        "tbl": ["tbl", "TBL"],
        "lty": ["lty", "LTY"],
        "ntis": ["ntis", "NTIS"],
        "infl": ["infl", "Infl"],
        "ltr": ["ltr", "LTR"],
        "corpr": ["corpr", "CORPR"],
        "svar": ["svar", "SVAR"],
    }
    for output_name, names in optional_passthrough.items():
        value = _maybe_numeric_column(work, lookup, names)
        if value is not None:
            result[output_name] = value.to_numpy(dtype=float)

    aaa = _maybe_numeric_column(work, lookup, ["AAA", "aaa"])
    baa = _maybe_numeric_column(work, lookup, ["BAA", "baa"])
    if aaa is not None and baa is not None:
        result["dfy"] = baa.to_numpy(dtype=float) - aaa.to_numpy(dtype=float)
    if "lty" in result and "tbl" in result:
        result["tms"] = result["lty"] - result["tbl"]
    if "corpr" in result and "ltr" in result:
        result["dfr"] = result["corpr"] - result["ltr"]

    result = result.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    _validate_cleaned_monthly(result)
    return result


def build_analysis_panel(cleaned: pd.DataFrame, x_col: str = "dp", w_col: str = "tms", dropna: bool = False) -> pd.DataFrame:
    missing = [col for col in ["date", "equity_premium", x_col, w_col] if col not in cleaned.columns]
    if missing:
        raise ValueError(f"cleaned monthly data is missing required columns: {missing}")
    panel = cleaned.copy()
    panel["y"] = pd.to_numeric(panel["equity_premium"], errors="coerce")
    panel["x_lag"] = pd.to_numeric(panel[x_col], errors="coerce").shift(1)
    panel["w_lag"] = pd.to_numeric(panel[w_col], errors="coerce").shift(1)
    panel["x_variable"] = x_col
    panel["w_variable"] = w_col
    if dropna:
        panel = panel.dropna(subset=["date", "y", "x_lag", "w_lag"]).reset_index(drop=True)
    return panel


def make_comparison_table(
    panel: pd.DataFrame,
    beta0: float,
    K: int,
    q_max: int,
    min_size: int,
    grid_step: int,
    penalty_multiplier: float,
    allow_exploratory_multiple_breaks: bool = False,
) -> pd.DataFrame:
    q_max = int(q_max)
    if q_max > 1 and not allow_exploratory_multiple_breaks:
        raise ValueError("paper-facing Goyal-Welch comparison is one-break; pass allow_exploratory_multiple_breaks=True for exploratory multi-break output")

    data = panel.dropna(subset=["date", "y", "x_lag", "w_lag"]).copy().reset_index(drop=True)
    y = data["y"].to_numpy(dtype=float)
    x = _zscore(data["x_lag"].to_numpy(dtype=float))
    nuisance_w = _zscore(data["w_lag"].to_numpy(dtype=float))
    score_weight = np.tanh(x)
    dates = pd.to_datetime(data["date"]).reset_index(drop=True)
    if len(y) < max(20, 2 * min_size):
        raise ValueError("analysis panel is too short for the requested break search")

    standard = _ols_with_controls(y, x, np.ones((len(y), 1)))
    rho_hat = _ar1_rho(x)

    linear_breaks = _select_linear_breaks(
        y,
        x,
        q_max=q_max,
        min_size=min_size,
        grid_step=grid_step,
        penalty_multiplier=penalty_multiplier,
        allow_multiple_breaks=allow_exploratory_multiple_breaks,
    )
    linear_controls = _segment_intercepts(len(y), linear_breaks)
    linear = _ols_with_controls(y, x, linear_controls)

    stable = profile_el_statistic(y=y, x=x, nuisance_w=nuisance_w, beta0=beta0, K=K, breaks=(), score_weight=score_weight)
    stable_beta = _profile_beta_hat(y, x, nuisance_w, K=K, breaks=())
    selected = select_workbook_partition(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=beta0,
        K=K,
        q_max=q_max,
        min_size=min_size,
        grid_step=grid_step,
        a_K=0.0,
        delta_T=None,
        allow_multiple_breaks=allow_exploratory_multiple_breaks,
    )
    selected_metadata = next(row for row in selected.by_q if int(row["q"]) == int(selected.q))
    adaptive = profile_el_statistic(
        y=y,
        x=x,
        nuisance_w=nuisance_w,
        beta0=beta0,
        K=K,
        breaks=selected.breaks,
        score_weight=score_weight,
    )
    adaptive_beta = _profile_beta_hat(y, x, nuisance_w, K=K, breaks=selected.breaks)
    adaptive_method = "exploratory_multiple_breaks" if q_max > 1 else "one_break_profile_sieve_el"

    empty_metadata: dict[str, object] = {
        "criterion_type": None,
        "mode": None,
        "workbook_penalty": None,
        "R_T": None,
        "r_T": None,
        "Delta_T": None,
        "kappa_T": None,
        "rate_check_available": None,
        "rate_check_pass": None,
        "r_T_over_sqrt_T": None,
        "kappa_R_over_T_delta_sq": None,
        "score_weight": "tanh(x_lag)",
    }
    adaptive_metadata = {
        "criterion_type": selected_metadata["criterion_type"],
        "mode": selected_metadata["mode"],
        "workbook_penalty": selected_metadata["workbook_penalty"],
        "R_T": selected_metadata["R_T"],
        "r_T": selected_metadata["r_T"],
        "Delta_T": selected_metadata["Delta_T"],
        "kappa_T": selected_metadata["kappa_T"],
        "rate_check_available": selected_metadata["rate_check_available"],
        "rate_check_pass": selected_metadata["rate_check_pass"],
        "r_T_over_sqrt_T": selected_metadata["r_T_over_sqrt_T"],
        "kappa_R_over_T_delta_sq": selected_metadata["kappa_R_over_T_delta_sq"],
        "score_weight": "tanh(x_lag)",
    }

    rows = [
        {
            **empty_metadata,
            "method": "standard_predictive_regression",
            "beta_hat": standard["beta_hat"],
            "t_stat": standard["t_stat"],
            "p_value": standard["p_value"],
            "el_stat": np.nan,
            "qhat": 0,
            "break_dates": "",
            "rho_hat": rho_hat,
            "notes": "OLS predictive regression of equity premium on lagged predictor.",
        },
        {
            **empty_metadata,
            "method": "linear_break_benchmark",
            "beta_hat": linear["beta_hat"],
            "t_stat": linear["t_stat"],
            "p_value": linear["p_value"],
            "el_stat": np.nan,
            "qhat": len(linear_breaks),
            "break_dates": _format_break_dates(dates, linear_breaks),
            "rho_hat": rho_hat,
            "notes": "Local piecewise-intercept linear break benchmark selected by the workbook RSS-plus-penalty criterion.",
        },
        {
            **empty_metadata,
            "method": "no_break_sieve_el",
            "beta_hat": stable_beta,
            "t_stat": np.nan,
            "p_value": stable.p_value,
            "el_stat": stable.el_stat,
            "qhat": 0,
            "break_dates": "",
            "rho_hat": rho_hat,
            "notes": "Profile sieve empirical likelihood with a stable nonlinear nuisance; W is the nuisance covariate and score_weight is the manuscript's lower-case w_t.",
        },
        {
            **adaptive_metadata,
            "method": adaptive_method,
            "beta_hat": adaptive_beta,
            "t_stat": np.nan,
            "p_value": adaptive.p_value,
            "el_stat": adaptive.el_stat,
            "qhat": selected.q,
            "break_dates": _format_break_dates(dates, selected.breaks),
            "rho_hat": rho_hat,
            "notes": "Paper-facing one-break profile sieve EL; W is the nuisance covariate and score_weight is the manuscript's lower-case w_t." if q_max <= 1 else "Exploratory multiple-break profile sieve EL; not current main-theorem paper evidence.",
        },
    ]
    return pd.DataFrame(rows)


def scan_goyal_welch_grid(
    cleaned: pd.DataFrame,
    x_vars: list[str],
    w_vars: list[str],
    beta0: float,
    K: int,
    q_max: int,
    min_size: int,
    grid_step: int,
    penalty_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for x_col in x_vars:
        for w_col in w_vars:
            if x_col == w_col or x_col not in cleaned.columns or w_col not in cleaned.columns:
                continue
            panel = build_analysis_panel(cleaned, x_col=x_col, w_col=w_col, dropna=True)
            if len(panel) < max(2 * min_size, 60):
                continue
            try:
                table = make_comparison_table(
                    panel,
                    beta0=beta0,
                    K=K,
                    q_max=q_max,
                    min_size=min_size,
                    grid_step=grid_step,
                    penalty_multiplier=penalty_multiplier,
                )
            except Exception as exc:
                rows.append({"x": x_col, "w": w_col, "n": len(panel), "error": str(exc)})
                continue
            by_method = {str(row["method"]): row for _, row in table.iterrows()}
            standard = by_method["standard_predictive_regression"]
            sieve = by_method["no_break_sieve_el"]
            adaptive = by_method["one_break_profile_sieve_el"]
            std_p = float(standard["p_value"])
            adaptive_p = float(adaptive["p_value"])
            rows.append(
                {
                    "x": x_col,
                    "w": w_col,
                    "n": len(panel),
                    "std_beta": float(standard["beta_hat"]),
                    "std_p": std_p,
                    "rho_hat": float(standard["rho_hat"]),
                    "sieve_p": float(sieve["p_value"]),
                    "adaptive_p": adaptive_p,
                    "adaptive_q": int(adaptive["qhat"]),
                    "adaptive_breaks": str(adaptive["break_dates"]),
                    "pattern_score": _visual_pattern_score(std_p, float(sieve["p_value"]), adaptive_p, int(adaptive["qhat"])),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty and "pattern_score" in result.columns:
        result = result.sort_values("pattern_score", ascending=False).reset_index(drop=True)
    return result


def plot_method_comparison(table: pd.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    order = [
        "standard_predictive_regression",
        "linear_break_benchmark",
        "no_break_sieve_el",
        "one_break_profile_sieve_el",
    ]
    labels = {
        "standard_predictive_regression": "OLS",
        "linear_break_benchmark": "Linear\nbreak",
        "no_break_sieve_el": "No-break\nsieve EL",
        "one_break_profile_sieve_el": "Break-aware\nsieve EL",
    }
    data = table.set_index("method").loc[order].reset_index()
    p_values = data["p_value"].astype(float).to_numpy()
    beta_values = data["beta_hat"].astype(float).to_numpy()
    x = np.arange(len(data))
    colors = ["#9c3b2e", "#637487", "#376d5a", "#264f8f"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.45, 1.05]})
    ax = axes[0]
    ax.bar(x, p_values, color=colors, width=0.72)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in data["method"]], fontsize=8, rotation=18, ha="right")
    ax.set_ylim(0.0, min(1.0, max(0.12, float(np.nanmax(p_values)) * 1.18)))
    ax.set_ylabel("$p$-value")
    ax.set_title(title)
    for xi, value in zip(x, p_values):
        label_y = max(value + 0.025, 0.065) if value < 0.055 else value + 0.025
        ax.text(xi, min(label_y, ax.get_ylim()[1] * 0.95), f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    ax_beta = axes[1]
    ax_beta.axhline(0.0, color="black", linewidth=0.8)
    ax_beta.bar(x, beta_values, color=colors, width=0.72)
    ax_beta.set_xticks(x)
    ax_beta.set_xticklabels([labels[m] for m in data["method"]], fontsize=8, rotation=18, ha="right")
    ax_beta.set_ylabel(r"$\widehat\beta$")
    ax_beta.set_title("Slope estimate")

    adaptive = data[data["method"] == "one_break_profile_sieve_el"].iloc[0]
    break_text = str(adaptive["break_dates"]) if str(adaptive["break_dates"]) else "none"
    fig.text(0.02, 0.01, f"Selected nuisance breaks: q={int(adaptive['qhat'])}, dates={break_text}", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

def write_provenance(
    raw_path: str | Path,
    processed_path: str | Path,
    output_path: str | Path,
    source_page_url: str,
    download_url: str,
    processed_rows: int,
) -> dict[str, Any]:
    raw = Path(raw_path)
    processed = Path(processed_path)
    workbook = _workbook_metadata(raw)
    data: dict[str, Any] = {
        "source_page_url": source_page_url,
        "download_url": download_url,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": {
            "name": raw.name,
            "path": raw.as_posix(),
            "size_bytes": raw.stat().st_size,
            "sha256": sha256_file(raw),
        },
        "processed_file": {
            "name": processed.name,
            "path": processed.as_posix(),
            "exists": processed.exists(),
            "rows": int(processed_rows),
            "sha256": sha256_file(processed) if processed.exists() else None,
        },
        "workbook": workbook,
        "processing": {
            "panel": "monthly",
            "dependent_variable": "equity_premium = CRSP_SPvw - Rfree",
            "default_x": "dp",
            "default_w": "tms",
            "lag_convention": "x_lag and w_lag are one-month lags of the cleaned predictor columns.",
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def write_latex_table(table: pd.DataFrame, path: str | Path) -> None:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Goyal--Welch empirical comparison}",
        r"\label{tab:goyal-welch-empirical-comparison}",
        r"\begin{tabular}{lrrrrl}",
        r"\hline",
        r"Method & $\widehat\beta$ & $t$ / EL & $p$-value & $\widehat q$ & Break dates \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        stat = row["t_stat"] if np.isfinite(row["t_stat"]) else row["el_stat"]
        breaks = str(row["break_dates"]) if str(row["break_dates"]) else "--"
        rows.append(
            "{} & {:.4f} & {:.3f} & {:.3f} & {} & {} \\\\".format(
                str(row["method"]).replace("_", r"\_"),
                float(row["beta_hat"]),
                float(stat) if np.isfinite(stat) else np.nan,
                float(row["p_value"]) if np.isfinite(row["p_value"]) else np.nan,
                int(row["qhat"]),
                breaks,
            )
        )
    rows.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\begin{minipage}{0.94\linewidth}",
            r"\footnotesize Notes: This is the paper-facing Goyal--Welch one-break comparison. $W$ is the nuisance covariate; score\_weight is the manuscript lower-case $w_t$ and defaults to a bounded tanh transform. The machine-readable CSV reports $\widehat\rho$ for persistence diagnostics; no Campbell--Yogo or Stambaugh correction is claimed here.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows), encoding="ascii")


def _visual_pattern_score(std_p: float, sieve_p: float, adaptive_p: float, adaptive_q: int) -> float:
    classical_gate = 1.0 if std_p <= 0.05 else 0.25
    classical_signal = 2.0 * max(0.0, 0.05 - std_p)
    nonlinear_weakening = max(0.0, sieve_p - std_p)
    break_weakening = max(0.0, adaptive_p - sieve_p)
    selected_break_bonus = 0.10 * max(0, adaptive_q)
    return float(classical_signal + classical_gate * (nonlinear_weakening + break_weakening + selected_break_bonus))

def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): str(col).strip() for col in frame.columns}


def _find_column(lookup: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _find_date_column(frame: pd.DataFrame) -> str:
    lookup = _column_lookup(frame)
    named = _find_column(lookup, ["yyyymm", "YYYYMM", "date", "month"])
    if named is not None:
        return named
    for col in frame.columns:
        parsed = _parse_monthly_dates(frame[col])
        if parsed.notna().sum() >= max(3, int(0.5 * len(frame))):
            return str(col)
    raise ValueError("could not identify a monthly date column")


def _parse_monthly_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce") + pd.offsets.MonthEnd(0)
    numeric = pd.to_numeric(values, errors="coerce")
    year = (numeric // 100).astype("Int64")
    month = (numeric % 100).astype("Int64")
    valid = year.between(1800, 2200) & month.between(1, 12)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[valid] = pd.to_datetime(
        {
            "year": year.loc[valid].astype(int),
            "month": month.loc[valid].astype(int),
            "day": 1,
        },
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    return parsed


def _numeric_column(frame: pd.DataFrame, lookup: dict[str, str], candidates: list[str]) -> pd.Series:
    col = _find_column(lookup, candidates)
    if col is None:
        raise ValueError(f"missing required column; tried {candidates}")
    return pd.to_numeric(frame[col], errors="coerce")


def _maybe_numeric_column(frame: pd.DataFrame, lookup: dict[str, str], candidates: list[str]) -> pd.Series | None:
    col = _find_column(lookup, candidates)
    if col is None:
        return None
    return pd.to_numeric(frame[col], errors="coerce")


def _safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def _validate_cleaned_monthly(frame: pd.DataFrame) -> None:
    required = ["date", "equity_premium", "dp", "tms"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"cleaned Goyal-Welch data is missing required columns: {missing}")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("cleaned Goyal-Welch dates must be sorted")
    if frame["date"].duplicated().any():
        raise ValueError("cleaned Goyal-Welch dates must be unique")


def _zscore(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.nanmean(values))
    scale = float(np.nanstd(centered))
    return centered / scale if scale > 0 else centered


def _ols_with_controls(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> dict[str, float]:
    X = np.column_stack([controls, x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    df = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / df)
    inv = np.linalg.pinv(X.T @ X)
    se = math.sqrt(max(sigma2 * inv[-1, -1], 0.0))
    beta_hat = float(coef[-1])
    t_stat = float(beta_hat / se) if se > 0 else np.nan
    p_value = float(2.0 * student_t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan
    return {"beta_hat": beta_hat, "t_stat": t_stat, "p_value": p_value}


def _ar1_rho(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    left = x[:-1] - float(np.mean(x[:-1]))
    right = x[1:] - float(np.mean(x[1:]))
    denom = float(left @ left)
    return float(left @ right / denom) if denom > 0 else np.nan


def _segment_intercepts(T: int, breaks: tuple[int, ...]) -> np.ndarray:
    controls = np.zeros((T, len(breaks) + 1), dtype=float)
    bounds = [0, *breaks, T]
    for j in range(len(bounds) - 1):
        controls[bounds[j] : bounds[j + 1], j] = 1.0
    return controls


def _select_linear_breaks(
    y: np.ndarray,
    x: np.ndarray,
    q_max: int,
    min_size: int,
    grid_step: int,
    penalty_multiplier: float,
    allow_multiple_breaks: bool = False,
) -> tuple[int, ...]:
    zeros = np.zeros_like(x)
    selected = select_workbook_partition(
        y=y,
        x=x,
        nuisance_w=zeros,
        beta0=0.0,
        K=1,
        q_max=q_max,
        min_size=min_size,
        grid_step=grid_step,
        a_K=0.0,
        delta_T=None,
        allow_multiple_breaks=allow_multiple_breaks,
    )
    return selected.breaks


def _profile_beta_hat(y: np.ndarray, x: np.ndarray, w: np.ndarray, K: int, breaks: tuple[int, ...]) -> float:
    P = block_sieve_matrix(w, breaks, K)
    y_resid = residualize_against(P, y)
    x_resid = residualize_against(P, x)
    denom = float(x_resid @ x_resid)
    return float(x_resid @ y_resid / denom) if denom > 0 else np.nan


def _format_break_dates(dates: pd.Series, breaks: tuple[int, ...]) -> str:
    labels = []
    for break_index in breaks:
        if 0 <= break_index < len(dates):
            labels.append(pd.Timestamp(dates.iloc[break_index]).strftime("%Y-%m"))
    return ", ".join(labels)


def _workbook_metadata(path: Path) -> dict[str, Any]:
    excel = pd.ExcelFile(path)
    sheets = []
    for sheet in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        sheets.append({"name": sheet, "rows": int(len(frame)), "columns": int(len(frame.columns))})
    return {"sheets": sheets}





