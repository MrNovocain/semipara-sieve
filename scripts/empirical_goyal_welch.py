from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pseel.goyal_welch import (  # noqa: E402
    DEFAULT_DOWNLOAD_URL,
    SOURCE_PAGE_URL,
    build_analysis_panel,
    clean_goyal_welch_monthly,
    download_raw_dataset,
    load_monthly_workbook,
    make_comparison_table,
    plot_method_comparison,
    scan_goyal_welch_grid,
    write_latex_table,
    write_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and process the Goyal-Welch equity-premium predictor data.")
    parser.add_argument("--download-url", default=DEFAULT_DOWNLOAD_URL)
    parser.add_argument("--source-page-url", default=SOURCE_PAGE_URL)
    parser.add_argument("--raw-path", default="data/raw/goyal_welch/PredictorData2025.xlsx")
    parser.add_argument("--processed-path", default="data/processed/goyal_welch_monthly.csv")
    parser.add_argument("--provenance-path", default="data/raw/goyal_welch/provenance.json")
    parser.add_argument("--output-dir", default="results/goyal_welch_empirical")
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--x-col", default="bm")
    parser.add_argument("--w-col", default="ntis")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--q-max", type=int, default=1)
    parser.add_argument("--min-size", type=int, default=120)
    parser.add_argument("--grid-step", type=int, default=12)
    parser.add_argument("--penalty-multiplier", type=float, default=1.0)
    parser.add_argument("--skip-download", action="store_true", help="Use the existing raw workbook instead of downloading.")
    parser.add_argument("--smoke", action="store_true", help="Use a lighter break-search grid for quick verification.")
    return parser.parse_args()


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
def main() -> int:
    args = parse_args()
    raw_path = ROOT / args.raw_path
    processed_path = ROOT / args.processed_path
    provenance_path = ROOT / args.provenance_path
    output_dir = ROOT / args.output_dir

    requested_q_max = int(args.q_max)
    paper_q_max = 1
    if args.smoke:
        min_size = min(args.min_size, 60)
        grid_step = max(args.grid_step, 24)
    else:
        min_size = args.min_size
        grid_step = args.grid_step

    if not args.skip_download or not raw_path.exists():
        download_raw_dataset(args.download_url, raw_path)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise FileNotFoundError(f"raw workbook is missing or empty: {raw_path}")

    raw = load_monthly_workbook(raw_path, sheet_name=args.sheet_name)
    cleaned = clean_goyal_welch_monthly(raw)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_path, index=False)

    write_provenance(
        raw_path=raw_path,
        processed_path=processed_path,
        output_path=provenance_path,
        source_page_url=args.source_page_url,
        download_url=args.download_url,
        processed_rows=len(cleaned),
    )

    panel = build_analysis_panel(cleaned, x_col=args.x_col, w_col=args.w_col, dropna=True)
    table = make_comparison_table(
        panel,
        beta0=0.0,
        K=args.K,
        q_max=paper_q_max,
        min_size=min_size,
        grid_step=grid_step,
        penalty_multiplier=args.penalty_multiplier,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "goyal_welch_comparison.csv"
    grid_scan_path = output_dir / "goyal_welch_grid_scan.csv"
    exploratory_path = None
    figure_path = ROOT / "paper" / "figures" / "goyal_welch_method_comparison.png"
    diagnostics_path = output_dir / "goyal_welch_diagnostics.json"
    table.to_csv(comparison_path, index=False)

    if requested_q_max > 1:
        exploratory = make_comparison_table(
            panel,
            beta0=0.0,
            K=args.K,
            q_max=requested_q_max,
            min_size=min_size,
            grid_step=grid_step,
            penalty_multiplier=args.penalty_multiplier,
            allow_exploratory_multiple_breaks=True,
        )
        exploratory_path = output_dir / "goyal_welch_exploratory_multiple_breaks.csv"
        exploratory.to_csv(exploratory_path, index=False)

    x_vars = ["dp", "dy", "ep", "de", "bm", "tbl", "lty", "tms", "dfy", "dfr", "ntis", "infl", "svar"]
    w_vars = ["tms", "dfy", "tbl", "lty", "infl", "svar", "bm", "ntis", "ep"]
    scan = scan_goyal_welch_grid(
        cleaned,
        x_vars=x_vars,
        w_vars=w_vars,
        beta0=0.0,
        K=args.K,
        q_max=paper_q_max,
        min_size=min_size,
        grid_step=grid_step,
        penalty_multiplier=args.penalty_multiplier,
    )
    scan.to_csv(grid_scan_path, index=False)
    write_latex_table(table, ROOT / "paper" / "tables" / "goyal_welch_empirical_comparison.tex")
    plot_method_comparison(table, figure_path, title=f"Goyal-Welch: X={args.x_col}, W={args.w_col}")
    by_method = table.set_index("method")
    standard_row = by_method.loc["standard_predictive_regression"]
    adaptive_row = by_method.loc["one_break_profile_sieve_el"]
    diagnostics = {
        "raw_path": raw_path.as_posix(),
        "processed_path": processed_path.as_posix(),
        "comparison_path": comparison_path.as_posix(),
        "grid_scan_path": grid_scan_path.as_posix(),
        "exploratory_multiple_breaks_path": exploratory_path.as_posix() if exploratory_path is not None else None,
        "figure_path": figure_path.as_posix(),
        "n_cleaned_rows": int(len(cleaned)),
        "n_analysis_rows": int(len(panel)),
        "x_col": args.x_col,
        "w_col": args.w_col,
        "K": int(args.K),
        "requested_q_max": requested_q_max,
        "paper_q_max": paper_q_max,
        "min_size": int(min_size),
        "grid_step": int(grid_step),
        "paper_facing_methods": [str(method) for method in table["method"]],
        "predictor_rho_hat": float(standard_row["rho_hat"]),
        "selected_q": int(adaptive_row["qhat"]),
        "selected_break_dates": str(adaptive_row["break_dates"]),
        "selected_mode": _json_value(adaptive_row["mode"]),
        "criterion_type": _json_value(adaptive_row["criterion_type"]),
        "workbook_penalty": _json_value(adaptive_row["workbook_penalty"]),
        "R_T": _json_value(adaptive_row["R_T"]),
        "r_T": _json_value(adaptive_row["r_T"]),
        "Delta_T": _json_value(adaptive_row["Delta_T"]),
        "kappa_T": _json_value(adaptive_row["kappa_T"]),
        "rate_check_available": _json_value(adaptive_row["rate_check_available"]),
        "rate_check_pass": _json_value(adaptive_row["rate_check_pass"]),
        "empirical_delta_T_available": False,
        "theorem_rate_note": "Goyal-Welch is real data: Delta_T is not known, so R_T is recorded but r_T and Delta_T-based theorem-rate checks are unavailable.",
        "score_weight_note": "W is the nuisance covariate; score_weight is the manuscript's lower-case w_t and defaults to tanh(x_lag).",
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"raw={raw_path}")
    print(f"processed={processed_path} rows={len(cleaned)}")
    print(f"comparison={comparison_path} rows={len(table)}")
    print(f"grid_scan={grid_scan_path} rows={len(scan)}")
    if exploratory_path is not None:
        print(f"exploratory_multiple_breaks={exploratory_path}")
    print(f"figure={figure_path}")
    print(f"provenance={provenance_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


