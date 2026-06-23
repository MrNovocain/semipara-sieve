from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _group_keys(raw: pd.DataFrame) -> list[str]:
    keys = ["T", "rho_label", "rho_value", "K"]
    if "weight_b" in raw.columns:
        keys.append("weight_b")
    keys.append("method")
    return keys


def make_size_table(raw: pd.DataFrame) -> pd.DataFrame:
    finite_stat = raw["el_stat"].replace([np.inf, -np.inf], np.nan)
    tmp = raw.copy()
    tmp["finite_el_stat"] = finite_stat
    return (
        tmp.groupby(_group_keys(tmp), dropna=False)
        .agg(
            n=("rep", "nunique"),
            rejection_rate_5=("reject_5", "mean"),
            feasible_rate=("feasible", "mean"),
            mean_el_stat=("finite_el_stat", "mean"),
            median_el_stat=("finite_el_stat", "median"),
            mean_RE=("RE", "mean"),
            mean_oracle_var=("oracle_var", "mean") if "oracle_var" in tmp.columns else ("finite_el_stat", "size"),
        )
        .reset_index()
    )


def make_oracle_equiv_table(raw: pd.DataFrame) -> pd.DataFrame:
    diagnostics = raw[raw["method"].isin(["profile_bounded", "profile_bounded_frontier"])].copy()
    if diagnostics.empty:
        return pd.DataFrame()
    return (
        diagnostics.groupby(_group_keys(diagnostics), dropna=False)
        .agg(
            n=("rep", "nunique"),
            mean_abs_DS=("DS", lambda s: float(np.nanmean(np.abs(s)))),
            sd_DS=("DS", "std"),
            mean_abs_DV=("DV", lambda s: float(np.nanmean(np.abs(s)))),
            sd_DV=("DV", "std"),
            mean_RE=("RE", "mean"),
            mean_oracle_var=("oracle_var", "mean"),
            mean_instrument_var=("instrument_var", "mean"),
            mean_efficient_var=("efficient_var", "mean"),
        )
        .reset_index()
    )


def make_frontier_table(raw: pd.DataFrame) -> pd.DataFrame:
    if "weight_b" not in raw.columns:
        return pd.DataFrame()
    frontier = raw[raw["method"].isin(["profile_bounded_frontier", "profile_bounded"])].copy()
    if frontier.empty:
        return pd.DataFrame()
    table = (
        frontier.groupby(["T", "rho_label", "rho_value", "K", "weight_b", "method"], dropna=False)
        .agg(
            n=("rep", "nunique"),
            mean_RE=("RE", "mean"),
            rejection_rate_5=("reject_5", "mean"),
            feasible_rate=("feasible", "mean"),
            mean_el_stat=("el_stat", lambda s: float(np.nanmean(s.replace([np.inf, -np.inf], np.nan)))),
            mean_oracle_var=("oracle_var", "mean"),
            mean_instrument_var=("instrument_var", "mean"),
            mean_efficient_var=("efficient_var", "mean"),
        )
        .reset_index()
    )
    table["size_distortion_5"] = table["rejection_rate_5"] - 0.05
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic tables from a pseel run.")
    parser.add_argument("--run-dir", required=True, help="Path to results/<run_id>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_path = run_dir / "raw_replications.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_parquet(raw_path)

    size = make_size_table(raw)
    oracle_equiv = make_oracle_equiv_table(raw)
    frontier = make_frontier_table(raw)

    size_path = run_dir / "summary_size.csv"
    oracle_path = run_dir / "summary_oracle_equiv.csv"
    frontier_path = run_dir / "summary_frontier.csv"
    size.to_csv(size_path, index=False)
    oracle_equiv.to_csv(oracle_path, index=False)
    frontier.to_csv(frontier_path, index=False)
    print(size_path)
    print(oracle_path)
    print(frontier_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())