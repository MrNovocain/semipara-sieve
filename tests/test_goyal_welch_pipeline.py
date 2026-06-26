import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pseel.goyal_welch import (
    REQUIRED_COMPARISON_METHODS,
    build_analysis_panel,
    clean_goyal_welch_monthly,
    make_comparison_table,
    plot_method_comparison,
    scan_goyal_welch_grid,
    write_provenance,
)


def _sample_raw_monthly(n: int = 72) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    i = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "yyyymm": [int(d.strftime("%Y%m")) for d in dates],
            "Index": 100.0 + i,
            "D12": 3.0 + 0.01 * i,
            "E12": 5.0 + 0.02 * i,
            "b/m": 0.45 + 0.001 * i,
            "tbl": 0.010 + 0.0001 * i,
            "AAA": 0.040 + 0.0001 * i,
            "BAA": 0.055 + 0.0002 * i,
            "lty": 0.030 + 0.00015 * i,
            "ntis": -0.01 + 0.0002 * i,
            "Rfree": 0.001 + 0.00001 * i,
            "infl": 0.002 + 0.00001 * i,
            "ltr": 0.006 + 0.00001 * i,
            "corpr": 0.008 + 0.00002 * i,
            "svar": 0.20 + 0.001 * i,
            "CRSP_SPvw": 0.010 + 0.0003 * i + 0.001 * np.sin(i / 3.0),
        }
    )


def test_clean_goyal_welch_monthly_constructs_predictors_and_lagged_panel():
    raw = _sample_raw_monthly()

    cleaned = clean_goyal_welch_monthly(raw)
    panel = build_analysis_panel(cleaned, x_col="dp", w_col="tms")

    assert list(cleaned.columns[:4]) == ["date", "equity_premium", "market_return", "risk_free"]
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned["date"].is_unique
    assert np.isclose(cleaned.loc[0, "equity_premium"], raw.loc[0, "CRSP_SPvw"] - raw.loc[0, "Rfree"])
    assert np.isclose(cleaned.loc[0, "dp"], np.log(raw.loc[0, "D12"]) - np.log(raw.loc[0, "Index"]))
    assert np.isclose(cleaned.loc[0, "tms"], raw.loc[0, "lty"] - raw.loc[0, "tbl"])
    assert np.isclose(cleaned.loc[0, "dfy"], raw.loc[0, "BAA"] - raw.loc[0, "AAA"])
    assert np.isclose(panel.loc[1, "x_lag"], cleaned.loc[0, "dp"])
    assert np.isclose(panel.loc[1, "w_lag"], cleaned.loc[0, "tms"])


def test_write_provenance_records_hash_sheet_names_and_processing_summary():
    scratch = Path("result/goyal_welch_empirical/test_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    workbook = scratch / "PredictorData2025.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        _sample_raw_monthly(4).to_excel(writer, sheet_name="Monthly", index=False)
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="Annual", index=False)
    processed = scratch / "goyal_welch_monthly.csv"
    processed.write_text("date,equity_premium\n2000-01-31,0.01\n", encoding="utf-8")
    provenance = scratch / "provenance.json"

    write_provenance(
        raw_path=workbook,
        processed_path=processed,
        output_path=provenance,
        source_page_url="https://example.test/source",
        download_url="https://example.test/export.xlsx",
        processed_rows=1,
    )

    data = json.loads(provenance.read_text(encoding="utf-8"))
    assert data["raw_file"]["name"] == "PredictorData2025.xlsx"
    assert len(data["raw_file"]["sha256"]) == 64
    assert data["processed_file"]["rows"] == 1
    assert [sheet["name"] for sheet in data["workbook"]["sheets"]] == ["Monthly", "Annual"]
    shutil.rmtree(scratch, ignore_errors=True)


def test_make_comparison_table_contains_required_empirical_methods():
    cleaned = clean_goyal_welch_monthly(_sample_raw_monthly(96))
    panel = build_analysis_panel(cleaned, x_col="dp", w_col="tms")

    table = make_comparison_table(
        panel,
        beta0=0.0,
        K=2,
        q_max=1,
        min_size=24,
        grid_step=6,
        penalty_multiplier=1.0,
    )

    assert REQUIRED_COMPARISON_METHODS.issubset(set(table["method"]))
    assert "persistent_predictor_correction_benchmark" not in set(table["method"])
    assert "linear_bai_perron_break_model" not in set(table["method"])
    assert "linear_break_benchmark" in set(table["method"])
    assert "block_sieve_break_aware_profile_el" not in set(table["method"])
    assert "one_break_profile_sieve_el" in set(table["method"])
    one_break = table.loc[table["method"] == "one_break_profile_sieve_el"].iloc[0]
    assert one_break["rate_check_available"] is False
    assert pd.isna(one_break["Delta_T"])
    assert table["method"].is_unique
    assert np.all(np.isfinite(table["beta_hat"].dropna()))
    assert np.all(np.isfinite(table["rho_hat"].dropna()))
    adaptive = table.loc[table["method"] == "one_break_profile_sieve_el"].iloc[0]
    assert adaptive["qhat"] in {0, 1}

def test_scan_goyal_welch_grid_ranks_classical_signal_that_weakens_after_profiling():
    cleaned = clean_goyal_welch_monthly(_sample_raw_monthly(144))
    cleaned["bm"] = cleaned["bm"] + np.linspace(-0.2, 0.2, len(cleaned))
    cleaned["ntis"] = np.r_[np.zeros(72), np.ones(72)]

    scan = scan_goyal_welch_grid(
        cleaned,
        x_vars=["bm", "dp"],
        w_vars=["ntis", "tms"],
        beta0=0.0,
        K=2,
        q_max=1,
        min_size=36,
        grid_step=12,
        penalty_multiplier=0.5,
    )

    assert {"x", "w", "std_p", "rho_hat", "sieve_p", "adaptive_p", "adaptive_q", "pattern_score"}.issubset(scan.columns)
    top = scan.sort_values("pattern_score", ascending=False).iloc[0]
    assert top["adaptive_p"] >= top["std_p"]


def test_plot_method_comparison_writes_nonblank_png():
    cleaned = clean_goyal_welch_monthly(_sample_raw_monthly(96))
    panel = build_analysis_panel(cleaned, x_col="bm", w_col="ntis")
    table = make_comparison_table(
        panel,
        beta0=0.0,
        K=2,
        q_max=1,
        min_size=24,
        grid_step=6,
        penalty_multiplier=1.0,
    )
    output = Path("result/goyal_welch_empirical/test_method_comparison.png")

    plot_method_comparison(table, output, title="Test plot")

    assert output.exists()
    assert output.stat().st_size > 1000

def test_goyal_welch_multiple_breaks_require_explicit_exploratory_mode():
    cleaned = clean_goyal_welch_monthly(_sample_raw_monthly(120))
    panel = build_analysis_panel(cleaned, x_col="bm", w_col="ntis")

    with pytest.raises(ValueError, match="one-break"):
        make_comparison_table(
            panel,
            beta0=0.0,
            K=2,
            q_max=2,
            min_size=24,
            grid_step=12,
            penalty_multiplier=1.0,
        )

    exploratory = make_comparison_table(
        panel,
        beta0=0.0,
        K=2,
        q_max=2,
        min_size=24,
        grid_step=12,
        penalty_multiplier=1.0,
        allow_exploratory_multiple_breaks=True,
    )

    assert "one_break_profile_sieve_el" not in set(exploratory["method"])
    assert "exploratory_multiple_breaks" in set(exploratory["method"])
    row = exploratory.loc[exploratory["method"] == "exploratory_multiple_breaks"].iloc[0]
    assert row["mode"] == "exploratory_multiple_breaks"

