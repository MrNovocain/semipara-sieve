import numpy as np

from pseel.workbook_mc import run_workbook_break_monte_carlo, workbook_break_replication


def _paper_params():
    return {
        "a_w": 0.4,
        "kappa": 0.0,
        "xi": 0.0,
        "burnin": 20,
        "break_fraction": 0.5,
        "m_left": {"name": "zero", "params": {"level": 0.0}},
        "m_right": {"name": "zero", "params": {"level": 1.5}},
    }


def test_workbook_break_replication_reports_known_and_estimated_partition_wilks_inputs():
    row = workbook_break_replication(
        seed=123,
        T=120,
        K=1,
        beta0=0.0,
        beta=0.0,
        rho_design={"label": "stationary", "formula": "fixed", "value": 0.4},
        dgp_params=_paper_params(),
        min_size=30,
        grid_step=1,
        a_K=0.0,
        kappa_T=1.0,
    )

    expected = {
        "selected_q",
        "true_break",
        "selected_break",
        "break_error",
        "Delta_T",
        "R_T",
        "r_T",
        "known_el_stat",
        "known_p_value",
        "estimated_el_stat",
        "estimated_p_value",
        "gram_stable",
        "convex_hull_contains_zero",
    }
    assert expected.issubset(row)
    assert row["selected_q"] == 1
    assert abs(row["break_error"]) <= 2
    assert 0.0 <= row["known_p_value"] <= 1.0
    assert 0.0 <= row["estimated_p_value"] <= 1.0
    assert row["R_T"] > 0.0
    assert row["r_T"] > 0.0


def test_run_workbook_break_monte_carlo_returns_one_row_per_replication():
    result = run_workbook_break_monte_carlo(
        seeds=[1, 2, 3],
        T=90,
        K=1,
        beta0=0.0,
        beta=0.0,
        rho_design={"label": "stationary", "formula": "fixed", "value": 0.4},
        dgp_params=_paper_params(),
        min_size=25,
        grid_step=1,
        a_K=0.0,
        kappa_T=1.0,
    )

    assert len(result) == 3
    assert set(result["selected_q"]).issubset({0, 1})
    assert np.all(np.isfinite(result["estimated_el_stat"]))
    assert "break_error" in result.columns

def test_broken_nuisance_one_break_config_targets_workbook_mc():
    import yaml
    from pathlib import Path

    path = Path("configs/mc/broken_nuisance_one_break.yaml")
    assert path.exists()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["dgp"]["name"] == "broken_nuisance_ar1"
    assert config["workbook_breaks"]["q_max"] == 1
    assert config["workbook_breaks"]["selector"] == "workbook_one_break"
    assert config["workbook_breaks"]["report_rate_diagnostics"] is True
    assert config["outputs"]["root"] == "results"

