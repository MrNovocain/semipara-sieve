from pathlib import Path
from uuid import uuid4
import shutil

import pandas as pd

from pseel.diagnostics import (
    DiagnosticScenario,
    default_scenarios,
    run_diagnostics,
    write_diagnostic_outputs,
)


def test_smoke_scenarios_are_labeled():
    scenarios = default_scenarios("smoke")
    assert {scenario.purpose for scenario in scenarios} >= {"core", "negative_intercept"}
    assert all(scenario.T > scenario.K for scenario in scenarios)


def test_run_diagnostics_produces_contract_tables():
    scenario = DiagnosticScenario(
        name="tiny_stationary_contract",
        T=50,
        K=3,
        rho_design={"label": "stationary", "formula": "fixed", "value": 0.5},
        burnin=30,
    )
    result = run_diagnostics([scenario], n_replications=4, seed=123)
    assert not result.raw.empty
    assert set(result.raw["method"]) == {
        "oracle_bounded",
        "profile_bounded",
        "intercept_only_bounded",
        "profile_efficient",
    }
    assert "contract_pass" in result.contract_summary.columns
    assert "projection_contract_pass" in result.deterministic_checks.columns
    assert bool(result.deterministic_checks["projection_contract_pass"].iloc[0])


def test_write_diagnostic_outputs():
    scenario = DiagnosticScenario(
        name="tiny_output_contract",
        T=45,
        K=3,
        rho_design={"label": "stationary", "formula": "fixed", "value": 0.5},
        burnin=20,
    )
    result = run_diagnostics([scenario], n_replications=3, seed=456)
    tmp_root = Path("results") / f"_test_theory_diagnostics_{uuid4().hex}"
    try:
        out = write_diagnostic_outputs(result, tmp_root)
        expected = {
            "raw_replications.csv",
            "method_summary.csv",
            "contract_summary.csv",
            "deterministic_checks.csv",
            "contract_report.json",
        }
        assert expected <= {path.name for path in out.iterdir()}
        contract = pd.read_csv(out / "contract_summary.csv")
        assert len(contract) == 1
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
