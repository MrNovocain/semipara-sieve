from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pseel.io import load_yaml  # noqa: E402
from pseel.workbook_mc import run_workbook_break_monte_carlo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the workbook-faithful broken-nuisance one-break Monte Carlo.")
    parser.add_argument("config", nargs="?", default="configs/mc/broken_nuisance_one_break.yaml")
    parser.add_argument("--output-dir", default=None, help="Override the config output directory.")
    parser.add_argument("--replications", type=int, default=None, help="Override the configured replication count.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def _summary(frame: pd.DataFrame, beta: float, beta0: float) -> pd.DataFrame:
    abs_break_error = frame["break_error"].abs().replace([np.inf, -np.inf], np.nan)
    under_null = bool(np.isclose(beta, beta0))
    return pd.DataFrame(
        [
            {
                "n_replications": int(len(frame)),
                "beta": float(beta),
                "beta0": float(beta0),
                "under_null": under_null,
                "empirical_size_estimated_5": float(frame["estimated_reject_5"].mean()) if under_null else np.nan,
                "empirical_size_known_5": float(frame["known_reject_5"].mean()) if under_null else np.nan,
                "el_rejection_rate_estimated_5": float(frame["estimated_reject_5"].mean()),
                "el_rejection_rate_known_5": float(frame["known_reject_5"].mean()),
                "selected_q_mean": float(frame["selected_q"].mean()),
                "selected_one_break_rate": float((frame["selected_q"] == 1).mean()),
                "mean_abs_break_error": float(abs_break_error.mean()),
                "median_abs_break_error": float(abs_break_error.median()),
                "rate_check_available_rate": float(frame["rate_check_available"].mean()),
                "rate_check_pass_rate": float(frame["rate_check_pass"].mean()),
                "mean_r_T_over_sqrt_T": float(frame["r_T_over_sqrt_T"].mean()),
                "mean_kappa_R_over_T_delta_sq": float(frame["kappa_R_over_T_delta_sq"].mean()),
            }
        ]
    )


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    config = load_yaml(config_path)

    experiment = config.get("experiment", {})
    dgp_config = config.get("dgp", {})
    if dgp_config.get("name") != "broken_nuisance_ar1":
        raise ValueError("this runner is for dgp.name=broken_nuisance_ar1")
    workbook = config.get("workbook_breaks", {})
    if int(workbook.get("q_max", 1)) != 1:
        raise ValueError("the paper-faithful workbook MC runner is one-break only")

    n_replications = int(args.replications if args.replications is not None else experiment.get("n_replications", 200))
    global_seed = int(experiment.get("global_seed", 20260701))
    beta0 = float(experiment.get("beta0", 0.0))
    beta = float(experiment.get("beta", beta0))
    dgp_params = dict(dgp_config.get("params", {}))
    T = int(dgp_params.pop("T", 250))
    rho_design = config.get("rho_design", {"label": "stationary_high", "formula": "fixed", "value": 0.95})
    weights = config.get("weights", {}).get("bounded_main", {}).get("params", {})
    weight_b = float(weights.get("b", 8.0))

    seed_sequence = np.random.SeedSequence(global_seed)
    seeds = [int(seed) for seed in seed_sequence.generate_state(n_replications, dtype=np.uint32)]
    frame = run_workbook_break_monte_carlo(
        seeds=seeds,
        T=T,
        K=int(config.get("sieve", {}).get("K", 4)),
        beta0=beta0,
        beta=beta,
        rho_design=rho_design,
        dgp_params=dgp_params,
        min_size=int(workbook.get("min_size", 60)),
        grid_step=int(workbook.get("grid_step", 4)),
        a_K=float(workbook.get("a_K", 0.0)),
        kappa_T=workbook.get("kappa_T"),
        weight_b=weight_b,
    )

    output_root = Path(args.output_dir) if args.output_dir is not None else ROOT / str(config.get("outputs", {}).get("root", "results")) / str(config.get("outputs", {}).get("run_id", "broken_nuisance_one_break"))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if output_root.exists() and any(output_root.iterdir()) and not (args.overwrite or bool(config.get("outputs", {}).get("overwrite", False))):
        raise FileExistsError(f"output directory exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    raw_path = output_root / "workbook_break_raw.csv"
    summary_path = output_root / "workbook_break_summary.csv"
    diagnostics_path = output_root / "workbook_break_diagnostics.json"
    frame.to_csv(raw_path, index=False)
    summary = _summary(frame, beta=beta, beta0=beta0)
    summary.to_csv(summary_path, index=False)
    diagnostics = {
        "config_path": config_path.as_posix(),
        "raw_path": raw_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "dgp_name": dgp_config.get("name"),
        "q0": 1,
        "selector": workbook.get("selector", "workbook_one_break"),
        "q_max": int(workbook.get("q_max", 1)),
        "true_break_mean": float(frame["true_break"].mean()),
        "Delta_T_mean": float(frame["Delta_T"].mean()),
        "report_rate_diagnostics": bool(workbook.get("report_rate_diagnostics", True)),
        "rate_columns": ["r_T_over_sqrt_T", "kappa_R_over_T_delta_sq"],
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"raw={raw_path}")
    print(f"summary={summary_path}")
    print(f"diagnostics={diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
