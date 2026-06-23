from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pseel.diagnostics import default_scenarios, run_diagnostics, write_diagnostic_outputs


DEFAULT_REPLICATIONS = {
    "smoke": 80,
    "core": 200,
    "negative": 200,
    "adversarial": 80,
}


def _default_output_dir(preset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / f"theory_diagnostics_{preset}_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run automated theorem-risk diagnostics for profile-sieve empirical likelihood."
    )
    parser.add_argument("--preset", choices=sorted(DEFAULT_REPLICATIONS), default="smoke")
    parser.add_argument("--replications", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-scenarios", type=int, default=None, help="Limit scenarios, useful for adversarial smoke runs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on-contract",
        action="store_true",
        help="Exit with status 1 if any scenario contract fails or any negative control is not detected.",
    )
    args = parser.parse_args()

    scenarios = default_scenarios(args.preset)
    if args.max_scenarios is not None:
        scenarios = scenarios[: args.max_scenarios]
    n_replications = args.replications or DEFAULT_REPLICATIONS[args.preset]
    output_dir = args.output_dir or _default_output_dir(args.preset)

    print(
        f"Running theory diagnostics preset={args.preset} "
        f"scenarios={len(scenarios)} replications={n_replications} seed={args.seed}"
    )
    result = run_diagnostics(scenarios, n_replications=n_replications, seed=args.seed)
    write_diagnostic_outputs(result, output_dir, overwrite=args.overwrite)

    display_cols = [
        "scenario",
        "purpose",
        "projection_pass",
        "oracle_pass",
        "feasible_oracle_pass",
        "negative_control_detected",
        "contract_pass",
        "oracle_rejection_rate_5",
        "profile_rejection_rate_5",
        "profile_mean_abs_DS",
        "profile_mean_abs_DV",
    ]
    print(result.contract_summary[display_cols].to_string(index=False))
    failures = result.contract_summary.loc[~result.contract_summary["contract_pass"], "scenario"].astype(str).tolist()
    print(f"Outputs written to {output_dir}")
    if failures:
        print("Contract failures:")
        for name in failures:
            print(f"  - {name}")
    else:
        print("All scenario contracts passed.")
    return 1 if args.fail_on_contract and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
