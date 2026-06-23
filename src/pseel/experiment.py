from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .basis import PolynomialBasis  # noqa: F401 - ensure basis registration
from .checks import validate_config
from .dgp import PredictiveAR1DGP  # noqa: F401 - ensure DGP registration
from .io import config_hash, make_run_id, write_run_outputs
from .methods import BaseMethod  # noqa: F401 - ensure methods registration
from .registry import DGP_REGISTRY, METHOD_REGISTRY


@dataclass
class ExperimentResult:
    run_id: str
    run_dir: Path
    raw: pd.DataFrame
    summary: pd.DataFrame


class ExperimentRunner:
    def __init__(self, config: dict[str, Any], output_root: str | Path | None = None, overwrite: bool = False):
        validate_config(config)
        self.config = config
        self.digest = config_hash(config)
        self.run_id = make_run_id(config, self.digest)
        root = output_root or config["outputs"].get("root", "results")
        self.output_root = Path(root)
        self.run_dir = self.output_root / self.run_id
        self.overwrite = overwrite or bool(config["outputs"].get("overwrite", False))
        if self.run_dir.exists() and not self.overwrite:
            raise FileExistsError(f"Run directory exists: {self.run_dir}. Use --overwrite or outputs.overwrite=true.")
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        self.logs.append(line)
        print(message)

    def _methods(self, weight_b: float | None = None) -> list:
        basis_config = self.config["sieve"]
        weight_config = deepcopy(self.config["weights"]["bounded_main"])
        if weight_b is not None and weight_config.get("name") == "tanh":
            weight_config.setdefault("params", {})["b"] = float(weight_b)
        el_config = self.config.get("el", {})
        methods = []
        for name in self.config["methods"]:
            cls = METHOD_REGISTRY[name]
            methods.append(cls(name, basis_config, weight_config, el_config))
        return methods

    def _weight_grid(self) -> list[float]:
        frontier_b_values = self.config.get("frontier", {}).get("b_values")
        if frontier_b_values:
            return [float(b) for b in frontier_b_values]
        return [float(self.config["weights"]["bounded_main"].get("params", {}).get("b", np.nan))]

    def _summarize(self, raw: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["T", "rho_label", "rho_value", "K", "weight_b", "method"]
        aggregations = {
            "el_stat": "mean",
            "reject_5": "mean",
            "feasible": "mean",
            "DS": lambda s: float(np.nanmean(np.abs(s))) if s.notna().any() else np.nan,
            "DV": lambda s: float(np.nanmean(np.abs(s))) if s.notna().any() else np.nan,
            "RE": "mean",
            "oracle_var": "mean",
            "instrument_var": "mean",
            "efficient_var": "mean",
        }
        available = {key: value for key, value in aggregations.items() if key in raw.columns}
        summary = raw.groupby(group_cols, dropna=False).agg(available).reset_index()
        summary = summary.rename(
            columns={
                "el_stat": "mean_el_stat",
                "reject_5": "rejection_rate_5",
                "feasible": "feasible_rate",
                "DS": "mean_abs_DS",
                "DV": "mean_abs_DV",
                "RE": "mean_RE",
                "oracle_var": "mean_oracle_var",
                "instrument_var": "mean_instrument_var",
                "efficient_var": "mean_efficient_var",
            }
        )
        return summary

    def run(self) -> ExperimentResult:
        exp = self.config["experiment"]
        params = self.config["dgp"]["params"]
        n_rep = int(exp["n_replications"])
        beta0 = float(exp.get("beta0", 0.0))
        beta = float(params.get("beta", beta0))
        seed_seq = np.random.SeedSequence(int(exp["global_seed"]))
        dgp_cls = DGP_REGISTRY[self.config["dgp"]["name"]]
        dgp = dgp_cls(params)
        weight_grid = self._weight_grid()
        rows: list[dict[str, Any]] = []
        scenarios = [
            (int(T), dict(rho_design), int(K))
            for T in params["T_values"]
            for rho_design in params["rho_designs"]
            for K in self.config["sieve"]["K_values"]
        ]
        self.log(
            f"Starting run_id={self.run_id} with {len(scenarios)} scenarios, "
            f"{len(weight_grid)} weight settings, and {n_rep} replications each."
        )
        for T, rho_design, K in scenarios:
            if K >= T / 5:
                raise ValueError(f"Safety check failed: K={K} must be < T/5={T/5}.")
            child_seeds = seed_seq.spawn(n_rep)
            rep_seeds = [int(s.generate_state(1)[0]) for s in child_seeds]
            self.log(f"Scenario T={T}, rho={rho_design.get('label')}, K={K}")
            for weight_b in weight_grid:
                methods = self._methods(weight_b=None if np.isnan(weight_b) else weight_b)
                self.log(f"  Weight b={weight_b:g}")
                for rep, seed in enumerate(rep_seeds):
                    data = dgp.simulate(seed=seed, T=T, rho_design=rho_design, beta=beta)
                    for method in methods:
                        result = method.evaluate(data, beta0=beta0, K=K)
                        row = {
                            "run_id": self.run_id,
                            "rep": rep,
                            "seed": seed,
                            "T": T,
                            "rho_label": data.meta["rho_label"],
                            "rho_value": data.meta["rho_value"],
                            "beta": beta,
                            "beta0": beta0,
                            "kappa": data.meta["kappa"],
                            "xi": data.meta["xi"],
                            "m_type": data.meta["m_type"],
                            "x_initialization": data.meta["x_initialization"],
                            "w_process": data.meta["w_process"],
                            "w_stationary": data.meta["w_stationary"],
                            "w_alpha_mixing": data.meta["w_alpha_mixing"],
                            "K": K,
                            "weight_b": weight_b,
                            "method": result.method_name,
                            "el_stat": result.el_stat,
                            "reject_5": result.reject_5,
                            "feasible": result.feasible,
                            "lambda_hat": result.lambda_hat,
                        }
                        row.update(result.diagnostics)
                        rows.append(row)
        raw = pd.DataFrame(rows)
        summary = self._summarize(raw)
        diagnostics = {
            "n_rows": int(len(raw)),
            "n_scenarios": int(len(scenarios)),
            "n_weight_settings": int(len(weight_grid)),
            "n_replications": n_rep,
            "methods": list(self.config["methods"]),
        }
        manifest = {
            "run_id": self.run_id,
            "config_hash": self.digest,
            "n_replications": n_rep,
            "started_at": self.logs[0].split(" ")[0] if self.logs else None,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
        write_run_outputs(self.run_dir, self.config, self.digest, rows, summary, diagnostics, manifest, self.logs)
        self.log(f"Completed run. Outputs written to {self.run_dir}")
        return ExperimentResult(self.run_id, self.run_dir, raw, summary)