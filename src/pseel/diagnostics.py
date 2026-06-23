from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest

from .basis import PolynomialBasis
from .contracts import Dataset
from .dgp import PredictiveAR1DGP
from .el import empirical_likelihood_scalar
from .methods import InterceptOnlyBoundedEL, OracleBoundedEL, ProfileBoundedEL, ProfileEfficientEL
from .registry import build_weight
from .residualize import Residualizer
from .weights import LinearWeight, TanhWeight  # noqa: F401 - ensure registration


@dataclass(frozen=True)
class DiagnosticThresholds:
    alpha: float = 0.05
    min_feasible_rate: float = 0.98
    min_score_var: float = 1e-8
    orthogonality_tol: float = 1e-8
    projector_tol: float = 1e-8
    scaling_tol: float = 1e-8
    gram_condition_max: float = 1e10
    base_size_tolerance: float = 0.04
    base_rejection_gap_tolerance: float = 0.04
    base_ks_tolerance: float = 0.10
    mean_abs_ds_tolerance: float = 0.12
    mean_abs_dv_tolerance: float = 0.20
    p95_max_score_ratio_tolerance: float = 0.85
    negative_rejection_gap_min: float = 0.08


@dataclass(frozen=True)
class DiagnosticScenario:
    name: str
    T: int
    K: int
    rho_design: dict[str, Any]
    b: float = 8.0
    kappa: float = 0.0
    xi: float = 0.3
    a_w: float = 0.5
    burnin: int = 200
    beta: float = 0.0
    beta0: float = 0.0
    m_name: str = "sinus_quad"
    m_params: dict[str, float] | None = None
    weight_name: str = "tanh"
    weight_params: dict[str, float] | None = None
    x_initialization: str = "zero_at_sample_start"
    purpose: str = "core"

    def dgp_params(self) -> dict[str, Any]:
        return {
            "a_w": self.a_w,
            "kappa": self.kappa,
            "xi": self.xi,
            "burnin": self.burnin,
            "x_initialization": self.x_initialization,
            "m": {"name": self.m_name, "params": dict(self.m_params or {})},
        }

    def basis_config(self) -> dict[str, Any]:
        return {
            "basis": "polynomial",
            "include_intercept": True,
            "standardize_w": True,
            "K": self.K,
        }

    def weight_config(self) -> dict[str, Any]:
        if self.weight_params is not None:
            return {"name": self.weight_name, "params": dict(self.weight_params)}
        if self.weight_name == "tanh":
            return {"name": "tanh", "params": {"b": self.b}}
        return {"name": self.weight_name, "params": {}}

    def row(self) -> dict[str, Any]:
        out = asdict(self)
        out["rho_label"] = self.rho_design.get("label", "rho")
        out["rho_formula"] = self.rho_design.get("formula", "fixed")
        out["rho_value_input"] = self.rho_design.get("value", np.nan)
        out["rho_c_input"] = self.rho_design.get("c", np.nan)
        out.pop("rho_design")
        return out


@dataclass(frozen=True)
class DiagnosticResult:
    raw: pd.DataFrame
    method_summary: pd.DataFrame
    contract_summary: pd.DataFrame
    deterministic_checks: pd.DataFrame
    thresholds: DiagnosticThresholds


def default_scenarios(preset: str) -> list[DiagnosticScenario]:
    rho_grid = [
        {"label": "stationary_low", "formula": "fixed", "value": 0.5},
        {"label": "stationary_high", "formula": "fixed", "value": 0.95},
        {"label": "local_to_unity", "formula": "local", "c": 5},
        {"label": "unit_root", "formula": "fixed", "value": 1.0},
    ]
    if preset == "smoke":
        return [
            DiagnosticScenario("smoke_stationary", 80, 4, rho_grid[0], purpose="core"),
            DiagnosticScenario("smoke_unit_root", 80, 4, rho_grid[3], purpose="core"),
            DiagnosticScenario(
                "smoke_negative_intercept",
                120,
                4,
                rho_grid[1],
                xi=0.8,
                m_name="sinus_quad",
                m_params={"a1": 0.0, "a2": 2.0},
                purpose="negative_intercept",
            ),
        ]
    if preset == "core":
        return [DiagnosticScenario(f"core_{rho['label']}", 250, 6, rho, purpose="core") for rho in rho_grid]
    if preset == "negative":
        return [
            DiagnosticScenario(
                "negative_intercept_omits_nuisance",
                180,
                4,
                rho_grid[1],
                xi=0.8,
                m_name="sinus_quad",
                m_params={"a1": 0.0, "a2": 2.0},
                purpose="negative_intercept",
            ),
            DiagnosticScenario(
                "negative_linear_unit_root_weight",
                180,
                6,
                rho_grid[3],
                weight_name="linear",
                purpose="negative_oracle",
            ),
            DiagnosticScenario(
                "negative_large_k_projection",
                120,
                12,
                rho_grid[2],
                purpose="negative_large_k",
            ),
        ]
    if preset == "adversarial":
        scenarios: list[DiagnosticScenario] = []
        for rho, kappa, xi, b, K in product(rho_grid, [0.0, 0.3], [0.0, 0.5], [1.0, 4.0, 8.0], [4, 8]):
            name = f"adv_{rho['label']}_k{kappa:g}_xi{xi:g}_b{b:g}_K{K:g}".replace(".", "p")
            scenarios.append(DiagnosticScenario(name, 180, K, rho, b=b, kappa=kappa, xi=xi, purpose="adversarial"))
        return scenarios
    raise ValueError(f"Unknown diagnostic preset: {preset}")


def _norm_t(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v, dtype=float) ** 2)))


def _basis_and_residualizer(data: Dataset, scenario: DiagnosticScenario) -> tuple[np.ndarray, Residualizer]:
    P = PolynomialBasis(K=scenario.K, include_intercept=True, standardize_w=True).make(data.w_lag)
    return P, Residualizer(P)


def _raw_weight(data: Dataset, scenario: DiagnosticScenario) -> np.ndarray:
    return build_weight(scenario.weight_config()).transform(data.x_lag)


def _score_vector(method_name: str, data: Dataset, scenario: DiagnosticScenario) -> np.ndarray:
    raw = _raw_weight(data, scenario)
    if method_name == "intercept_only_bounded":
        P = np.ones((data.T, 1))
        R = Residualizer(P)
        instrument = R.residualize(raw)
        uhat = R.residualize(data.y - scenario.beta0 * data.x_lag)
        return uhat * instrument

    P, R = _basis_and_residualizer(data, scenario)
    if method_name == "oracle_bounded":
        return data.u * R.residualize(raw)
    if method_name == "profile_bounded":
        uhat = R.residualize(data.y - scenario.beta0 * data.x_lag)
        return uhat * R.residualize(raw)
    if method_name == "profile_efficient":
        uhat = R.residualize(data.y - scenario.beta0 * data.x_lag)
        return uhat * R.residualize(data.x_lag)
    raise ValueError(f"Unknown diagnostic method: {method_name}")


def _method_objects(scenario: DiagnosticScenario):
    basis_config = scenario.basis_config()
    weight_config = scenario.weight_config()
    el_config = {"solver": {"tolerance": 1e-10, "max_iter": 100}}
    return [
        OracleBoundedEL("oracle_bounded", basis_config, weight_config, el_config),
        ProfileBoundedEL("profile_bounded", basis_config, weight_config, el_config),
        InterceptOnlyBoundedEL("intercept_only_bounded", basis_config, weight_config, el_config),
        ProfileEfficientEL("profile_efficient", basis_config, weight_config, el_config),
    ]


def _deterministic_checks(data: Dataset, scenario: DiagnosticScenario, thresholds: DiagnosticThresholds) -> dict[str, Any]:
    P, R = _basis_and_residualizer(data, scenario)
    raw = _raw_weight(data, scenario)
    y0 = data.y - scenario.beta0 * data.x_lag
    uhat = R.residualize(y0)
    instrument = R.residualize(raw)
    gram = P.T @ P
    M = np.eye(data.T) - P @ np.linalg.solve(gram, P.T)
    oracle_z = data.u * instrument
    el = empirical_likelihood_scalar(oracle_z)
    scaled_el = empirical_likelihood_scalar(7.5 * oracle_z)

    k1_scenario = DiagnosticScenario(
        name=f"{scenario.name}_k1_identity",
        T=scenario.T,
        K=1,
        rho_design=scenario.rho_design,
        b=scenario.b,
        kappa=scenario.kappa,
        xi=scenario.xi,
        a_w=scenario.a_w,
        burnin=scenario.burnin,
        beta=scenario.beta,
        beta0=scenario.beta0,
        m_name=scenario.m_name,
        m_params=scenario.m_params,
        weight_name=scenario.weight_name,
        weight_params=scenario.weight_params,
        x_initialization=scenario.x_initialization,
        purpose=scenario.purpose,
    )
    profile_k1 = ProfileBoundedEL(
        "profile_bounded", k1_scenario.basis_config(), k1_scenario.weight_config(), {"solver": {}}
    ).evaluate(data, scenario.beta0, K=1)
    intercept_k1 = InterceptOnlyBoundedEL(
        "intercept_only_bounded", k1_scenario.basis_config(), k1_scenario.weight_config(), {"solver": {}}
    ).evaluate(data, scenario.beta0, K=1)
    k1_diff = abs(profile_k1.el_stat - intercept_k1.el_stat)

    contraction_margins = []
    for z in [y0, raw, data.x_lag, data.u]:
        denom = max(_norm_t(z), 1e-12)
        contraction_margins.append(_norm_t(R.residualize(z)) / denom - 1.0)

    checks = {
        **scenario.row(),
        "orth_profile_inf": float(np.max(np.abs(P.T @ uhat))),
        "orth_weight_inf": float(np.max(np.abs(P.T @ instrument))),
        "projector_symmetry_inf": float(np.max(np.abs(M - M.T))),
        "projector_idempotence_inf": float(np.max(np.abs(M @ M - M))),
        "projector_contraction_margin_max": float(max(contraction_margins)),
        "scaling_invariance_abs": float(abs(el.statistic - scaled_el.statistic)),
        "k1_profile_intercept_abs": float(k1_diff),
        "gram_condition": float(np.linalg.cond(gram / data.T)),
        "basis_zeta": float(np.max(np.linalg.norm(P, axis=1))),
    }
    checks["projection_contract_pass"] = bool(
        checks["orth_profile_inf"] <= thresholds.orthogonality_tol
        and checks["orth_weight_inf"] <= thresholds.orthogonality_tol
        and checks["projector_symmetry_inf"] <= thresholds.projector_tol
        and checks["projector_idempotence_inf"] <= thresholds.projector_tol
        and checks["projector_contraction_margin_max"] <= thresholds.projector_tol
        and checks["scaling_invariance_abs"] <= thresholds.scaling_tol
        and checks["k1_profile_intercept_abs"] <= thresholds.scaling_tol
        and np.isfinite(checks["gram_condition"])
        and checks["gram_condition"] <= thresholds.gram_condition_max
    )
    return checks


def _ks_chisq1(values: pd.Series) -> float:
    finite = values[np.isfinite(values)].to_numpy(dtype=float)
    if finite.size < 2:
        return float("nan")
    return float(kstest(finite, chi2(df=1).cdf).statistic)


def _size_tolerance(n: int, thresholds: DiagnosticThresholds) -> float:
    se = np.sqrt(thresholds.alpha * (1.0 - thresholds.alpha) / max(n, 1))
    return float(max(thresholds.base_size_tolerance, 2.5 * se))


def _gap_tolerance(n: int, thresholds: DiagnosticThresholds) -> float:
    se = np.sqrt(2.0 * thresholds.alpha * (1.0 - thresholds.alpha) / max(n, 1))
    return float(max(thresholds.base_rejection_gap_tolerance, 2.5 * se))


def _ks_tolerance(n: int, thresholds: DiagnosticThresholds) -> float:
    return float(max(thresholds.base_ks_tolerance, 1.63 / np.sqrt(max(n, 1))))


def _summarize_methods(raw: pd.DataFrame, thresholds: DiagnosticThresholds) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["scenario", "purpose", "method", "T", "K", "rho_label", "weight_name", "b", "kappa", "xi"]
    for keys, group in raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        n = int(len(group))
        row.update(
            {
                "n_replications": n,
                "feasible_rate": float(group["feasible"].mean()),
                "rejection_rate_5": float(group["reject_5"].mean()),
                "mean_el_stat": float(np.nanmean(group["el_stat"])),
                "ks_chisq1": _ks_chisq1(group["el_stat"]),
                "mean_score_var": float(np.nanmean(group["score_var"])),
                "min_score_var": float(np.nanmin(group["score_var"])),
                "mean_max_score_ratio": float(np.nanmean(group["max_score_ratio"])),
                "p95_max_score_ratio": float(np.nanpercentile(group["max_score_ratio"], 95)),
                "mean_abs_DS": float(np.nanmean(np.abs(group["DS"]))) if group["DS"].notna().any() else np.nan,
                "mean_abs_DV": float(np.nanmean(np.abs(group["DV"]))) if group["DV"].notna().any() else np.nan,
                "mean_RE": float(np.nanmean(group["RE"])) if group["RE"].notna().any() else np.nan,
                "size_tolerance": _size_tolerance(n, thresholds),
                "ks_tolerance": _ks_tolerance(n, thresholds),
            }
        )
        row["oracle_distribution_pass"] = bool(
            row["method"] == "oracle_bounded"
            and row["feasible_rate"] >= thresholds.min_feasible_rate
            and abs(row["rejection_rate_5"] - thresholds.alpha) <= row["size_tolerance"]
            and row["ks_chisq1"] <= row["ks_tolerance"]
            and row["min_score_var"] >= thresholds.min_score_var
            and row["p95_max_score_ratio"] <= thresholds.p95_max_score_ratio_tolerance
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_contracts(
    method_summary: pd.DataFrame, deterministic_checks: pd.DataFrame, thresholds: DiagnosticThresholds
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    det_by_scenario = deterministic_checks.set_index("name").to_dict(orient="index")
    for scenario, group in method_summary.groupby("scenario", dropna=False):
        methods = {str(row.method): row for row in group.itertuples(index=False)}
        oracle = methods.get("oracle_bounded")
        profile = methods.get("profile_bounded")
        intercept = methods.get("intercept_only_bounded")
        det = det_by_scenario.get(str(scenario), {})
        n = int(oracle.n_replications) if oracle is not None else 0
        rejection_gap = np.nan
        if oracle is not None and profile is not None:
            rejection_gap = float(profile.rejection_rate_5 - oracle.rejection_rate_5)
        feasible_oracle_pass = bool(
            profile is not None
            and oracle is not None
            and abs(rejection_gap) <= _gap_tolerance(n, thresholds)
            and profile.mean_abs_DS <= thresholds.mean_abs_ds_tolerance
            and profile.mean_abs_DV <= thresholds.mean_abs_dv_tolerance
            and profile.feasible_rate >= thresholds.min_feasible_rate
        )
        projection_pass = bool(det.get("projection_contract_pass", False))
        oracle_pass = bool(oracle.oracle_distribution_pass) if oracle is not None else False
        purpose = str(group["purpose"].iloc[0])
        negative_detected = False
        if purpose == "negative_intercept" and oracle is not None and intercept is not None:
            negative_detected = bool(abs(intercept.rejection_rate_5 - oracle.rejection_rate_5) >= thresholds.negative_rejection_gap_min)
        elif purpose == "negative_oracle":
            negative_detected = not oracle_pass
        elif purpose == "negative_large_k":
            negative_detected = not projection_pass or not feasible_oracle_pass
        contract_pass = bool(projection_pass and oracle_pass and feasible_oracle_pass)
        if purpose.startswith("negative"):
            contract_pass = bool(negative_detected)
        rows.append(
            {
                "scenario": scenario,
                "purpose": purpose,
                "T": int(group["T"].iloc[0]),
                "K": int(group["K"].iloc[0]),
                "rho_label": group["rho_label"].iloc[0],
                "weight_name": group["weight_name"].iloc[0],
                "b": float(group["b"].iloc[0]),
                "kappa": float(group["kappa"].iloc[0]),
                "xi": float(group["xi"].iloc[0]),
                "projection_pass": projection_pass,
                "oracle_pass": oracle_pass,
                "feasible_oracle_pass": feasible_oracle_pass,
                "negative_control_detected": negative_detected,
                "contract_pass": contract_pass,
                "oracle_rejection_rate_5": float(oracle.rejection_rate_5) if oracle is not None else np.nan,
                "profile_rejection_rate_5": float(profile.rejection_rate_5) if profile is not None else np.nan,
                "intercept_rejection_rate_5": float(intercept.rejection_rate_5) if intercept is not None else np.nan,
                "profile_oracle_rejection_gap": rejection_gap,
                "profile_mean_abs_DS": float(profile.mean_abs_DS) if profile is not None else np.nan,
                "profile_mean_abs_DV": float(profile.mean_abs_DV) if profile is not None else np.nan,
                "oracle_ks_chisq1": float(oracle.ks_chisq1) if oracle is not None else np.nan,
                "oracle_p95_max_score_ratio": float(oracle.p95_max_score_ratio) if oracle is not None else np.nan,
                "size_tolerance": _size_tolerance(n, thresholds),
                "gap_tolerance": _gap_tolerance(n, thresholds),
                "ks_tolerance": _ks_tolerance(n, thresholds),
            }
        )
    return pd.DataFrame(rows)


def run_diagnostics(
    scenarios: list[DiagnosticScenario],
    n_replications: int,
    seed: int = 20260623,
    thresholds: DiagnosticThresholds | None = None,
) -> DiagnosticResult:
    if n_replications <= 0:
        raise ValueError("n_replications must be positive.")
    thresholds = thresholds or DiagnosticThresholds()
    rng = np.random.default_rng(seed)
    raw_rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        if scenario.K >= scenario.T:
            raise ValueError(f"Scenario {scenario.name} has K >= T.")
        dgp = PredictiveAR1DGP(scenario.dgp_params())
        rep_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_replications, dtype=np.uint32)
        first_data = dgp.simulate(int(rep_seeds[0]), scenario.T, scenario.rho_design, scenario.beta)
        deterministic_rows.append(_deterministic_checks(first_data, scenario, thresholds))
        methods = _method_objects(scenario)
        for rep, rep_seed in enumerate(rep_seeds):
            data = dgp.simulate(int(rep_seed), scenario.T, scenario.rho_design, scenario.beta)
            scenario_row = scenario.row()
            for method in methods:
                result = method.evaluate(data, beta0=scenario.beta0, K=scenario.K)
                z = _score_vector(result.method_name, data, scenario)
                row = {
                    "scenario": scenario.name,
                    "rep": rep,
                    "seed": int(rep_seed),
                    "method": result.method_name,
                    "rho_value": float(data.meta["rho_value"]),
                    "el_stat": result.el_stat,
                    "reject_5": result.reject_5,
                    "feasible": result.feasible,
                    "lambda_hat": result.lambda_hat,
                    "score_var": float(np.mean(z**2)),
                    "max_score_ratio": float(np.max(np.abs(z)) / np.sqrt(scenario.T)),
                }
                row.update(scenario_row)
                row.update(result.diagnostics)
                raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)
    deterministic = pd.DataFrame(deterministic_rows)
    method_summary = _summarize_methods(raw, thresholds)
    contract_summary = _summarize_contracts(method_summary, deterministic, thresholds)
    return DiagnosticResult(raw, method_summary, contract_summary, deterministic, thresholds)


def write_diagnostic_outputs(result: DiagnosticResult, output_dir: str | Path, overwrite: bool = False) -> Path:
    out = Path(output_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output directory exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    result.raw.to_csv(out / "raw_replications.csv", index=False)
    result.method_summary.to_csv(out / "method_summary.csv", index=False)
    result.contract_summary.to_csv(out / "contract_summary.csv", index=False)
    result.deterministic_checks.to_csv(out / "deterministic_checks.csv", index=False)
    report = {
        "thresholds": asdict(result.thresholds),
        "n_raw_rows": int(len(result.raw)),
        "n_scenarios": int(len(result.contract_summary)),
        "n_contract_failures": int((~result.contract_summary["contract_pass"]).sum()),
        "contract_failures": result.contract_summary.loc[
            ~result.contract_summary["contract_pass"], "scenario"
        ].astype(str).tolist(),
    }
    (out / "contract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
