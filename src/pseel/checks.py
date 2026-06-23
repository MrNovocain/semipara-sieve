from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import DGP_REGISTRY, METHOD_REGISTRY


def validate_config(config: dict[str, Any]) -> None:
    for section in ["experiment", "dgp", "sieve", "weights", "methods", "el", "outputs"]:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    exp = config["experiment"]
    if int(exp.get("n_replications", 0)) <= 0:
        raise ValueError("experiment.n_replications must be positive.")
    if "global_seed" not in exp:
        raise ValueError("experiment.global_seed is required.")
    dgp_name = config["dgp"].get("name")
    if dgp_name not in DGP_REGISTRY:
        raise ValueError(f"Unregistered DGP: {dgp_name}")
    params = config["dgp"].get("params", {})
    if not params.get("T_values"):
        raise ValueError("dgp.params.T_values is required.")
    if not params.get("rho_designs"):
        raise ValueError("dgp.params.rho_designs is required.")
    K_values = config["sieve"].get("K_values")
    if not K_values:
        raise ValueError("sieve.K_values is required.")
    for K in K_values:
        if int(K) <= 0:
            raise ValueError("All K values must be positive.")
    if not config["methods"]:
        raise ValueError("At least one method is required.")
    for method in config["methods"]:
        if method not in METHOD_REGISTRY:
            raise ValueError(f"Unregistered method: {method}")
    if "bounded_main" not in config["weights"]:
        raise ValueError("weights.bounded_main is required for M1 methods.")
    output_root = Path(config["outputs"].get("root", "results"))
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"outputs.root is not a directory: {output_root}")
