import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from pseel.experiment import ExperimentRunner
from pseel.io import load_yaml


TEST_ROOT = Path("results") / "_pytest_repro"


def _mini_config(run_id: str) -> dict:
    config = load_yaml("configs/mc/size_main.yaml")
    config["experiment"]["n_replications"] = 3
    config["experiment"]["global_seed"] = 777
    config["dgp"]["params"]["T_values"] = [40]
    config["dgp"]["params"]["rho_designs"] = [{"label": "stationary", "formula": "fixed", "value": 0.8}]
    config["dgp"]["params"]["burnin"] = 20
    config["sieve"]["K_values"] = [3]
    config["outputs"]["root"] = str(TEST_ROOT)
    config["outputs"]["run_id"] = run_id
    config["outputs"]["overwrite"] = True
    return config


def test_same_config_and_seed_gives_identical_raw_results():
    suffix = uuid4().hex
    result1 = ExperimentRunner(_mini_config(f"run_a_{suffix}"), overwrite=True).run()
    raw1 = result1.raw.drop(columns=["run_id"])

    result2 = ExperimentRunner(_mini_config(f"run_b_{suffix}"), overwrite=True).run()
    raw2 = result2.raw.drop(columns=["run_id"])

    pd.testing.assert_frame_equal(raw1, raw2)
    assert (Path(result1.run_dir) / "raw_replications.parquet").exists()
    assert (Path(result1.run_dir) / "summary.csv").exists()
    assert (Path(result1.run_dir) / "source_hash.txt").exists()
    manifest = json.loads((Path(result1.run_dir) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_hash"] == (Path(result1.run_dir) / "source_hash.txt").read_text(encoding="utf-8").strip()


def test_frontier_reuses_replication_seeds_across_b_grid():
    config = _mini_config(f"frontier_{uuid4().hex}")
    config["experiment"]["task"] = "frontier"
    config["frontier"] = {"b_values": [0.5, 1.0]}
    config["methods"] = ["profile_bounded_frontier"]
    result = ExperimentRunner(config, overwrite=True).run()
    raw = result.raw.sort_values(["rep", "weight_b"])
    assert set(raw["weight_b"]) == {0.5, 1.0}
    seeds_by_rep = raw.groupby("rep")["seed"].nunique()
    assert int(seeds_by_rep.max()) == 1
    assert raw["RE"].between(0.0, 1.0 + 1e-12).all()