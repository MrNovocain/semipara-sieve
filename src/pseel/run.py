from __future__ import annotations

import argparse
from pathlib import Path

# Import modules for registry side effects.
from . import basis as _basis  # noqa: F401
from . import dgp as _dgp  # noqa: F401
from . import methods as _methods  # noqa: F401
from .experiment import ExperimentRunner
from .io import load_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a config-driven profile-sieve EL Monte Carlo experiment.")
    parser.add_argument("config", help="Path to YAML config, e.g. configs/mc/size_main.yaml")
    parser.add_argument("--output-root", default=None, help="Override outputs.root from the config.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing run directory.")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    runner = ExperimentRunner(config, output_root=args.output_root, overwrite=args.overwrite)
    result = runner.run()
    print(f"run_id={result.run_id}")
    print(f"run_dir={Path(result.run_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
