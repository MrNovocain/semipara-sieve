from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def canonical_config_bytes(config: dict[str, Any]) -> bytes:
    return yaml.safe_dump(config, sort_keys=True).encode("utf-8")


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def source_hash() -> str:
    package_dir = Path(__file__).resolve().parent
    root = package_dir.parent.parent if package_dir.parent.name == "src" else package_dir.parent
    candidates = set(package_dir.glob("*.py"))
    for pattern in ["pseel/*.py", "scripts/*.py", "scripts/**/*.py"]:
        candidates.update(root.glob(pattern))

    digest = hashlib.sha256()
    for path in sorted(p for p in candidates if p.is_file()):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_text() -> str:
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"pseel_source_hash={source_hash()}",
    ]
    try:
        import scipy
        lines.append(f"scipy={scipy.__version__}")
    except Exception:
        pass
    try:
        import yaml as _yaml
        lines.append(f"pyyaml={getattr(_yaml, '__version__', 'unknown')}")
    except Exception:
        pass
    return "\n".join(lines) + "\n"


def make_run_id(config: dict[str, Any], digest: str) -> str:
    fixed = config.get("outputs", {}).get("run_id")
    if fixed:
        return str(fixed)
    name = config.get("experiment", {}).get("name", "run")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{name}_{stamp}_{digest[:7]}"


def write_run_outputs(
    run_dir: Path,
    config: dict[str, Any],
    digest: str,
    rows: list[dict[str, Any]],
    summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    logs: list[str],
) -> None:
    source_digest = source_hash()
    manifest = dict(manifest)
    manifest.setdefault("source_hash", source_digest)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, run_dir / "config.yaml")
    (run_dir / "config_hash.txt").write_text(digest + "\n", encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    (run_dir / "source_hash.txt").write_text(source_digest + "\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text(environment_text(), encoding="utf-8")
    pd.DataFrame(rows).to_parquet(run_dir / "raw_replications.parquet", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "logs.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
