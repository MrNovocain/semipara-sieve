# Semiparametric Profile Sieve

This repository contains a working paper project targeting *Statistica Sinica*.
The project is organized so math source, paper-facing assets, simulations, data,
generated outputs, archived notebooks, and archived drafts have separate roles.

## Repository Layout

- `paper/math/`: current KKT math main body and minimal math main body.
- `paper/math/archive/`: supporting or older KKT workbook materials.
- `paper/figures/`: stable paper-facing figure copies.
- `paper/tables/`: stable paper-facing table copies.
- `paper/notes/`: proof audits, source notes, implementation notes, and submission checklists.
- `paper/archive/drafts/`: historical manuscript-like drafts and PDFs that are not current authority.
- `src/pseel/`: importable Python package for profile-sieve empirical likelihood implementation.
- `pseel/`: source-tree import shim so `python -m pseel.run ...` works before installation.
- `scripts/`: reproducible command-line workflows for simulations, figures, tables, diagnostics, and empirical examples.
- `configs/`: YAML configs; paper-facing MC configs live in `configs/mc/main/`, stress designs in `configs/mc/stress/`, and smoke/demo configs in `configs/mc/demo/`.
- `archive/code_20260628/notebooks/`: archived exploratory notebook snapshots, not part of the active reproducible pipeline.
- `data/raw/`: raw empirical inputs.
- `data/processed/`: cleaned inputs used by scripts and diagnostics.
- `data/interim/`: intermediate working data when needed.
- `results/`: ignored config-driven run-output directories with raw parquet, summaries, diagnostics, and lineage files.
- `archive/generated_20260628/`: legacy generated snapshots moved out of root-level `result/` and `tmp/`.
- `archive/code_20260628/`: legacy code and notebook snapshots moved out of active `scripts/` and `notebooks/` paths.
- `knowledge/`: local background PDFs, intentionally ignored by Git.
- `application_materials/`: local CV, transcript, and advisor-selection materials, intentionally ignored by Git.

## Manuscript Authority

For math or paper-text decisions, use these files first:

- Main math body: `paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex`
- Minimal math body: `paper/math/l2pt_l2p_kkt_mainbody_minimal.tex`

Historical drafts live under `paper/archive/drafts/` or `paper/math/archive/`.
Do not treat them as current authority unless explicitly requested.

## Reproducibility Policy

Keep source files, scripts, small data inputs, selected PDFs, stable figures,
and stable CSV/table summaries in Git when they support the paper's claims.
Do not track LaTeX build artifacts, Python caches, virtual environments,
local Jupyter runtime folders, private application materials, or local background PDFs.

Generated computational runs should write to `results/<run_id>/` or another
ignored directory under `results/`. Stable paper copies should be written to
`paper/figures/` and `paper/tables/`. Do not create new root-level `result/`
or `tmp/` project state.

## Main Computational Files

- `src/pseel/dgp.py`: predictive-regression DGPs, including broken-nuisance designs.
- `src/pseel/methods.py`: oracle, profile, intercept-only, frontier, and efficient-score EL methods.
- `src/pseel/breaks.py`: block-sieve partition construction, profile RSS, and break-aware EL statistics.
- `src/pseel/workbook_mc.py`: one-break workbook-faithful Monte Carlo utilities.
- `src/pseel/el.py`: scalar empirical-likelihood solver.
- `src/pseel/experiment.py`: config-driven Monte Carlo runner and summarizer.
- `src/pseel/io.py`: config hashing, source hashing, environment capture, and output writing.
- `src/pseel/run.py`: command-line entry point for YAML configs.
- `scripts/make_tables.py`: deterministic post-processing tables.
- `scripts/make_figures.py`: deterministic QQ and frontier figures.
- `scripts/theory_diagnostics.py`: theorem-risk diagnostics for projection/oracle/feasible contracts.
- `scripts/workbook_break_mc.py`: workbook-faithful one-break Monte Carlo runner.

## Running Scripts

Run the current config-driven pipeline from the repository root, for example:

```powershell
python -m pseel.run configs/mc/main/size_main.yaml
python scripts/make_tables.py --run-dir results/<size_run_id>
python scripts/make_figures.py --run-dir results/<size_run_id> --method profile_bounded --fig qq --T 250
```

For package-style imports, install the project in editable mode:

```powershell
python -m pip install -e .
```

## Local Background

For `L2(P)` geometry background, use the local ignored folder `knowledge/`, especially:

- `knowledge/CHPT.06 GEOMETRY OF DATA.pdf`

## Config-Driven Monte Carlo Pipeline

The current KKT workbook one-break simulation is run separately from the older
stable/no-break baseline pipeline:

```powershell
python scripts/workbook_break_mc.py configs/mc/main/broken_nuisance_one_break.yaml
```

The paper-facing simulation path is available through the `pseel` package. The main size design uses stationary alpha-mixing `W_t`, controlled-initial-condition persistent `X_t`, the smooth nuisance `m(w)=0.5 sin(w)+0.3(w^2-1)`, and the bounded score `tanh(X / 8)`:

```powershell
python -m pseel.run configs/mc/main/size_main.yaml
python scripts/make_tables.py --run-dir results/<size_run_id>
python scripts/make_figures.py --run-dir results/<size_run_id> --method profile_bounded --fig qq --T 250
```

The robustness-efficiency frontier is generated separately:

```powershell
python -m pseel.run configs/mc/main/frontier_main.yaml
python scripts/make_tables.py --run-dir results/<frontier_run_id>
python scripts/make_figures.py --run-dir results/<frontier_run_id> --fig frontier
```

A contemporaneous-endogeneity stress design is kept separate from the headline Wilks table:

```powershell
python -m pseel.run configs/mc/stress/endogeneity_stress.yaml
```

Each run writes a lineage-complete directory under `results/<run_id>/` containing `config.yaml`, `config_hash.txt`, `git_commit.txt`, `source_hash.txt`, `environment.txt`, `manifest.json`, `raw_replications.parquet`, `summary.csv`, `diagnostics.json`, and `logs.txt`. The source hash covers the package shim, `src/pseel`, and Python scripts so uncommitted code-state changes are auditable.

The currently verified reproducible evidence pack includes these run ids when present locally:

- `size_main_20260622_220035_894be7b`: 2000 replications, size/oracle-equivalence table, QQ diagnostic.
- `frontier_main_20260622_220246_9ff471a`: 2000 replications, robustness-efficiency frontier.
- `endogeneity_stress_20260622_220035_e634f37`: 1000 replications, contemporaneous-endogeneity stress diagnostic.

Stable paper copies remain under `paper/figures/` and `paper/tables/`.

## Automated Theory Diagnostics

Use the theorem-risk diagnostic runner to test the proof chain mechanically rather than by inspection:

```powershell
python scripts/theory_diagnostics.py --preset smoke --replications 80
python scripts/theory_diagnostics.py --preset core --replications 200
python scripts/theory_diagnostics.py --preset negative --replications 200
python scripts/theory_diagnostics.py --preset adversarial --replications 80 --max-scenarios 12
```

Each run writes `raw_replications.csv`, `method_summary.csv`, `contract_summary.csv`, `deterministic_checks.csv`, and `contract_report.json` under `results/theory_diagnostics_<preset>_<timestamp>/`. Read the contract table as a proof-chain audit: first `projection_pass`, then `oracle_pass`, then `feasible_oracle_pass`.

Run the checks with:

```powershell
pytest -q tests -p no:cacheprovider
```