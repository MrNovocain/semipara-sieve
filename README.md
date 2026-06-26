# Semiparametric Profile Sieve

This repository contains a working paper project targeting *Statistica Sinica*.
The project is organized so the paper, simulations, data inputs, generated outputs, notebooks, and local background material stay separated by purpose.

## Repository Layout

- `src/pseel/`: importable Python package for the config-driven profile-sieve empirical likelihood implementation.
- `pseel/`: source-tree import shim so `python -m pseel.run ...` works before installation.
- `scripts/make_tables.py`: deterministic table generation from saved run directories.
- `scripts/make_figures.py`: deterministic figure generation from saved run directories.
- `notebooks/`: exploratory and executed notebooks.
- `writing_samples/`: paper drafts, LaTeX source, and selected compiled PDFs.
- `data/raw/`: raw empirical inputs.
- `data/processed/`: cleaned inputs used by scripts and diagnostics.
- `data/interim/`: intermediate working data when needed.
- `results/`: config-driven Monte Carlo run directories with raw parquet, summaries, diagnostics, and lineage files.
- `agent.md`: local project operating rules for future agent work.
- `knowledge/`: local background PDFs, intentionally ignored by Git.
- `application_materials/`: local CV, transcript, and advisor-selection materials, intentionally ignored by Git.

## Reproducibility Policy

Keep source files, scripts, small data inputs, selected PDFs, figures, and CSV summaries in Git when they support the paper's claims.
Do not track LaTeX build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_latexmk`, or `.synctex.gz` files.
Do not track Python caches, virtual environments, local Jupyter runtime folders, private application materials, or local background PDFs.
Do not put generated plots in the repository root; write reproducible outputs to `results/<run_id>/` or stable paper copies under `writing_samples/figures/` and `writing_samples/tables/`.

## Main Computational Files

- `src/pseel/dgp.py`: predictive-regression DGP used by the paper-facing Monte Carlo pipeline.
- `src/pseel/methods.py`: oracle, profile, intercept-only, frontier, and efficient-score EL methods.
- `src/pseel/el.py`: scalar empirical-likelihood solver.
- `src/pseel/experiment.py`: config-driven Monte Carlo runner and summarizer.
- `src/pseel/io.py`: config hashing, source hashing, environment capture, and output writing.
- `src/pseel/run.py`: command-line entry point for YAML configs.
- `scripts/make_tables.py`: deterministic post-processing tables.
- `scripts/make_figures.py`: deterministic QQ and frontier figures.

## Running Scripts

Run the current config-driven pipeline from the repository root, for example:

```powershell
python -m pseel.run configs/mc/size_main.yaml
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

The paper-facing simulation path is available through the `pseel` package. The main size design uses stationary alpha-mixing `W_t`, controlled-initial-condition persistent `X_t`, the smooth nuisance `m(w)=0.5 sin(w)+0.3(w^2-1)`, and the bounded score `tanh(X / 8)`:

```powershell
python -m pseel.run configs/mc/size_main.yaml
python scripts/make_tables.py --run-dir results/<size_run_id>
python scripts/make_figures.py --run-dir results/<size_run_id> --method profile_bounded --fig qq --T 250
```

The robustness-efficiency frontier is generated separately:

```powershell
python -m pseel.run configs/mc/frontier_main.yaml
python scripts/make_tables.py --run-dir results/<frontier_run_id>
python scripts/make_figures.py --run-dir results/<frontier_run_id> --fig frontier
```

A contemporaneous-endogeneity stress design is kept separate from the headline Wilks table:

```powershell
python -m pseel.run configs/mc/endogeneity_stress.yaml
```

Each run writes a lineage-complete directory under `results/<run_id>/` containing `config.yaml`, `config_hash.txt`, `git_commit.txt`, `source_hash.txt`, `environment.txt`, `manifest.json`, `raw_replications.parquet`, `summary.csv`, `diagnostics.json`, and `logs.txt`. The source hash covers the package shim, `src/pseel`, and Python scripts so uncommitted code-state changes are auditable. Deterministic post-processing additionally writes `summary_size.csv`, `summary_oracle_equiv.csv`, and `summary_frontier.csv` when applicable.

The currently verified reproducible evidence pack includes these runs:

- `size_main_20260622_220035_894be7b`: 2000 replications, size/oracle-equivalence table, QQ diagnostic.
- `frontier_main_20260622_220246_9ff471a`: 2000 replications, robustness-efficiency frontier.
- `endogeneity_stress_20260622_220035_e634f37`: 1000 replications, contemporaneous-endogeneity stress diagnostic.

The current manuscript source of truth is `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`. Stable figure copies remain under `writing_samples/figures/`, compact CSV/table copies remain under `writing_samples/tables/`, and noncanonical manuscript-like TeX files are listed in `writing_samples/source_notes/noncanonical_manuscript_sources.md`.


## Automated Theory Diagnostics

Use the theorem-risk diagnostic runner to test the proof chain mechanically rather than by inspection:

```powershell
python scripts/theory_diagnostics.py --preset smoke --replications 80
python scripts/theory_diagnostics.py --preset core --replications 200
python scripts/theory_diagnostics.py --preset negative --replications 200
python scripts/theory_diagnostics.py --preset adversarial --replications 80 --max-scenarios 12
```

Each run writes `raw_replications.csv`, `method_summary.csv`, `contract_summary.csv`, `deterministic_checks.csv`, and `contract_report.json` under `results/theory_diagnostics_<preset>_<timestamp>/`. Read the contract table as a proof-chain audit: first `projection_pass`, then `oracle_pass`, then `feasible_oracle_pass`. If `oracle_pass` fails, the high-level oracle Wilks input failed; if `oracle_pass` holds but `feasible_oracle_pass` fails, the sieve/profile replacement is the problem. Negative-control presets are expected to be detected rather than pass the core contract.

Run the checks with:

```powershell
pytest -q tests -p no:cacheprovider
```
