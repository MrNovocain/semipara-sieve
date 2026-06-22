# Semiparametric Profile Sieve

This repository contains a working paper project targeting *Statistica Sinica*.
The project is organized so the paper, simulations, data inputs, generated outputs, notebooks, and local background material stay separated by purpose.

## Repository Layout

- `src/profile_sieve/`: importable Python package for the profile-sieve empirical likelihood implementation.
- `scripts/simulation/`: command-line simulation and diagnostic drivers.
- `scripts/figures/`: figure-bundling and plotting workflows.
- `notebooks/`: exploratory and executed notebooks.
- `writing_samples/`: paper drafts, LaTeX source, and selected compiled PDFs.
- `data/raw/`: raw empirical inputs.
- `data/processed/`: cleaned inputs used by scripts and diagnostics.
- `data/interim/`: intermediate working data when needed.
- `result/`: generated figures, CSV summaries, and final visual sets.
- `result/legacy/`: older root-level plots preserved for reference.
- `agent.md`: local project operating rules for future agent work.
- `knowledge/`: local background PDFs, intentionally ignored by Git.
- `application_materials/`: local CV, transcript, and advisor-selection materials, intentionally ignored by Git.

## Reproducibility Policy

Keep source files, scripts, small data inputs, selected PDFs, figures, and CSV summaries in Git when they support the paper's claims.
Do not track LaTeX build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_latexmk`, or `.synctex.gz` files.
Do not track Python caches, virtual environments, local Jupyter runtime folders, private application materials, or local background PDFs.
Do not put generated plots in the repository root; write them to `result/` or a timestamped subfolder.

## Main Computational Files

- `src/profile_sieve/mc_sieve_el.py`: core simulation and profile-sieve empirical likelihood routines.
- `scripts/simulation/batched_qq_convergence.py`: batched QQ convergence experiments.
- `scripts/simulation/diagnostic_chisq_test.py`: chi-square diagnostic workflow.
- `scripts/simulation/diagnostic_chisq_test_sequence.py`: sequence diagnostic workflow.
- `scripts/simulation/k_sensitivity_chisq.py`: sieve-dimension sensitivity diagnostics.
- `scripts/figures/make_final_visual_set.py`: final figure/CSV bundle generation.
- `scripts/figures/plot_qq_convergence.py`: QQ convergence plotting workflow.

## Running Scripts

Run scripts from the repository root, for example:

```powershell
python scripts/simulation/batched_qq_convergence.py --jobs 1
python scripts/figures/make_final_visual_set.py
```

For package-style imports, install the project in editable mode:

```powershell
python -m pip install -e .
```

## Local Background

For `L2(P)` geometry background, use the local ignored folder `knowledge/`, especially:

- `knowledge/CHPT.06 GEOMETRY OF DATA.pdf`