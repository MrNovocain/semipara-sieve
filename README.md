# Semiparametric Profile Sieve

This repository contains a working paper project targeting *Statistica Sinica*.
The project is organized so the paper, simulations, data inputs, generated outputs, and local background material stay separate.

## Repository Layout

- `writing_samples/`: paper drafts, LaTeX source, and selected compiled PDFs.
- `data/raw/`: raw empirical inputs.
- `data/processed/`: cleaned inputs used by scripts and diagnostics.
- `data/interim/`: intermediate working data when needed.
- `result/`: generated figures, CSV summaries, and final visual sets.
- root Python scripts: reproducible simulation and diagnostic workflows.
- `agent.md`: local project operating rules for future agent work.
- `knowledge/`: local background PDFs, intentionally ignored by Git.
- `application_materials/`: local CV, transcript, and advisor-selection materials, intentionally ignored by Git.

## Reproducibility Policy

Keep source files, scripts, small data inputs, selected PDFs, figures, and CSV summaries in Git when they support the paper's claims.
Do not track LaTeX build artifacts such as `.aux`, `.log`, `.fls`, `.fdb_latexmk`, or `.synctex.gz` files.
Do not track Python caches, virtual environments, local Jupyter runtime folders, private application materials, or local background PDFs.

## Main Computational Files

- `mc_sieve_el.py`: core simulation and profile-sieve empirical likelihood routines.
- `batched_qq_convergence.py`: batched QQ convergence experiments.
- `diagnostic_chisq_test.py`: chi-square diagnostic workflow.
- `diagnostic_chisq_test_sequence.py`: sequence diagnostic workflow.
- `k_sensitivity_chisq.py`: sieve-dimension sensitivity diagnostics.
- `make_final_visual_set.py`: final figure/CSV bundle generation.
- `plot_qq_convergence.py`: QQ convergence plotting workflow.

## Local Background

For `L2(P)` geometry background, use the local ignored folder `knowledge/`, especially:

- `knowledge/CHPT.06 GEOMETRY OF DATA.pdf`