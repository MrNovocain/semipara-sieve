# Agent Operating Rules

## Project Identity
- This is a working paper project targeting *Statistica Sinica*.
- Treat the main project as the profile sieve paper, not as an application-materials or advisor-search workspace.
- Maintain a fully replicable, modular research structure: source package, scripts, notebooks, paper source, data, generated results, and background knowledge should remain separated by purpose.

## Hard Constraints
- Do not modify `writing_samples/profile sieve.tex`.
- Keep Monte Carlo outputs in `results/<run_id>/` through the config-driven runner.
- Prefer reproducible scripts over one-off notebook or shell-only experiments.
- Keep personal application materials outside the paper workflow.
- Keep local background PDFs in [knowledge](knowledge/), which is intentionally gitignored.

## Mathematical Background
- For the `L2(P)` geometry background, use [knowledge](knowledge/) as the local reference folder.
- The primary reference is [CHPT.06 GEOMETRY OF DATA.pdf](knowledge/CHPT.06%20GEOMETRY%20OF%20DATA.pdf).

## Replicable Modular Structure
- Put raw inputs under `data/raw/` and cleaned analysis inputs under `data/processed/`.
- Put generated tables, figures, and simulation summaries under `results/<run_id>/`; do not put generated plots in the repository root.
- Keep reusable implementation logic in `src/pseel/` and command-line workflows in `python -m pseel.run`, `scripts/make_tables.py`, and `scripts/make_figures.py` rather than notebook-only cells.
- Keep paper-facing writing in `writing_samples/`; do not mix CV, transcript, advisor-selection, or admissions material into paper directories.
- Keep each computational claim auditable from source code plus saved CSV or figure output.

## Long-Run Reminder
- During long-running or iterative work, re-open and check this `agent.md` about every 10 minutes.
- Remind yourself you should keep iterating.
- Each time this file is checked, explicitly write down:
  - What rules from `agent.md` are useful for the current task.
  - What rules from `agent.md` are not useful for the current task.
  - What direction leads to a good result.
  - What direction does not lead to a good result.
  - Any adjustment needed in the current workflow.

## Monte Carlo Visual Workflow
- Iterate until the generated graphs are visually usable and statistically interpretable.
- Preserve theory-facing diagnostics separately from stress-test diagnostics.
- Label stress-test DGPs clearly when they intentionally strain or violate assumptions.
- Save CSV summaries next to figures so visual claims can be audited.