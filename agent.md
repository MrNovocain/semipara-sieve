# Agent Operating Rules

## Project Identity

- This is a working paper project targeting *Statistica Sinica*.
- Treat the main project as the profile-sieve empirical likelihood paper, not as an application-materials or advisor-search workspace.
- Maintain a replicable, modular research structure: source package, scripts, paper source, data, generated results, archived notebooks, and local background material should remain separated by purpose.

## Hard Constraints

- For manuscript-theory or paper-text decisions, treat `paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex` as the current math main body.
- Treat `paper/math/l2pt_l2p_kkt_mainbody_minimal.tex` as the compact/minimal math body.
- Do not use historical manuscript-like files under `paper/archive/drafts/` or `paper/math/archive/` as current authority unless the user explicitly asks to inspect one of them.
- Do not modify archived historical drafts unless the user explicitly requests work on that file.
- Keep Monte Carlo and diagnostic run outputs in ignored `results/<run_id>/` directories through the config-driven runner.
- Do not create new root-level `result/` or `tmp/` project state.
- Prefer reproducible scripts over one-off notebook or shell-only experiments.
- Keep personal application materials outside the paper workflow.
- Keep local background PDFs in `knowledge/`, which is intentionally gitignored.

## Mathematical Background

- For the `L2(P)` geometry background, use `knowledge/` as the local reference folder.
- The primary reference is `knowledge/CHPT.06 GEOMETRY OF DATA.pdf`.

## Replicable Modular Structure

- Put current paper math under `paper/math/`.
- Put stable paper-facing figures under `paper/figures/` and stable tables under `paper/tables/`.
- Put proof audits, source notes, and planning notes under `paper/notes/`.
- Put old drafts under `paper/archive/drafts/` or `paper/math/archive/`.
- Put legacy code and notebook snapshots under `archive/code_YYYYMMDD/`, not active `scripts/` or `notebooks/` paths.
- Put raw inputs under `data/raw/` and cleaned analysis inputs under `data/processed/`.
- Put generated tables, figures, simulation summaries, and diagnostics under `results/<run_id>/`; copy only selected stable outputs into `paper/figures/` or `paper/tables/`.
- Keep reusable implementation logic in `src/pseel/` and active command-line workflows in `python -m pseel.run`, `scripts/workbook_break_mc.py`, `scripts/empirical_goyal_welch.py`, `scripts/make_tables.py`, `scripts/make_figures.py`, and `scripts/theory_diagnostics.py`.
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
  - Double check everything again from the beginning whenever you have modified anything.

## Monte Carlo Visual Workflow

- Iterate until generated graphs are visually usable and statistically interpretable.
- Preserve theory-facing diagnostics separately from stress-test diagnostics.
- Label stress-test DGPs clearly when they intentionally strain or violate assumptions.
- Save CSV summaries next to figures so visual claims can be audited.