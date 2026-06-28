# Project Organization Design

## Goal

Reorganize the repository so the current KKT math main body is the clear
paper authority, generated artifacts no longer live at the repository root,
and future work has predictable locations for code, data, paper assets,
results, notes, and archives.

## Current Problems

- The documented manuscript authority still points to
  `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`, but the
  current math main body is the KKT workbook mainbody/minimal pair.
- `writing_samples/` mixes current paper assets, old drafts, workbook files,
  notes, figures, tables, PDFs, and local LaTeX build artifacts.
- `result/` is tracked and contains legacy generated plots and CSVs, while
  `results/` is the intended ignored run-output directory.
- `tmp/` contains tracked render checks and ignored local scratch material.
- Several scripts and tests still write generated outputs under `result/`.

## Target Layout

- `paper/math/`: current math source of truth.
  - Main authority:
    `paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex`
  - Minimal authority:
    `paper/math/l2pt_l2p_kkt_mainbody_minimal.tex`
  - Supporting full/workbook/report KKT materials stay in the same folder or
    under `paper/math/archive/` when they are not current.
- `paper/figures/` and `paper/tables/`: stable paper-facing outputs copied
  from reproducible runs.
- `paper/notes/`: proof audits, source notes, planning notes, and submission
  checklists.
- `paper/archive/drafts/`: old manuscript-like drafts and historical PDFs that
  should not be used as current authority.
- `archive/generated_20260628/`: previously tracked generated snapshots from
  root-level `result/` and `tmp/`.
- `results/`: ignored config-driven run output directory, keeping only
  `.gitkeep` in Git.
- `src/`, `scripts/`, `configs/`, `tests/`, and `data/`: retain their current
  engineering/data roles, with path references updated to the new paper/results
  layout.

## Safety Rules

- Keep the pre-reorganization checkpoint commit `bd4c2c4` as the rollback
  point.
- Move files rather than deleting tracked material.
- Remove only targeted ignored local artifacts after tracked files have been
  moved out of their old locations.
- Do not move or delete `application_materials/`, `knowledge/`, raw data, or
  source code.
- Update docs and tests in the same change so future agents do not follow the
  old `writing_samples/` and `result/` conventions.

## Verification

- Run `git status --short` to inspect the final move set.
- Run the Python test suite with `pytest -q tests -p no:cacheprovider`.
- Verify no active docs still declare
  `writing_samples/profile_sieve_bai_perron_theory_workbook.tex` as the current
  manuscript authority.
- Verify root-level `result/` and `tmp/` are no longer required for tracked
  project state.
