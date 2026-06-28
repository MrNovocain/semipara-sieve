# Project Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository around the KKT workbook math main body while preserving reproducibility and rollback safety.

**Architecture:** Create a `paper/` hierarchy for current math, paper-facing assets, notes, and archived drafts. Move previously tracked generated snapshots out of root-level `result/` and `tmp/` into `archive/generated_20260628/`, then update docs, scripts, tests, and ignore rules to use `paper/` and `results/`.

**Tech Stack:** Git, PowerShell, Python package under `src/pseel`, pytest, LaTeX source/PDF assets.

---

### Task 1: Establish Paper Directories

**Files:**
- Create directories under `paper/`
- Move tracked files from `writing_samples/`

- [ ] **Step 1: Create the destination directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path paper\math,paper\math\archive,paper\figures,paper\tables,paper\notes,paper\archive\drafts,archive\generated_20260628 | Out-Null
```

Expected: directories exist.

- [ ] **Step 2: Move current KKT math files**

Run `git mv` for the current KKT mainbody/minimal files from
`writing_samples/workbook/` into `paper/math/`.

Expected: `paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex` and
`paper/math/l2pt_l2p_kkt_mainbody_minimal.tex` exist.

- [ ] **Step 3: Move supporting KKT materials**

Move non-current KKT workbook/report files into `paper/math/archive/`.

Expected: older KKT workbook, full detail, primitive solved workbook, and
professor-friendly report files are retained but clearly not current authority.

- [ ] **Step 4: Move stable paper assets**

Move `writing_samples/figures/` to `paper/figures/` and
`writing_samples/tables/` to `paper/tables/`.

Expected: paper-facing figures and tables live under `paper/`.

- [ ] **Step 5: Move notes and audits**

Move `writing_samples/source_notes/`, `proof verify.md`,
`writing_samples/statistica_sinica_submission_checklist.md`, and
`writing_samples/implementation_plan.md` into `paper/notes/`.

Expected: notes are grouped under `paper/notes/`.

- [ ] **Step 6: Move old drafts**

Move remaining tracked manuscript-like `writing_samples/*.tex` and matching
PDFs into `paper/archive/drafts/`.

Expected: `writing_samples/` no longer contains current tracked paper source.

### Task 2: Archive Generated Snapshots

**Files:**
- Move tracked `result/`
- Move tracked `tmp/`
- Update `.gitignore`

- [ ] **Step 1: Move tracked legacy generated results**

Run:

```powershell
git mv result archive/generated_20260628/result
```

Expected: historical generated snapshots remain tracked under
`archive/generated_20260628/result/`.

- [ ] **Step 2: Move tracked temporary render checks**

Run:

```powershell
git mv tmp archive/generated_20260628/tmp
```

Expected: historical render checks remain tracked under
`archive/generated_20260628/tmp/`.

- [ ] **Step 3: Update ignore rules**

Add root-level ignores for:

```gitignore
result/
tmp/
pytest-cache-files-*/
```

Expected: new generated scratch material does not appear as untracked project
state.

### Task 3: Update Project References

**Files:**
- Modify `README.md`
- Modify `agent.md`
- Modify `paper/notes/*.md`
- Modify `docs/superpowers/plans/2026-06-27-l2pt-kkt-identification.md`
- Modify scripts/tests that use `result/` or `writing_samples/`

- [ ] **Step 1: Update canonical manuscript authority**

Replace current-source references with:

```text
paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex
```

and record the minimal source:

```text
paper/math/l2pt_l2p_kkt_mainbody_minimal.tex
```

Expected: active docs no longer state that
`writing_samples/profile_sieve_bai_perron_theory_workbook.tex` is the current
authority.

- [ ] **Step 2: Update paper asset paths**

Replace `writing_samples/figures/` with `paper/figures/` and
`writing_samples/tables/` with `paper/tables/` in scripts and notes.

Expected: scripts write stable paper copies to `paper/`.

- [ ] **Step 3: Update generated-output paths**

Replace script defaults and tests that write to `result/` with `results/` or a
test-specific temporary path.

Expected: new generated outputs use ignored run-output locations.

- [ ] **Step 4: Update archive references**

For historical files, point notes to `paper/archive/drafts/` or
`paper/math/archive/` rather than old `writing_samples/` paths.

Expected: old drafts remain findable but not authoritative.

### Task 4: Clean Ignored Local Artifacts

**Files:**
- Remove local ignored build/cache artifacts only

- [ ] **Step 1: Remove LaTeX build artifacts**

Remove ignored `*.aux`, `*.log`, `*.out`, `*.toc`, `*.fls`,
`*.fdb_latexmk`, and `*.synctex.gz` files under old and new paper folders.

Expected: source/PDF files remain; build clutter is gone.

- [ ] **Step 2: Remove Python caches**

Remove `__pycache__/` directories under the repository.

Expected: no Python bytecode cache directories remain.

- [ ] **Step 3: Remove pytest scratch directories**

Remove root-level `pytest-cache-files-*`, `tmp/pytest`, and ignored
`results/_test_*` / `results/_verify_*` directories.

Expected: local test scratch material is gone while named reproducible run
directories under `results/` remain.

### Task 5: Verify and Checkpoint

**Files:**
- No manual source edits except fixes found by verification

- [ ] **Step 1: Inspect status**

Run:

```powershell
git status --short
```

Expected: changes reflect intended moves and edits.

- [ ] **Step 2: Search for stale authority references**

Run:

```powershell
rg -n "profile_sieve_bai_perron_theory_workbook|writing_samples|result/" README.md agent.md docs paper scripts tests src configs
```

Expected: matches are archive-only or intentionally historical.

- [ ] **Step 3: Run tests**

Run:

```powershell
pytest -q tests -p no:cacheprovider
```

Expected: tests pass.

- [ ] **Step 4: Commit the reorganization**

Run:

```powershell
git add -A
git commit -m "chore: organize project layout"
```

Expected: one reorganization commit after the pre-reorg checkpoint.
