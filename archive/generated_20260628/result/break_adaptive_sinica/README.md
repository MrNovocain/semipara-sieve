# Break-Adaptive Statistica Sinica Results

This directory contains a historical reproducible numerical-evidence pack originally prepared for the noncanonical geometric draft. The current manuscript source of truth is `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`; see `writing_samples/source_notes/noncanonical_manuscript_sources.md` for the legacy manuscript inventory.

## Reproduction Command

Run from the repository root:

```powershell
python scripts\break_adaptive_results.py --mc-reps 300 --output-dir result/break_adaptive_sinica
```

## Files

- `break_mc_raw.csv`: 7,200 replication-level Monte Carlo rows.
- `break_mc_summary.csv`: grouped rejection rates and diagnostics used in Table 1.
- `ghana_break_empirical.csv`: stable and break-adaptive empirical EL statistics.
- `ghana_break_empirical_meta.json`: sample dates, selected breaks, and weather-index loadings.

The script also writes manuscript-ready artifacts to:

- `writing_samples/tables/break_mc_summary.tex`
- `writing_samples/tables/ghana_break_empirical.tex`
- `writing_samples/figures/break_mc_false_rejection.png`
- `writing_samples/figures/ghana_weather_response.png`

## Data Inputs

- `data/processed/cocoa_ghana.csv`

## Verification

The historical geometric-draft build was verified with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_geometric_paper.tex
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_geometric_supplement.tex
python -m pytest -q -p no:cacheprovider
```
