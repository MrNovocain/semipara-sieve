# Empirical Section Plan: Goyal-Welch Flagship Application

## Central Question

Does canonical equity-premium predictability survive after allowing the nuisance component to be unknown, nonlinear, and piecewise unstable?

The empirical model is

```text
Y_t = beta_0 X_{t-1} + m_{j(t)}(W_{t-1}) + U_t,
```

where beta_0 is the stable predictive slope and the unknown nuisance functions m_j may break across regimes.

## Dataset Choice

The flagship benchmark is the Goyal-Welch equity-premium prediction dataset. It is famous enough that the proposed method can be read against a recognized empirical frontier rather than as an isolated demonstration.

The raw input is saved at:

```text
data/raw/goyal_welch/PredictorData2025.xlsx
```

The cleaned monthly panel is saved at:

```text
data/processed/goyal_welch_monthly.csv
```

The raw-data provenance record is saved at:

```text
data/raw/goyal_welch/provenance.json
```

The default dependent variable is the monthly equity premium, constructed as the market return less the risk-free rate. The visual diagnostic default is X=bm and W=ntis, because the dp/tms smoke pair did not display the paper narrative: it had no conventional rejection and no selected nuisance break. The bm/ntis pair uses canonical Goyal-Welch variables and shows the intended diagnostic pattern: conventional predictability weakens after nonlinear and break-aware nuisance profiling.

## Empirical Table

The paper-facing comparison table should report four implemented methods:

1. Standard predictive regression.
2. Linear Bai-Perron break model.
3. No-break sieve empirical likelihood.
4. Proposed block-sieve break-aware profile empirical likelihood.

The table and figure should not include an ad hoc persistent-predictor correction. The pipeline records the lagged predictor AR(1) coefficient as rho_hat for persistence diagnostics, but a Campbell-Yogo or Stambaugh-style correction must be implemented separately before it is reported as a benchmark.

The current visual diagnostic output is:

```text
paper/figures/goyal_welch_method_comparison.png
```

The current grid audit output is:

```text
archive/generated_20260628/result/goyal_welch_empirical/goyal_welch_grid_scan.csv
```

The grid scan is kept to make the visual choice auditable rather than hidden.
The current reproducible empirical output is:

```text
archive/generated_20260628/result/goyal_welch_empirical/goyal_welch_comparison.csv
paper/tables/goyal_welch_empirical_comparison.tex
```

The paper-facing version should report beta estimates, confidence intervals or empirical-likelihood confidence sets, p-values, selected break number, selected break dates, and a compact description of the fitted nuisance channel.

## Current Evidence Interpretation

The current Goyal-Welch bm/ntis diagnostic should be read as evidence of fragility, not as proof that the proposed method is superior in real data. Under the paper-faithful workbook one-break RSS-plus-penalty selector, OLS and the linear break benchmark reject beta=0 at p=0.012, while no-break sieve EL and break-aware sieve EL both give p=0.251 and the selected nuisance break count is q=0. The lagged predictor is highly persistent, with rho_hat about 0.988, so persistence remains a major diagnostic caveat. The earlier q=1 pattern was an exploratory penalty result, not the paper-facing workbook-penalty result.

The paper-ready claim is:

```text
The canonical linear signal is reproduced, but it weakens once nonlinear and broken nuisance geometry is profiled out.
```

The paper should not claim:

```text
A higher real-data p-value proves the proposed method is better.
```

Superiority must be established by calibrated Monte Carlo size, coverage, and power evidence where beta_0 is known.

## Figures And Diagnostics

The empirical section should include:

1. Selected break dates and, where feasible, break-date confidence regions.
2. Regime-specific fitted nuisance functions m_j(W).
3. Residualized predictor M_{K,hat Lambda}X.
4. Rolling or subsample estimates of beta under conventional and break-aware methods.
5. A visual comparison of standard and break-aware beta evidence across sample windows.

The most persuasive empirical figure is not a generic p-value plot. It is a time/subsample comparison showing which part of the apparent persistent signal remains after the broken nuisance geometry is removed.

## Calibrated Monte Carlo Role

Real data alone cannot prove superior inference because beta_0 is unknown. The empirical section therefore needs calibrated simulations based on the benchmark data.

The calibrated design should show:

1. Correct rejection probability under H0.
2. Correct confidence-set coverage.
3. Useful power under genuine alternatives.
4. Robustness when nuisance breaks are present.

The calibration target should be the persistence and scale of the Goyal-Welch predictor/control system, not an unrelated toy DGP.

## Robustness Grid

Vary:

```text
K, q_max, penalty, X, W, weight function, sample period, frequency.
```

Also compare no-break, one-break, multiple-break, linear-break, and nonlinear-break specifications. The main conclusion should not depend on one sieve dimension, one penalty, or one predictor-control pair.

## Secondary Application

Keep Ghana cocoa/weather as the secondary application. Its role is different from Goyal-Welch:

```text
Goyal-Welch: visibility and direct comparison with canonical finance literature.
Ghana weather: cleaner external nuisance covariates and stronger exogeneity narrative.
```

Do not let the secondary application displace the flagship benchmark.

## Governing Narrative

The empirical section should not claim that the proposed p-value is "better." The claim should be:

```text
We reproduce canonical predictability evidence and ask which part survives after broken nonlinear nuisance geometry is removed.
```

The desired high-level pattern is:

```text
classical inference detects predictability;
nonlinear adjustment weakens it;
break-aware profile EL shows that part of the signal belonged to the broken nuisance component.
```
