# Source Logic Versus KKT Mainbody Audit

## Scope

Current manuscript authority:

```text
paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex
```

Compared source directory:

```text
src/pseel/
```

This audit compares code logic against the current KKT geometry mainbody and its one-break profile-sieve implementation route. It does not treat archived manuscript-like TeX files as authority.

## Manuscript Logic To Match

The workbook's core logic is:

1. Model: `Y_t = beta_0 X_{t-1} + m_j(W_{t-1}) + U_t`, where breaks are in the nonlinear nuisance component and `beta_0` is stable.
2. Conditioning: conditional on `D_T = sigma{X_{t-1}, W_{t-1}, w_t}`, so the residualized score weight is fixed in conditional arguments.
3. Projection: construct block sieve design `Q_{K,Lambda}` and residual maker `M_{K,Lambda}`.
4. Score: use `uhat(beta,Lambda) = M_{K,Lambda}(Y - beta X)` and residualized score weight `M_{K,Lambda} w`, then form scalar empirical-likelihood scores.
5. Break search: minimize null-imposed profile RSS over admissible minimum-spaced partitions.
6. Main estimated-partition theory: one-break truth, `q_0=1`, with localization `|hat k-k_0|=O_p(r_T)`, `r_T=R_T/Delta_T^2`, `R_T=K+log T+T a_K^2`.
7. Unknown-order theory: concrete penalty `pen_T(q,K)=q kappa_T R_T`, with `kappa_T -> infinity` and `kappa_T R_T=o(T Delta_T^2)`.
8. Limitation: multiple breaks are only an extension sketch, not the main proved result.

## File-By-File Comparison

### `src/pseel/__init__.py`

Role in source: minimal package export for `Dataset`, `ELResult`, `MethodResult`, and `empirical_likelihood_scalar`.

Workbook alignment: infrastructure only. It does not encode manuscript assumptions, partitions, score weights, or break logic.

Status: harmless but incomplete relative to the workbook.

### `src/pseel/basis.py`

Role in source: builds a polynomial basis with an intercept and optional standardization of the observed nuisance covariate.

Workbook counterpart: `P_K(W_{t-1})` with first basis element constant, basis envelope, Gram stability, and sieve approximation assumptions.

Alignment:

- Correctly includes an intercept, matching the workbook's `p_1(w) == 1` requirement.
- Produces a finite polynomial design matrix suitable for projection.

Gaps:

- Does not track `K=K_T` rates, envelopes `zeta_K`, or smoothness/approximation rates `a_K`.
- Standardization makes the implemented basis data-normalized; this is acceptable computationally but should be stated as a numerical basis normalization rather than literally the abstract basis in the proof.

Status: aligned as a numerical basis constructor, not a proof-level assumption checker.

### `src/pseel/residualize.py`

Role in source: forms projections and residuals using `(P'P)^{-1}P'z`, requiring full column rank.

Workbook counterpart: `M_{K,Lambda}=I-Q(Q'Q)^dagger Q'` and projection identities.

Alignment:

- Implements the same residualization logic on full-rank designs.
- The full-rank requirement is compatible with the workbook's Gram-stability event.

Gaps:

- The workbook defines projections with Moore-Penrose inverses for algebraic existence; the code rejects rank-deficient designs instead.
- This is numerically conservative but should not be described as covering arbitrary singular candidate partitions.

Status: aligned on the regular event; stricter than the manuscript's algebraic definition.

### `src/pseel/el.py`

Role in source: scalar empirical-likelihood solver using convex-hull checks and a scalar Lagrange multiplier root.

Workbook counterpart: scalar empirical-likelihood log ratio `ell_T(Z_1,...,Z_T)` and convex-hull condition.

Alignment:

- Directly implements the one-dimensional EL statistic used by the workbook.
- Rejects infeasible convex-hull cases with `inf`, matching the logic that convex-hull feasibility is required.

Gaps:

- Does not implement the asymptotic expansion; it is the finite-sample numerical solver only.
- Uses finite tolerance and drops nonfinite scores; the manuscript assumes valid score arrays.

Status: strongly aligned computationally.

### `src/pseel/contracts.py`

Role in source: dataclasses for simulated data and method results.

Workbook counterpart: model arrays `Y`, `X_lag`, `W_lag`, `U`, and nuisance mean `m(W)`.

Alignment:

- Stores the correct basic model arrays.
- `u` and `m_w` enable oracle/profile comparisons in Monte Carlo.

Gaps:

- No field for partitions, score weights `w_t`, conditional design `D_T`, or break metadata.
- Dataset object is no-break/stable-nuisance oriented.

Status: aligned for stable/no-break Monte Carlo; incomplete for the Bai-Perron workbook.

### `src/pseel/dgp.py`

Role in source: predictive AR(1) Monte Carlo DGP with stationary `W`, persistent `X`, nonlinear stable nuisance `m(W)`, and Gaussian innovations.

Workbook counterpart: conditional-on-design model with unknown possibly broken nuisance `m_j(W)` and conditional independent sub-Gaussian errors.

Alignment:

- Supports the basic predictive-regression form `Y=m(W)+beta X+U`.
- Allows persistent predictors, which is relevant to the empirical motivation.
- Gaussian innovations are sub-Gaussian unconditionally.

Gaps and caveats:

- No nuisance breaks. It represents `q_0=0`, not the one-break Bai-Perron workbook case.
- In designs with correlated innovations, conditioning on the generated `X` path can violate the workbook's conditional mean-zero/independence route. These cases are useful stress tests, not literal proof-compliant DGPs.
- It does not generate `m_0`, `m_1`, jump size `Delta_T`, or localization targets.

Status: legacy/stable Monte Carlo engine; not the Perron-Bai one-break DGP.

### `src/pseel/methods.py`

Role in source: no-break profile/oracle/intercept/efficient EL methods.

Workbook counterpart: known-partition profile EL with `Lambda` fixed, especially the no-break special case.

Alignment:

- `ProfileBoundedEL` matches the stable/no-break profile score: residualize `Y-beta X` and a bounded score weight against `P_K(W)`.
- `OracleBoundedEL` matches the oracle comparison logic used in simulations.
- Orthogonality diagnostics correspond to projection identities.

Gaps:

- No block-sieve partition `Q_{K,Lambda}`.
- No estimated break, no unknown-order selection, no `R_T`, `r_T`, or one-break bridge.
- Uses configurable bounded weights such as `tanh(X/b)`, but the workbook denotes a generic score weight `w_t`; the code should be read as the special choice `w_t=tanh(X_{t-1}/b)` when aligned to the manuscript.

Status: aligned with the no-break/known-stable subcase only.

### `src/pseel/breaks.py`

Role in source: block-sieve construction, profile residualization, profile RSS, empirical-likelihood statistic, and dynamic-programming partition selection.

Workbook counterpart: this is the closest implementation of the Bai-Perron workbook logic.

Alignment:

- `block_sieve_matrix` implements the block design `Q_{K,Lambda}`.
- `residualize_against` implements `M_{K,Lambda} z` on the full-rank/lstsq path.
- `profile_rss` implements `RSS_K(beta,Lambda)`.
- `profile_el_statistic` implements the residualized score and scalar EL statistic.
- `select_profile_partition` implements fixed/unknown-order profile-RSS segmentation over minimum segment sizes.

Important mismatches:

- The workbook's main theorem is one-break truth; the code allows `q_max > 1`, but that should be treated as computational exploration unless the multiple-break extension is proved.
- The workbook's unknown-order penalty is `q kappa_T R_T`; the code uses a BIC-style criterion `T log(RSS/T) + penalty_multiplier * effective_dim * log(T)`. This is not the same penalty.
- The workbook's score weight is a generic `w_t` included in `D_T`; the code hard-codes a residualized `tanh(weight_b * x)` instrument inside `profile_el_statistic`.
- The code uses polynomial basis standardization separately within each segment; the proof's basis is abstract and normalized by assumptions.
- The code does not compute or verify `Delta_T`, `R_T`, `r_T`, Gram stability, convex-hull sign mass, score-bias rates, or localization rates.

Status: partial implementation of the workbook's computational skeleton, not a complete implementation of the workbook's theorem conditions.

### `src/pseel/goyal_welch.py`

Role in source: empirical Goyal-Welch data pipeline and comparison table/figure.

Workbook counterpart: empirical application of the break-aware profile-sieve EL idea.

Alignment:

- Uses `profile_el_statistic` and `select_profile_partition` from `breaks.py` for no-break and break-aware sieve EL rows.
- Builds paper-facing empirical outputs with provenance and persistence diagnostics.
- Keeps persistence as `rho_hat`, not as a fake persistent-predictor benchmark.

Gaps and caveats:

- It is an application layer, not a proof-level implementation of the workbook's assumptions.
- The row is now named `linear_break_benchmark`, which correctly describes the current local segmented linear benchmark and avoids claiming a full external Bai-Perron implementation.
- The selected break and EL p-values now use the workbook one-break RSS-plus-penalty selector in paper-facing mode.
- The empirical data do not prove superiority because `beta_0` is unknown; superiority must still come from calibrated simulations.

Status: useful empirical prototype; paper-facing labels need care.

### `src/pseel/diagnostics.py`

Role in source: simulation diagnostics for size, oracle equivalence, projection identities, convex-hull feasibility, and negative/stress scenarios.

Workbook counterpart: numerical stress checks for the stable/profile EL logic and parts of the projection/EL theory.

Alignment:

- Checks projector symmetry/idempotence-like behavior, orthogonality, EL scaling, feasibility, chi-square fit, and DS/DV score deviations.
- Supports the workbook's concern that projection and EL regularity must be checked, not assumed blindly.

Gaps:

- No break-aware DGP or one-break localization diagnostics.
- No direct check of `R_T`, `r_T`, jump size `Delta_T`, order consistency, or estimated-partition Wilks.
- Scenarios are mostly stable/no-break or stress designs.

Status: strong diagnostic infrastructure for stable/no-break theory; incomplete for the Perron-Bai one-break module.

### `src/pseel/experiment.py`

Role in source: config-driven Monte Carlo runner.

Workbook counterpart: reproducible numerical evidence for Wilks behavior and method comparisons.

Alignment:

- Provides auditable repeated simulations over `T`, persistence, `K`, methods, and score weights.
- Saves summaries needed for size/oracle-equivalence evidence.

Gaps:

- Uses `PredictiveAR1DGP`, so it does not currently run the one-break nuisance DGP required to validate estimated-partition Wilks.
- Does not call `breaks.py` partition selection.

Status: reproducibility engine for the older stable/no-break Monte Carlo layer.

### `src/pseel/io.py`

Role in source: reproducibility metadata, config hashes, source hashes, environment capture, and run outputs.

Workbook counterpart: not theoretical, but supports auditability of computational claims.

Alignment:

- Strongly aligned with the project rule that every computational claim should be auditable from source and saved outputs.

Gaps:

- No manuscript-specific theorem logic.

Status: infrastructure; good.

### `src/pseel/checks.py`

Role in source: validates Monte Carlo configs.

Workbook counterpart: none directly.

Alignment:

- Ensures valid registered DGPs/methods and positive `K`, but not theoretical rate conditions.

Gaps:

- Does not check `K/sqrt(T)`, `zeta_K`, `R_T`, `r_T`, minimum spacing for break search, or penalty compatibility.

Status: infrastructure; incomplete for Perron-Bai theoretical-rate compliance.

### `src/pseel/registry.py`

Role in source: registers DGPs, methods, weights, and bases.

Workbook counterpart: none directly.

Alignment: neutral infrastructure.

Gaps: no manuscript logic.

Status: infrastructure.

### `src/pseel/weights.py`

Role in source: bounded `tanh` and unbounded linear score-weight transformations.

Workbook counterpart: score weight `w_t`, boundedness, and score regularity assumptions.

Alignment:

- `TanhWeight` provides a bounded score weight candidate, consistent with the manuscript's bounded-weight route.
- `LinearWeight` is useful for efficiency/stress comparisons.

Gaps:

- `LinearWeight` violates the bounded-weight route unless separately justified.
- Naming can confuse the nuisance covariate `W` with the score weight `w_t`; source code often uses `w_lag` for nuisance and `weight` for score transformation, while the workbook uses lower-case `w_t` for the score weight.

Status: aligned for bounded `tanh`; stress-only for linear unless extra assumptions are supplied.

### `src/pseel/metrics.py`

Role in source: adds rejection indicators from chi-square critical values.

Workbook counterpart: Wilks chi-square limits.

Alignment: implements the final chi-square rejection rule implied by Wilks.

Gaps: no feasibility or theorem-condition checks.

Status: small aligned utility.

### `src/pseel/run.py`

Role in source: command-line entry point for config-driven Monte Carlo.

Workbook counterpart: none directly; reproducibility support.

Alignment: infrastructure only.

Status: infrastructure.

## Gap-Closure Status After Paper-Faithful Code Pass

1. `breaks.py` now has `select_workbook_partition`, which uses the workbook criterion `RSS + q kappa_T R_T`, and the old BIC-style selector is retained only as `select_profile_partition` for legacy/exploratory comparisons.
2. Paper mode is now one-break by default: `select_workbook_partition` rejects `q_max > 1` unless `allow_multiple_breaks=True`, in which case rows are marked `exploratory_multiple_breaks`.
3. `src/pseel/dgp.py` now includes `BrokenNuisanceAR1DGP`, and `src/pseel/workbook_mc.py` runs known-partition and estimated-partition one-break Monte Carlo replications with break error, selected order, EL statistics, and theorem-condition diagnostics.
4. The Goyal-Welch empirical method label is now `linear_break_benchmark`, not `linear_bai_perron_break_model`.
5. `profile_el_statistic` now separates `nuisance_w` from `score_weight`; legacy `w=` is accepted only as a compatibility alias for the nuisance covariate.
6. `diagnose_workbook_conditions` now reports Gram stability, convex-hull sign mass, `Delta_T`, `R_T`, `r_T`, `r_T/sqrt(T)`, and a score-bias proxy when an approximation-error vector is supplied.

## Remaining Limits

The code is now aligned with the workbook's one-break computational logic, but it still does not prove the theorem. Multiple-break searches remain exploratory unless the manuscript promotes the multiple-break sketch to a full theorem. Empirical Goyal-Welch results under the workbook penalty may be more conservative than exploratory BIC-style scans; paper-facing claims should use the workbook-penalty output.
