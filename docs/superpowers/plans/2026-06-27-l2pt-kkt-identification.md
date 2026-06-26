# L2PT KKT Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and integrate the claim that the sample profile-sieve EL construction is characterized by finite-sample KKT conditions, that the profile part lives in \(L^2(P_T)\), and that the limiting population KKT in \(L^2(P)\) identifies the true parameter under uniqueness.

**Architecture:** Split the argument into two KKT layers. The nuisance profiling and residualized score weight are exact \(L^2(P_T)\) projection KKT/normal equations; the empirical likelihood weights are a separate finite-dimensional simplex KKT problem. A final bridge lemma states the additional assumptions needed for the sample KKT system to converge to the \(L^2(P)\) population KKT and for uniqueness to imply recovery of \(\beta_0\).

**Tech Stack:** LaTeX manuscript source in `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`; Python diagnostics in `src/pseel/el.py`, `src/pseel/breaks.py`, and existing pytest files; verification through `pdflatex` and targeted `pytest`.

---

## Verification Verdict

The proposed idea is true after one correction:

1. The profile residualization step is exactly a KKT/normal-equation statement in \(L^2(P_T)\), with empirical inner product \(\langle a,b\rangle_T=T^{-1}a'b\).
2. The EL step is not an \(L^2(P_T)\) projection problem. It is a concave optimization over probability weights on the sample simplex. Its KKT conditions produce the multiplier equation already used in the manuscript:
   \[
   \sum_{t=1}^T\frac{Z_t}{1+\lambda_T Z_t}=0,\qquad 1+\lambda_TZ_t>0.
   \]
3. As \(T\to\infty\), the sample projection KKT can be bridged to \(L^2(P)\) only under uniform LLN, sieve approximation, Gram stability, convex-hull/constraint qualification, and stochastic equicontinuity. If the limiting population moment/KKT equation has a unique solution, then the identified solution is the true parameter.

This should become the recent two-day plan for June 27-28, 2026 (Asia/Shanghai).

## File Map

- Modify: `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`
  - Insert a short geometry subsection after the current EL definition block around lines 145-163.
  - Extend the scalar EL lemma around lines 559-637 with a preceding KKT derivation rather than only the multiplier expansion.
  - Add a high-level population KKT bridge after the deterministic projection algebra section around lines 639-711, or before the main theorem map if that reads better after implementation.
- Modify only if needed: `proof verify.md`
  - Add an audit note that the claim is accepted with the two-layer KKT split.
- Read only unless tests expose drift: `src/pseel/el.py`, `src/pseel/breaks.py`, `tests/test_el_solver.py`, `tests/test_residualize.py`, `tests/test_method_identities.py`, `tests/test_workbook_faithful_breaks.py`.

## Day 1: Finite-Sample KKT Layer

### Task 1: Add the \(L^2(P_T)\) Profile KKT Lemma

**Files:**
- Modify: `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Insert the sample inner-product definition**

Place this immediately after the current norm definition in the primitive setup:

```tex
It is useful to view the finite-sample profiling step as geometry in
\(L^2(P_T)\), where \(P_T=T^{-1}\sum_{t=1}^T\delta_t\) and
\[
    \langle \bm a,\bm b\rangle_T=T^{-1}\bm a'\bm b.
\]
The norm \(\|\cdot\|_T\) is the norm induced by this empirical inner product.
```

- [ ] **Step 2: Insert the lemma after the definitions of \(\widehat{\bm u}\), \(\bm w^c\), and \(\widehat Z_t\)**

Use this exact lemma:

```tex
\begin{lemma}[Sample \(L^2(P_T)\) profile KKT]
\label{lem:sample-l2pt-profile-kkt}
Fix \((\beta,\Lambda)\) and write \(\bm Q=\bm Q_{K,\Lambda}\).  The fitted
value \(\bm Q\widehat{\bm c}\) from
\[
    \min_{\bm c}\|\bm Y-\beta\bm X_{lag}-\bm Q\bm c\|_T^2
\]
is the \(L^2(P_T)\) orthogonal projection of \(\bm Y-\beta\bm X_{lag}\) onto
\(\operatorname{col}(\bm Q)\).  Its residual
\[
    \widehat{\bm u}(\beta,\Lambda)
    =\bm M_{K,\Lambda}(\bm Y-\beta\bm X_{lag})
\]
satisfies the KKT normal equations
\[
    \bm Q'\widehat{\bm u}(\beta,\Lambda)=\bm0.
\]
Likewise, \(\bm w^c(\Lambda)=\bm M_{K,\Lambda}\bm w\) is the residual from the
\(L^2(P_T)\) projection of \(\bm w\) onto \(\operatorname{col}(\bm Q)\), and
\[
    \bm Q'\bm w^c(\Lambda)=\bm0.
\]
\end{lemma}

\begin{proof}
The objective is a convex quadratic.  Its first-order normal equation is
\(\bm Q'(\bm Y-\beta\bm X_{lag}-\bm Q\bm c)=\bm0\).  When \(\bm Q'\bm Q\) is
singular, the coefficient vector need not be unique, but the fitted value is
the Moore--Penrose projection
\[
    \bm Q(\bm Q'\bm Q)^\dagger\bm Q'(\bm Y-\beta\bm X_{lag}),
\]
and the residual is \(\bm M_{K,\Lambda}(\bm Y-\beta\bm X_{lag})\).  Hence
\(\bm Q'\widehat{\bm u}(\beta,\Lambda)=\bm0\).  The same argument with
\(\bm w\) in place of \(\bm Y-\beta\bm X_{lag}\) gives
\(\bm Q'\bm w^c(\Lambda)=\bm0\).
\end{proof}
```

- [ ] **Step 3: Compile the manuscript**

Run:

```powershell
Set-Location 'W:\Research\semipara sieve\writing_samples'
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_bai_perron_theory_workbook.tex
```

Expected: the run reaches `Output written on profile_sieve_bai_perron_theory_workbook.pdf`.

### Task 2: Add the EL Simplex KKT Lemma

**Files:**
- Modify: `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`
- Read: `src/pseel/el.py`

- [ ] **Step 1: Add the finite-sample EL optimization display before Lemma `sol-scalar-el`**

Use this exact text:

```tex
For a scalar score array \(Z_1,\ldots,Z_T\), the empirical likelihood ratio is
obtained from the sample-simplex program
\[
    \max_{\pi_1,\ldots,\pi_T}
    \sum_{t=1}^T\log(T\pi_t)
    \quad\text{subject to}\quad
    \pi_t>0,\quad
    \sum_{t=1}^T\pi_t=1,\quad
    \sum_{t=1}^T\pi_tZ_t=0.
\]
This is a finite-dimensional concave KKT problem.  It should be kept separate
from the \(L^2(P_T)\) projection KKT used to build \(\widehat Z_t\).
```

- [ ] **Step 2: Add the KKT lemma**

Use this exact lemma:

```tex
\begin{lemma}[Sample empirical-likelihood KKT]
\label{lem:sample-el-kkt}
If zero lies in the interior of the convex hull of \(Z_1,\ldots,Z_T\), the
empirical-likelihood simplex program has weights
\[
    \widehat\pi_t
    =
    \frac{1}{T(1+\lambda_TZ_t)},
    \qquad t=1,\ldots,T,
\]
where \(\lambda_T\) is the unique value satisfying
\[
    \sum_{t=1}^T\frac{Z_t}{1+\lambda_TZ_t}=0,
    \qquad
    1+\lambda_TZ_t>0\quad(t\leq T).
\]
The twice negative empirical-likelihood log ratio is therefore
\[
    2\sum_{t=1}^T\log(1+\lambda_TZ_t).
\]
\end{lemma}

\begin{proof}
The Lagrangian first-order condition gives
\[
    \pi_t^{-1}=\nu+\eta Z_t.
\]
Multiplying by \(\pi_t\) and summing over \(t\), while using
\(\sum_t\pi_t=1\) and \(\sum_t\pi_tZ_t=0\), gives \(\nu=T\).  With
\(\lambda_T=\eta/T\),
\[
    \pi_t=\{T(1+\lambda_TZ_t)\}^{-1}.
\]
Substitution into the moment constraint gives the displayed multiplier
equation.  The interior convex-hull condition gives existence, positivity, and
uniqueness of the scalar multiplier.
\end{proof}
```

- [ ] **Step 3: Check code-text consistency**

Verify that the code still matches the mathematical equation:

```powershell
Select-String -Path 'W:\Research\semipara sieve\src\pseel\el.py' -Pattern 'z / denom','2.0 * np.sum'
```

Expected: matches in `empirical_likelihood_scalar`, showing the solver uses
\(\sum z_t/(1+\lambda z_t)=0\) and computes \(2\sum\log(1+\lambda z_t)\).

## Day 2: Population KKT Bridge and Identification

### Task 3: Add the \(L^2(P)\) Population Bridge

**Files:**
- Modify: `writing_samples/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Add a high-level bridge assumption**

Use this exact assumption after the projection algebra section:

```tex
\begin{assumption}[Population KKT bridge and identification]
\label{ass:population-kkt-identification}
Let \(\mathcal B\) be a compact neighborhood of \(\beta_0\).  Uniformly over
\(\beta\in\mathcal B\) and over the partitions used in the theorem, the
sample projection moments and EL score moments obey the required uniform laws
of large numbers, the empirical Gram matrices are stable, the sieve
approximation errors vanish at the rates in Assumption \ref{ass:prim-sieve},
and the EL convex-hull condition holds with probability tending to one.  The
limiting population profile residual \(u_\beta\) and residualized score weight
\(v\) satisfy
\[
    \E\{h(W_{t-1})u_\beta\}=0
    \quad\text{for every }h\in L^2(P_W),
    \qquad
    \E\{h(W_{t-1})v\}=0
    \quad\text{for every }h\in L^2(P_W).
\]
The population identifying equation
\[
    \Psi(\beta)=\E\{u_\beta v\}=0
\]
has the unique solution \(\beta=\beta_0\) in \(\mathcal B\).
\end{assumption}
```

- [ ] **Step 2: Add the bridge proposition**

Use this exact proposition:

```tex
\begin{proposition}[From sample KKT to population KKT]
\label{prop:sample-to-population-kkt}
Under Assumption \ref{ass:population-kkt-identification}, any sequence
\(\widehat\beta_T\in\mathcal B\) whose sample profiled score satisfies
\[
    T^{-1}\sum_{t=1}^T
    \widehat u_t(\widehat\beta_T,\widehat\Lambda)
    w_t^c(\widehat\Lambda)
    =o_p(1)
\]
also satisfies \(\Psi(\widehat\beta_T)=o_p(1)\).  Consequently,
\[
    \widehat\beta_T\overset{p}{\longrightarrow}\beta_0.
\]
\end{proposition}

\begin{proof}
The sample KKT equations in Lemma \ref{lem:sample-l2pt-profile-kkt} place the
profile residual and residualized score weight in the empirical orthogonal
complement of the candidate sieve space.  The uniform laws of large numbers,
Gram stability, and sieve approximation conditions in Assumption
\ref{ass:population-kkt-identification} transfer these empirical orthogonality
relations to the \(L^2(P)\) population orthogonality relations.  Therefore the
sample profiled score is uniformly close to \(\Psi(\beta)\) on
\(\mathcal B\).  If the sample score at \(\widehat\beta_T\) is \(o_p(1)\), then
\(\Psi(\widehat\beta_T)=o_p(1)\).  Compactness of \(\mathcal B\) and uniqueness
of the zero of \(\Psi\) imply \(\widehat\beta_T\to_p\beta_0\).
\end{proof}
```

- [ ] **Step 3: Add the non-overclaiming remark**

Use this exact remark:

```tex
\begin{remark}[What the KKT bridge does and does not prove]
The finite-sample KKT identities are exact.  Consistency for \(\beta\) does
not follow from those identities alone; it also requires uniform convergence,
constraint qualification for EL, sieve approximation control, and population
identification.  Thus the correct conclusion is not merely that the
\(L^2(P_T)\) KKT converges to an \(L^2(P)\) KKT, but that the limiting
population KKT has a unique zero at \(\beta_0\).
\end{remark}
```

### Task 4: Update the Audit Note

**Files:**
- Modify: `proof verify.md`

- [ ] **Step 1: Add a dated note near the top**

Use this exact note:

```markdown
## 2026-06-27 L2PT/L2P KKT Verification

Verdict: accepted with a necessary split. The profile residualization and
residualized score weight are exact \(L^2(P_T)\) projection KKT statements.
The empirical likelihood weights are a separate sample-simplex KKT problem.
Under uniform convergence, sieve approximation, Gram stability, convex-hull
qualification, and uniqueness of the population identifying equation, the
sample KKT system bridges to the \(L^2(P)\) population KKT and identifies
\(\beta_0\).
```

### Task 5: Verification Commands

**Files:**
- Test: `tests/test_el_solver.py`
- Test: `tests/test_residualize.py`
- Test: `tests/test_method_identities.py`
- Test: `tests/test_workbook_faithful_breaks.py`

- [ ] **Step 1: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_el_solver.py tests/test_residualize.py tests/test_method_identities.py tests/test_workbook_faithful_breaks.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Compile manuscript twice for references**

Run:

```powershell
Set-Location 'W:\Research\semipara sieve\writing_samples'
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_bai_perron_theory_workbook.tex
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_bai_perron_theory_workbook.tex
```

Expected: both runs finish and the second run has no undefined references for
the new labels `lem:sample-l2pt-profile-kkt`, `lem:sample-el-kkt`,
`ass:population-kkt-identification`, and `prop:sample-to-population-kkt`.

- [ ] **Step 3: Commit the finished writing change**

Run after review:

```powershell
git add writing_samples/profile_sieve_bai_perron_theory_workbook.tex proof\ verify.md docs/superpowers/plans/2026-06-27-l2pt-kkt-identification.md
git commit -m "docs: add KKT bridge plan for profile sieve EL"
```

Expected: commit succeeds only after the manuscript and tests pass.

## Completion Criteria

- The manuscript explicitly distinguishes \(L^2(P_T)\) projection KKT from EL simplex KKT.
- The sample-to-population statement says uniqueness plus uniform convergence and regularity identify \(\beta_0\).
- The plan does not claim that EL itself is an \(L^2(P_T)\) projection.
- The targeted tests and LaTeX compile complete successfully during implementation.
