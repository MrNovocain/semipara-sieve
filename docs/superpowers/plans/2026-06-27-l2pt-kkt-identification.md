# KKT L2 Geometry for Sieve EL Implementation Plan

> Superseded by the 2026-06-28 project reorganization. The current math main body is paper/math/l2pt_l2p_kkt_workbook_v2_mainbody.tex; the compact body is paper/math/l2pt_l2p_kkt_mainbody_minimal.tex. This file is retained as a historical implementation plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the manuscript proof around three linked KKT systems: profile projection KKT in \(L^2(P_T)\), empirical-likelihood simplex KKT with local \(L^2(P_T)\) curvature, and the limiting \(L^2(P)\) population KKT that identifies \(\beta_0\).

**Architecture:** Keep the stochastic CLT/LLN/max lemmas, but demote them from the proof architecture to supporting consequences that localize the EL KKT. The main narrative becomes: profiling solves an exact empirical projection problem; EL solves an exact entropy KKT problem whose Hessian is the empirical second moment \(P_TZ^2\) near the root; empirical inner products converge to population \(L^2(P)\) inner products, where the population normal-score equation has a unique zero at \(\beta_0\).

**Tech Stack:** LaTeX source in `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`; audit notes in `paper/notes/proof verify.md`; mathematical consistency checks against `src/pseel/el.py`, `src/pseel/breaks.py`, and targeted pytest files; manuscript verification with `pdflatex`.

---

## Current Status

The repository already has partial implementation of the older two-layer plan:

- `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex` defines \(\langle a,b\rangle_T=T^{-1}a'b\).
- It has a sample \(L^2(P_T)\) profile KKT lemma.
- It has a sample EL simplex KKT lemma.
- It has a high-level population KKT bridge.

This plan upgrades that draft. The key change is to avoid saying only "EL is not \(L^2(P_T)\)." The sharper statement is:

\[
\text{EL is an entropy/simplex KKT problem whose local Hessian is the }L^2(P_T)\text{ curvature }P_TZ^2.
\]

The final proof should read as a single geometry:

\[
\boxed{
\text{profile KKT in }L^2(P_T)
\quad+\quad
\text{EL KKT with local }L^2(P_T)\text{ curvature}
\quad+\quad
L^2(P_T)\to L^2(P)\text{ identification.}
}
\]

## File Map

- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`
  - Refine the existing sample inner-product paragraph near the primitive setup.
  - Keep and polish the existing sample profile KKT lemma.
  - Replace the current EL separation paragraph with an EL-local-curvature explanation.
  - Add a lemma deriving \(\lambda_T=S_1/S_2+o_p(T^{-1/2})\) directly from the EL KKT Hessian.
  - Tighten the population KKT bridge so it identifies \(\beta_0\) without pretending consistency proves Wilks.
  - Add a proof-roadmap remark after known-partition Wilks.
- Modify: `proof verify.md`
  - Add an audit note that the KKT architecture is accepted with the local-curvature refinement.
- Read/check: `src/pseel/el.py`
  - Confirm code solves \(\sum z_t/(1+\lambda z_t)=0\) and computes \(2\sum\log(1+\lambda z_t)\).
- Test: `tests/test_el_solver.py`, `tests/test_residualize.py`, `tests/test_method_identities.py`, `tests/test_workbook_faithful_breaks.py`

## Day 1: Build the Three-KKT Narrative

### Task 1: Polish the Empirical \(L^2(P_T)\) Profile KKT

**Files:**
- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Confirm the empirical inner product paragraph is present**

Run:

```powershell
Select-String -Path 'W:\Research\semipara sieve\paper\archive\drafts\profile_sieve_bai_perron_theory_workbook.tex' -SimpleMatch '\langle \bm a,\bm b\rangle_T'
```

Expected: one match in the primitive setup.

- [ ] **Step 2: Replace the current short paragraph with this exact framing**

Use this text immediately after the definition of \(\|\bm a\|_T\):

```tex
The empirical norm is the Euclidean geometry of the sample written as
\(L^2(P_T)\).  Throughout the proof, \(P_T=T^{-1}\sum_{t=1}^T\delta_t\) and
\[
    \langle \bm a,\bm b\rangle_T=T^{-1}\bm a'\bm b.
\]
Thus \(\|\cdot\|_T\) is the norm induced by the empirical inner product.  This
notation is not cosmetic: the profile step below is an exact projection KKT
statement in \(L^2(P_T)\).
```

- [ ] **Step 3: Revise the first sentence of `lem:sample-l2pt-profile-kkt`**

Use this sentence:

```tex
Fix \((\beta,\Lambda)\), write \(\bm Q=\bm Q_{K,\Lambda}\), and equip
\(\mathbb R^T\) with the empirical inner product
\(\langle \bm a,\bm b\rangle_T=T^{-1}\bm a'\bm b\).
```

- [ ] **Step 4: Add this closing sentence to the profile KKT proof**

Place it immediately before `\end{proof}` in `lem:sample-l2pt-profile-kkt`:

```tex
These two normal equations are the finite-sample source of the nuisance
annihilation used later: any component in \(\operatorname{col}(\bm Q)\) is
orthogonal to both the profiled residual and the residualized score weight.
```

### Task 2: Recast EL as Simplex KKT with Local \(L^2(P_T)\) Curvature

**Files:**
- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`
- Read: `src/pseel/el.py`

- [ ] **Step 1: Replace the current EL separation paragraph**

Find the paragraph beginning:

```tex
For a scalar score array \(Z_1,\ldots,Z_T\), the empirical likelihood ratio is
obtained from the sample-simplex program
```

Replace through the sentence that says it should be kept separate from projection KKT with this text:

```tex
For a scalar score array \(Z_1,\ldots,Z_T\), empirical likelihood is not a
projection problem.  It is the entropy KKT problem on the sample simplex
\[
    \max_{\pi_1,\ldots,\pi_T}
    \sum_{t=1}^T\log(T\pi_t)
    \quad\text{subject to}\quad
    \pi_t>0,\quad
    \sum_{t=1}^T\pi_t=1,\quad
    \sum_{t=1}^T\pi_tZ_t=0.
\]
Its connection with \(L^2(P_T)\) appears only locally: near a small multiplier,
the derivative of the EL KKT equation has curvature
\(-T^{-1}\sum_t Z_t^2\), the empirical \(L^2(P_T)\) norm squared of the score.
```

- [ ] **Step 2: Add this curvature lemma after `lem:sample-el-kkt`**

```tex
\begin{lemma}[Local \(L^2(P_T)\) curvature of the EL KKT]
\label{lem:sample-el-local-l2-curvature}
Let \(S_1=\sum_{t=1}^TZ_{t,T}\), \(S_2=\sum_{t=1}^TZ_{t,T}^2\), and
\(M_T=\max_{t\leq T}|Z_{t,T}|\).  Under the conditions of Lemma
\ref{lem:sol-scalar-el}, the EL multiplier satisfies
\[
    \lambda_T=\frac{S_1}{S_2}+o_p(T^{-1/2}),
    \qquad
    |\lambda_T|M_T=o_p(1).
\]
Equivalently,
\[
    \lambda_T
    =
    \frac{T^{-1}\sum_tZ_{t,T}}
         {T^{-1}\sum_tZ_{t,T}^2}
    +o_p(T^{-1/2}).
\]
Thus the local Hessian of the EL KKT equation is the empirical
\(L^2(P_T)\) curvature \(T^{-1}\sum_tZ_{t,T}^2\).
\end{lemma}

\begin{proof}
The EL KKT equation is
\[
    0=\sum_{t=1}^T\frac{Z_{t,T}}{1+\lambda_TZ_{t,T}}.
\]
The convex-hull condition gives a unique multiplier on the interval where all
\(1+\lambda_TZ_{t,T}\) are positive.  As in Lemma \ref{lem:sol-scalar-el},
\[
    S_1
    =
    \lambda_T\sum_{t=1}^T
      \frac{Z_{t,T}^2}{1+\lambda_TZ_{t,T}},
\]
which implies \(\lambda_T=O_p(T^{-1/2})\) and
\(|\lambda_T|M_T=o_p(1)\).  Taylor expansion of the KKT equation around
\(\lambda=0\) gives
\[
    0=S_1-\lambda_TS_2+R_T,
    \qquad
    |R_T|
    \leq
    \frac{\lambda_T^2M_TS_2}{1-|\lambda_T|M_T}
    =o_p(\sqrt T).
\]
Since \(S_2\) is bounded below by order \(T\), the displayed expansion for
\(\lambda_T\) follows.
\end{proof}
```


- [ ] **Step 3: Shorten the proof of `lem:sol-scalar-el` by referencing the curvature lemma**

Replace the portion from `The convex-hull condition gives a unique multiplier` through the display `\lambda_T=\frac{S_1}{S_2}+o_p(T^{-1/2}).` with:

```tex
Lemma \ref{lem:sample-el-local-l2-curvature} gives
\[
    \lambda_T=\frac{S_1}{S_2}+o_p(T^{-1/2}),
    \qquad
    |\lambda_T|M_T=o_p(1).
\]
```

- [ ] **Step 4: Check code-text consistency**

Run:

```powershell
Select-String -Path 'W:\Research\semipara sieve\src\pseel\el.py' -SimpleMatch -Pattern 'z / denom','2.0 * np.sum'
```

Expected: matches show the implementation solves the EL KKT equation and computes the log ratio.

### Task 3: Add the Three-KKT and Deterministic-Good-Event Roadmap

**Files:**
- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Insert the three-KKT roadmap after the EL definition block**

```tex
\begin{remark}[Three KKT systems]
\label{rem:three-kkt-systems}
The proof uses three linked KKT systems.  First, nuisance profiling solves a
quadratic projection KKT problem in \(L^2(P_T)\), producing
\(\widehat{\bm u}(\beta,\Lambda)\) and \(\bm w^c(\Lambda)\) in the empirical
orthogonal complement of the sieve space.  Second, empirical likelihood solves
an entropy KKT problem on the sample simplex, producing the multiplier
\(\lambda_T\).  Third, as empirical inner products converge to their population
limits, the sample normal-score equation approaches a population \(L^2(P)\)
KKT equation.  Identification comes from uniqueness of the population zero;
Wilks comes from the local \(L^2(P_T)\) curvature of the EL KKT plus the
conditional CLT and LLN for the oracle score.
\end{remark}
```


- [ ] **Step 2: Insert this deterministic-good-event remark after the three-KKT roadmap**

```tex
\begin{remark}[Deterministic geometry on good events]
\label{rem:deterministic-good-event-geometry}
Apart from the oracle CLT, denominator LLN, maximum-score bound, and the
probability of the required good events, the proof is deterministic geometry.
On events where the Gram matrices are stable, the sieve envelopes hold, the
candidate partition is localized, and the EL convex-hull qualification holds,
the required bounds follow from projection contraction, Pythagoras, Cauchy--
Schwarz, support size of the boundary block, and finite-dimensional KKT
calculus. The stochastic lemmas below therefore verify the inputs to this
algebra; they are not the organizing principle of the proof.
\end{remark}
```
## Day 2: Population Bridge and Wilks Narrative

### Task 4: Tighten the Population KKT Bridge

**Files:**
- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Replace the current population bridge assumption with this version**

```tex
\begin{assumption}[Population KKT bridge and identification]
\label{ass:population-kkt-identification}
Let \(\mathcal B\) be a compact neighborhood of \(\beta_0\).  Let
\(\mathcal N_{\Lambda_0}^{P}\) be the \(L^2(P)\) closure of the true
regime-specific nuisance space generated by
\(\ind\{j_{\Lambda_0}(t)=j\}h(W_{t-1})\).  Uniformly over
\(\beta\in\mathcal B\), the empirical projection moments and profiled score
moments converge to their \(L^2(P)\) limits, the empirical Gram matrices are
stable, and the sieve approximation errors vanish at the rates in Assumption
\ref{ass:prim-sieve}.  Let
\[
    u_\beta
    =
    Y-\beta X_{lag}
    -\Pi_P(Y-\beta X_{lag}\mid\mathcal N_{\Lambda_0}^{P}),
    \qquad
    v
    =
    w-\Pi_P(w\mid\mathcal N_{\Lambda_0}^{P}),
\]
where \(\Pi_P(\cdot\mid\mathcal N_{\Lambda_0}^{P})\) denotes \(L^2(P)\)
orthogonal projection.  The population normal-score equation
\[
    \Psi(\beta)=\E\{u_\beta v\}=0
\]
has the unique solution \(\beta=\beta_0\) in \(\mathcal B\).
\end{assumption}
```

- [ ] **Step 2: Replace the current bridge proposition proof with this version**

```tex
\begin{proof}
Lemma \ref{lem:sample-l2pt-profile-kkt} puts both the profiled residual and
the residualized score weight in the empirical orthogonal complement of the
sample sieve space.  Assumption \ref{ass:population-kkt-identification}
transfers these empirical normal equations to their \(L^2(P)\) limits, so the
sample profiled score is uniformly close to \(\Psi(\beta)\) on \(\mathcal B\).
If the sample score at \(\widehat\beta_T\) is \(o_p(1)\), then
\(\Psi(\widehat\beta_T)=o_p(1)\).  Compactness of \(\mathcal B\), continuity
of \(\Psi\), and uniqueness of the population zero imply
\(\widehat\beta_T\to_p\beta_0\).
\end{proof}
```

- [ ] **Step 3: Replace the current non-overclaiming remark with this version**

```tex
\begin{remark}[What the KKT bridge does and does not prove]
The KKT bridge proves identification and consistency once uniform convergence
and population uniqueness are available.  It does not by itself prove Wilks.
For Wilks, the EL KKT must be localized near \(\lambda_T=0\), which requires
the oracle numerator CLT, denominator LLN, and maximum-score control proved
below.  Those stochastic lemmas are therefore not replacements for the KKT
geometry; they are the conditions that make the EL KKT use its local
\(L^2(P_T)\) curvature.
\end{remark}
```

### Task 5: Reframe the Known-Partition Wilks Proof Around Local Curvature

**Files:**
- Modify: `paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex`

- [ ] **Step 1: Add this remark before `thm:wb-known-wilks`**

```tex
\begin{remark}[How the stochastic lemmas enter the KKT geometry]
The nuisance geometry has already reduced the feasible score to the oracle
array \(U_tv_{t,0}\) plus a negligible residual.  The next three lemmas do not
identify \(\beta_0\); they localize the EL KKT.  The numerator CLT gives the
scale of \(P_TZ\), the denominator LLN gives the \(L^2(P_T)\) curvature
\(P_TZ^2\), and the maximum bound guarantees that
\(|\lambda_T|\max_t|Z_t|=o_p(1)\), so the entropy KKT is locally quadratic.
\end{remark}
```

- [ ] **Step 2: In the proof of `thm:wb-known-wilks`, replace the sentence invoking `lem:sol-scalar-el`**

Replace:

```tex
Assumption \ref{ass:prim-convex} supplies the convex-hull condition.  Lemma
\ref{lem:sol-scalar-el} gives
```

with:

```tex
Assumption \ref{ass:prim-convex} supplies the convex-hull condition.  Lemma
\ref{lem:sample-el-local-l2-curvature} localizes the EL multiplier, and Lemma
\ref{lem:sol-scalar-el} gives the local quadratic form
```

### Task 6: Update the Audit Note

**Files:**
- Modify: `proof verify.md`

- [ ] **Step 1: Add this note near the top**

```markdown
## 2026-06-27 KKT/L2 Geometry Verification

Verdict: accepted with the local-curvature refinement. The profile step is an
exact \(L^2(P_T)\) projection KKT problem. The empirical likelihood step is an
entropy KKT problem on the sample simplex, not a projection problem, but near
the root its Hessian is the empirical \(L^2(P_T)\) curvature \(P_TZ^2\). The
population bridge is therefore: sample projection KKT plus EL local curvature
plus \(L^2(P_T)\to L^2(P)\) convergence. Uniqueness of the population
normal-score zero identifies \(\beta_0\); Wilks still requires the oracle CLT,
denominator LLN, and maximum-score localization.
```

## Verification

### Task 7: Compile and Run Targeted Tests

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

- [ ] **Step 2: Compile the workbook twice**

Run:

```powershell
Set-Location 'W:\Research\semipara sieve\paper\archive\drafts'
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_bai_perron_theory_workbook.tex
pdflatex -interaction=nonstopmode -halt-on-error profile_sieve_bai_perron_theory_workbook.tex
```

Expected: both runs finish and the second run has no undefined references for:

```text
lem:sample-l2pt-profile-kkt
lem:sample-el-kkt
lem:sample-el-local-l2-curvature
rem:three-kkt-systems
ass:population-kkt-identification
prop:sample-to-population-kkt
```

- [ ] **Step 3: Review the diff before any commit**

Run:

```powershell
git diff -- docs/superpowers/plans/2026-06-27-l2pt-kkt-identification.md paper/notes/proof\ verify.md paper/archive/drafts/profile_sieve_bai_perron_theory_workbook.tex
```

Expected: the diff only contains the KKT/L2 proof narrative and audit note. Do not commit in this shared dirty worktree unless the user explicitly asks.

## Completion Criteria

- The manuscript states the three KKT systems explicitly.
- Non-stochastic bounds are framed as deterministic geometry on high-probability good events.
- The profile KKT is described as exact \(L^2(P_T)\) projection geometry.
- The EL KKT is described as entropy/simplex KKT with local \(L^2(P_T)\) curvature, not as a global \(L^2(P_T)\) projection.
- The population bridge identifies \(\beta_0\) through uniqueness of \(\Psi(\beta)=0\).
- The Wilks proof explains why CLT, LLN, and max-score control are localization tools for the EL KKT.
- Targeted tests pass and the LaTeX workbook compiles twice.
