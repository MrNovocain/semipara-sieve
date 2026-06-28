# Professor-Friendly 3-Page Technical Report Plan

## Purpose

The report should be a professor-facing viewing memo, not a compressed appendix. Its job is to let a reader understand the paper's mechanism in three pages:

1. what problem the method solves;
2. why the geometric interpretation of breaks is essential;
3. how the main consistency and Wilks result follows once the stochastic regularity event is available.

The report should make the professor think: the method is not just projection notation; the geometry is the decontamination mechanism that separates genuine predictability from omitted structural-break contamination.

## First-Principles Motivation

The paper is about predictive inference when the nuisance mean may have structural breaks:

\[
Y_t=\beta_0X_{t-1}+\mu_{0,t}+U_t,
\qquad
\mu_{0,t}=m_j(W_{t-1}) \text{ on regime } j.
\]

The first-principles problem is that a wrong stable nuisance model does not remove the true nuisance. If a stable residual-maker is used, then generally

\[
M_{\mathrm{st}}\mu_0\neq0.
\]

At the true \(\beta_0\), the stable projected score becomes

\[
S_T^{\mathrm{st}}(\beta_0)
=
T^{-1/2}u'M_{\mathrm{st}}w
+
T^{-1/2}\mu_0'M_{\mathrm{st}}w.
\]

The second term is the false-predictability term:

\[
B_T
=
T^{-1/2}\mu_0'M_{\mathrm{st}}w
=
\sqrt T\langle M_{\mathrm{st}}\mu_0,M_{\mathrm{st}}w\rangle_T.
\]

If the unremoved nuisance component \(M_{\mathrm{st}}\mu_0\) has nonzero angle with the residualized score direction \(M_{\mathrm{st}}w\), this term can dominate the stochastic score. The procedure then reads omitted structural breaks as predictive signal.

This is the central motivation:

\[
\boxed{\text{false predictability is projection drift from an omitted nuisance break.}}
\]

## Why Geometry Is Crucial

The break geometry is not a presentation device. It is the mechanism that removes the false signal.

Each candidate partition defines a nuisance subspace:

\[
\Lambda\mapsto N_{K,\Lambda,T}=\operatorname{col}(Q_{K,\Lambda}).
\]

The residual-maker

\[
M_{K,\Lambda}=I-\Pi_{K,\Lambda,T}
\]

projects away the nuisance component associated with that candidate partition. Break search becomes nearest-subspace selection:

\[
RSS_K(\beta,\Lambda)
=
T\operatorname{dist}_T^2(y-\beta x,N_{K,\Lambda,T})
=
T\|M_{K,\Lambda}(y-\beta x)\|_T^2.
\]

Under the correct break-aware subspace,

\[
M_0\mu_0\approx0,
\]

while under the wrong stable subspace,

\[
M_{\mathrm{st}}\mu_0\not\approx0.
\]

Therefore the geometric interpretation is what lets the paper distinguish genuine predictability from omitted-break contamination.

The projected score is then formed only in the normal space:

\[
\Psi_T(\beta,\Lambda)
=
\langle M_{K,\Lambda}(y-\beta x),M_{K,\Lambda}w\rangle_T.
\]

This is the key paper-level message:

\[
\boxed{\text{predictability inference is done after projecting out a break-adaptive nuisance subspace.}}
\]

## What Professors Need to See

A professor reading a 3-page memo will not want every stochastic bound. They will want to see five things clearly.

### 1. The statistical danger

Start with false predictability. Explain that structural breaks in \(\mu_0\) can survive a stable nuisance projection and enter the score as a deterministic drift. This motivates the method before notation appears.

### 2. The geometric cure

Show that a partition \(\Lambda\) is a subspace selector:

\[
\Lambda\mapsto N_{K,\Lambda,T}.
\]

Then state the RSS identity. This makes break estimation look like nearest-subspace selection, not just date estimation.

### 3. The projected score

State the score as a normal-space inner product:

\[
\Psi_T(\beta,\Lambda)
=
\langle M_{K,\Lambda}(y-\beta x),M_{K,\Lambda}w\rangle_T.
\]

This shows how nuisance removal and predictability testing are tied together.

### 4. The deterministic proof on a good event

Define a compact good event:

\[
\mathcal G_T
=
G_T^{iso}\cap G_T^{approx}\cap G_T^{angle}\cap
G_T^{local}\cap G_T^{select}\cap G_T^{convex}.
\]

Use a table rather than paragraphs:

| Component | Meaning | Used for |
|---|---|---|
| \(G_T^{iso}\) | Gram and inner-product stability | projection and variance control |
| \(G_T^{approx}\) | \(M_0\mu_0\) small | nuisance removal |
| \(G_T^{angle}\) | \(\Gamma\neq0\), variance nonzero | identification and Wilks denominator |
| \(G_T^{local}\) | \(\widehat\Lambda\) replaces \(\Lambda_0\) | estimated-break inference |
| \(G_T^{select}\) | RSS selects local break subspace | break consistency/order selection |
| \(G_T^{convex}\) | EL convex hull and max-score | KKT and quadratic EL expansion |

Then state:

\[
\Pr(\mathcal G_T)\to1.
\]

The report should say explicitly: probability builds the random geometry; once \(\mathcal G_T\) holds, the proof is deterministic.

### 5. The theorem

Give only the theorem-level reduction:

\[
\Psi(\beta,\Lambda_0)=-(\beta-\beta_0)\Gamma,
\qquad
\Gamma=\langle M_0^PX,M_0^Pw\rangle_P\neq0.
\]

Thus consistency follows from the sample-to-population moment bridge.

For EL, give only the KKT identity and quadratic reduction:

\[
\pi_t(\lambda)=\frac{1}{T(1+\lambda g_t)},
\qquad
P_T\left\{\frac{g_t}{1+\lambda g_t}\right\}=0,
\]

\[
-2\log EL(\beta_0,\widehat\Lambda)
=
\frac{T\Psi_T(\beta_0,\widehat\Lambda)^2}{V_T(\widehat\Lambda)}+o_p(1).
\]

Then the good-event bridge gives

\[
\sqrt T\Psi_T(\beta_0,\widehat\Lambda)
=
T^{-1/2}u'M_0w+o_p(1).
\]

If

\[
T^{-1/2}u'M_0w\Rightarrow N(0,V),
\qquad
V_T(\widehat\Lambda)\to_p V,
\]

then

\[
-2\log EL(\beta_0,\widehat\Lambda)\Rightarrow\chi_1^2.
\]

## Suggested 3-Page Layout

### Page 1: Why break geometry is needed

Open with the false predictability mechanism:

\[
S_T^{\mathrm{st}}(\beta_0)
=
T^{-1/2}u'M_{\mathrm{st}}w
+
\sqrt T\langle M_{\mathrm{st}}\mu_0,M_{\mathrm{st}}w\rangle_T.
\]

Then state the cure:

\[
\Lambda\mapsto N_{K,\Lambda,T},
\qquad
RSS_K(\beta,\Lambda)
=
T\operatorname{dist}_T^2(y-\beta x,N_{K,\Lambda,T}).
\]

Close the page with the projected score:

\[
\Psi_T(\beta,\Lambda)
=
\langle M_{K,\Lambda}(y-\beta x),M_{K,\Lambda}w\rangle_T.
\]

### Page 2: Good-event geometry

Define \(\mathcal G_T\) and include the compact table. Then give population identification:

\[
\Psi(\beta,\Lambda_0)=-(\beta-\beta_0)\Gamma,
\qquad
\Gamma\neq0.
\]

State the deterministic implication:

\[
\sup_\beta|\Psi_T(\beta,\widehat\Lambda)-\Psi(\beta,\Lambda_0)|=o_p(1)
\Rightarrow
\widehat\beta\to_p\beta_0.
\]

### Page 3: EL/Wilks and what remains stochastic

State the EL KKT form and quadratic expansion. Then give the oracle score reduction and the Wilks theorem:

\[
-2\log EL(\beta_0,\widehat\Lambda)\Rightarrow\chi_1^2.
\]

End with one honest paragraph listing what the full proof supplies: Gram stability, sieve approximation, RSS localization, break replacement, score CLT, variance LLN, max-score, and convex-hull control.

Optional final lines for efficiency:

\[
w^*=X/\sigma^2(S,W),
\qquad
M_0^Pw^*=\{X-E(X\mid S,W)\}/\sigma^2(S,W).
\]

In the homoskedastic case, \(w=X\).

## What to Exclude

Exclude anything that does not help a professor evaluate the paper's core mechanism in three pages:

- full sieve approximation proof;
- primitive empirical process details;
- long RSS localization derivation;
- detailed break-date rate proof;
- toy examples unless there is extra space;
- full efficiency proof;
- repeated KKT algebra beyond the multiplier formula and quadratic expansion;
- notation that appears only once.

## Tone and Style

The report should be mathematically precise but not encyclopedic. Every displayed equation should answer one of three questions:

1. What is the contamination problem?
2. How does break-adaptive geometry remove it?
3. How does the resulting score deliver Wilks inference?

Use theorem statements sparingly. Prefer a proof-map style with exact equations. Professors should be able to identify the contribution in the first half-page and the validity path by the end of page two.

## One-Sentence Thesis

The report should make this thesis unavoidable:

\[
\boxed{\text{Break-adaptive projection is the decontamination step that turns false predictability from structural breaks into valid EL inference for }\beta.}
\]
