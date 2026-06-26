# Proof Verification Report - Math Only

Date: 2026-06-24

## 2026-06-27 KKT/L2 Geometry Verification

Verdict: accepted with the local-curvature refinement. The profile step is an
exact \(L^2(P_T)\) projection KKT problem. The empirical likelihood step is an
entropy KKT problem on the sample simplex, not a projection problem, but near
the root its Hessian is the empirical \(L^2(P_T)\) curvature \(P_TZ^2\). The
population bridge is therefore: sample projection KKT plus EL local curvature
plus \(L^2(P_T)\to L^2(P)\) convergence. Uniqueness of the population
normal-score zero identifies \(\beta_0\); Wilks still requires the oracle CLT,
denominator LLN, and maximum-score localization. All non-stochastic bounds
should be read as deterministic projection/KKT algebra on high-probability good
events; stochastic lemmas verify those events and the oracle score limits.

Files checked:

- writing_samples/profile_sieve_geometric_paper.tex
- writing_samples/profile_sieve_geometric_supplement.tex

Purpose: check the mathematical logic line by line for the assumptions, conditions, definitions, lemmas, propositions, theorems, corollaries, and supplement proofs.

## Verdict

I checked the proof logic line by line. I found no fatal algebraic error in the current main paper or supplement, conditional on the stated high-level assumptions.

The important qualification is that the paper is still a high-level asymptotic proof, not a primitive proof from only break spacing, smoothness, persistence, and the RSS search criterion. The strongest unproved inputs are:

- the persistent oracle mixed-normal score limit;
- feasible empirical-likelihood convex-hull conditions;
- the projection-bias condition T^{-1/2} r_0' M_{K,Lambda_0} w = o_p(1);
- fixed-order estimated-score equivalence;
- consistency of the unknown number of breaks;
- the stable no-break EL quadratic expansion when an omitted fixed break can make B_T large;
- local-alternative replacement relative to the shifted score array.

Those are not internal contradictions. They are assumptions. The theorems are correct as conditional implications from those assumptions.

## Small source edits made during this audit

I made two small mathematical-text cleanups in writing_samples/profile_sieve_geometric_paper.tex:

1. In Proposition 1, I changed the punctuation after the displayed definition of B_T from a comma to a period.
2. In Theorem 4, I rewrote the local-alternative convex-hull sentence so it explicitly refers to the feasible true-partition score array, not a generic convex-hull condition.

No mathematical statement was weakened or silently changed by these two edits.

## Main paper: line-by-line math audit

### Lines 169-178: norm conventions

Status: correct.

The paper defines ||a||_T^2 = T^{-1} sum_t a_t^2 and ||a||_infty = max_t |a_t|. It also specifies that growing-dimensional O_p and o_p statements use Euclidean norm for vectors and spectral norm for matrices. This is enough to interpret all later sieve-rate and projection-rate statements.

### Lines 180-197: model objects and true partition

Status: correct.

The construction of Y, X_lag, w, Lambda_0, I_j(Lambda), j_Lambda(t), mu_t, and mu is internally consistent. The model deliberately excludes breaks in beta_0 and in the predictor equation. That restriction is used later: nuisance instability is the only structural instability being adapted to.

### Lines 199-214: Assumption 1, finite nuisance breaks and smooth regimes

Status: correct, but high-level in detectability.

Checks:

- q_0 fixed: needed because block dimension K_q = (q_0+1)K is later treated with fixed q_0.
- minimum spacing at epsilon T: needed for block Gram stability and break localization.
- all m_j in the same smoothness class H^s: needed for a common sieve approximation rate a_K.
- adjacent break detectability Delta_T > 0: needed for break search consistency/localization.
- sufficient condition T Delta_T^2/(K log T) -> infinity: plausible for fixed-break detectability relative to sieve complexity.

This assumption does not itself prove estimated-score equivalence. The paper correctly separates those issues later.

### Lines 216-226: Assumption 2, identification after block residualization

Status: correct.

This requires T^{-1} w' M_{K,Lambda_0} w -> Q_w with Q_w bounded away from zero. That is exactly the condition that the predictive score direction survives residualization against the true regime-specific nuisance sieve. Without it, the EL denominator could degenerate.

### Lines 228-235: stable no-break score

Status: correct.

S_T^st(beta_0) = T^{-1/2}(Y - beta_0 X_lag)' M_{K,0} w is the intended stable-nuisance score. It intentionally uses the no-break projection M_{K,0}, so it is the right object for the omitted-break failure comparison.

### Lines 237-258: Proposition 1, stable-score decomposition

Status: correct.

Line-by-line proof check:

- Under the model at beta_0, Y - beta_0 X_lag = mu + U.
- Multiplying by T^{-1/2} M_{K,0} w gives T^{-1/2} U' M_{K,0} w + T^{-1/2} mu' M_{K,0} w.
- Therefore S_T^st(beta_0) = A_T^st + B_T exactly.
- No sieve approximation is used in the identity because B_T uses the exact nuisance vector mu.

The statement that B_T is asymptotically negligible when q_0=0 relies on the later projection-bias condition, not on the algebra of the proposition. That is fine because it is explanatory, not the core identity.

### Lines 260-272: Definition 1, break-induced spurious predictivity

Status: correct.

The definition identifies spurious predictivity with B_T not being o_p(1). This is the right directional condition: only the component of the omitted nuisance break aligned with M_{K,0} w matters.

### Lines 274-280: V_T^st definition

Status: correct.

V_T^st is defined as the feasible no-break score second moment. This is the denominator used in Theorem 1. The definition is normalized by T, which is consistent with the scalar EL expansion.

### Lines 282-310: Theorem 1, omitted-break failure

Status: correct conditional theorem.

Line-by-line proof check:

- The theorem assumes the stable EL quadratic expansion ell_T^st(beta_0) = {S_T^st(beta_0)}^2 / V_T^st + o_p(1).
- It also assumes V_T^st is bounded away from zero in probability.
- Define a_T = A_T^st / sqrt(V_T^st) and b_T = B_T / sqrt(V_T^st).
- By Proposition 1, S_T^st / sqrt(V_T^st) = a_T + b_T.
- If a_T = O_p(1) and |b_T| -> infinity in probability, then |a_T+b_T| -> infinity in probability, so ell_T^st exceeds any fixed chi-square critical value with probability tending to one.
- If (a_T,b_T) converges stably to (Z,B), continuous mapping gives ell_T^st => (Z+B)^2.
- Conditional on B, Z+B is normal with mean B and variance 1, so (Z+B)^2 is chi-square with 1 degree of freedom and noncentrality B^2.

No algebraic gap found. The theorem is not a primitive fixed-break divergence theorem because the quadratic expansion is assumed.

### Lines 312-314: omitted-break caveat remark

Status: correct and necessary.

The remark correctly says that if B_T can be order sqrt(T), the usual null-style scalar EL expansion need not automatically hold. This avoids overclaiming.

### Lines 319-357: block sieve matrix and residual maker

Status: correct.

Checks:

- Q_{K,Lambda} is the block design with one sieve block per regime.
- M_{K,Lambda} = I - Q(Q'Q)^dagger Q' uses the Moore-Penrose inverse.
- Because the Moore-Penrose inverse is used, M is defined for every candidate partition, including singular candidate Gram matrices.
- When Q'Q is nonsingular, the definition reduces to the usual inverse.

This fixes the candidate-partition invertibility issue.

### Lines 359-378: profiled residual, residualized weight, scalar score

Status: correct.

The profiled residual M(Y-beta X_lag), residualized weight Mw, and score Zhat_t = uhat_t w_t^c are consistent. The score uses the same projection for the residual and the weight, which is essential for the projection identities.

### Lines 381-398: Lemma 1, exact block projection identities

Status: correct.

Proof logic:

- Pi = Q(Q'Q)^dagger Q' is the orthogonal projector onto col(Q).
- Hence Pi is symmetric and idempotent.
- M = I - Pi is also symmetric and idempotent.
- Q'M = 0 because M projects onto the orthogonal complement of col(Q).
- Therefore any block-sieve nuisance component Qc satisfies (Qc)' M w = c' Q' M w = 0.

This is finite-sample algebra and does not require asymptotics.

### Lines 400-420: empirical likelihood definition

Status: correct.

The scalar EL constraints are pi_t >= 0, sum pi_t = 1, and sum pi_t Zhat_t = 0. The log empirical likelihood ratio is written in the standard scalar multiplier form with lambda solving the sample moment equation.

### Lines 429-496: partition learning definitions

Status: correct as definitions.

Checks:

- Segment cost C_K(s,e;beta) minimizes squared residuals over a sieve coefficient on the interval.
- RSS_K(beta,Lambda) sums segment costs over the partition.
- Lambda_hat_q(beta) is the global minimizer over spaced q-break partitions.
- q_hat(beta) minimizes penalized RSS over 0 <= q <= qbar.
- ell_T^BA(beta) evaluates EL at the selected partition.

These are definitions. The paper does not claim here that the computational heuristic proves consistency; it correctly says the global minimizer is the proof object.

### Lines 501-529: Assumption 3, block-sieve rates and projection bias

Status: correct high-level bias condition.

Checks:

- Each regime nuisance m_j is approximated by the sieve plus r_{j,K}.
- The paper requires both ||r_j||_T = O_p(a_K) and ||r_j||_infty = O_p(a_K).
- The key bias condition is directional: T^{-1/2} r_0' M_{K,Lambda_0} w = o_p(1).
- This condition is weaker than sqrt(T) a_K -> 0 because it only controls the score direction, not the full nuisance approximation norm.
- The true-partition block Gram eigenvalue bound is enough for the known-partition algebra.
- The rate conditions a_K(1+zeta_K)->0 and a_K(1+zeta_K)^2=o(sqrt(T)) are the right type for denominator and maximum score replacement.
- The dimension restrictions K_q/sqrt(T)->0 and zeta_K sqrt(K_q/T)->0 are consistent with controlling projection effects under fixed q_0.

No internal inconsistency found.

### Lines 531-533: projection-bias remark

Status: correct.

The remark correctly states that the new condition is not the old strong undersmoothing condition sqrt(T) a_K -> 0. The product-rate intuition sqrt(T) a_{m,K} a_{w,K} -> 0 is mathematically plausible because the score bias involves the nuisance approximation error projected against a residualized/approximable weight direction.

### Lines 535-576: Assumption 4, known-partition oracle score

Status: correct high-level oracle condition.

Checks:

- v_0 = M_{K,Lambda_0} w is the residualized weight under the true block partition.
- A_T^(q) = T^{-1/2} sum U_t v_{t,0} is the oracle numerator.
- V_T^(q) = T^{-1} sum U_t^2 v_{t,0}^2 is the oracle denominator.
- Stable convergence A_T^(q) -> eta_q Z and V_T^(q) -> eta_q^2 gives the mixed-normal numerator and random-scale denominator needed for Wilks.
- eta_q bounded away from zero prevents denominator degeneracy.
- max_t |U_t v_{t,0}| = o_p(sqrt(T)) is the scalar EL maximum condition.
- The two projected-noise replacement conditions compare M U to U in denominator and maximum; these are required because the feasible residual is M(U+r), not U+r.
- The convex-hull condition is imposed on the feasible array Zhat_t(beta_0,Lambda_0), which is the array used by EL. This is the correct condition.

No algebraic error found. This remains an assumption, not a primitive persistence proof.

### Lines 578-584: Theorem 2, known-partition Wilks theorem

Status: correct conditional theorem.

Dependency check:

- Assumption 1 supplies the true partition setup and fixed q_0.
- Assumption 2 supplies nondegenerate residualized weight direction.
- Assumption 3 supplies projection-bias and sieve replacement control.
- Assumption 4 supplies the oracle mixed-normal limit, denominator consistency, max condition, projected-noise replacement, and feasible convex hull.
- Supplement Lemmas S2 and S3 then imply ell_T(beta_0,Lambda_0) => chi-square_1.

No missing dependency found.

### Lines 588-620: Assumption 5, fixed-order partition localization

Status: correct and explicitly limited.

The condition max_j |ktilde_j-k_{j,0}| = O_p(r_T), r_T=o(sqrt(T)), implies only o_p(sqrt(T)) directly misclassified boundary observations. The paper correctly says this alone does not prove score equivalence, because moving estimated breaks can perturb fitted block coefficients outside the misclassified set.

### Lines 622-638: Assumption 6, high-level fixed-order estimated-score equivalence

Status: correct high-level transfer assumption.

Checks:

- Numerator difference between estimated and true partition scores is o_p(1) after T^{-1/2} summation.
- Denominator difference is o_p(1) after T^{-1} second-moment normalization.
- Estimated-partition maximum score is o_p(sqrt(T)).
- Zero lies in the interior of the scalar convex hull for the estimated-partition feasible array.

This is exactly what is needed for EL likelihood transfer. It is not proved by Assumption 5, and the paper correctly says so.

### Lines 640-647: Theorem 3, fixed-order estimated-partition Wilks theorem

Status: correct conditional theorem.

Given Theorem 2 and Assumption 6, the estimated-partition EL statistic differs from the true-partition statistic by o_p(1). Therefore it has the same chi-square_1 limit. The theorem correctly states that it is high-level.

### Lines 649-655: Corollary 1, unknown number of breaks

Status: correct.

If P{q_hat(beta_0)=q_0}->1, then the unknown-order selected partition equals the fixed-order object with probability tending to one. On that event, Theorem 3 applies. Therefore ell_T^BA(beta_0) => chi-square_1.

### Lines 657-702: Theorem 4, local predictive alternatives

Status: correct conditional local-power theorem.

Line-by-line proof check:

- beta_T = beta_0 + d_T and the statistic is evaluated at beta_0.
- Under beta_T, the null-imposed residual contains U + mu + d_T X_lag.
- After true-partition projection, the predictive drift contribution to the score is g_{t,T} = d_T (M_{K,Lambda_0} X_lag)_t v_{t,0}.
- The shifted oracle score is Z_{t,T}^{loc} = U_t v_{t,0} + g_{t,T}. This is the correct comparison array under the local alternative.
- The drift condition T^{-1/2} sum g_{t,T} -> eta_q delta_h creates the noncentral mean shift.
- The condition T^{-1} sum g_{t,T}^2 = o_p(1) keeps the denominator unchanged asymptotically.
- The max condition max_t |g_{t,T}|=o_p(sqrt(T)) is needed for scalar EL.
- The theorem assumes numerator, denominator, and maximum replacement relative to the shifted oracle array, not relative to U_t v_{t,0} alone. This is correct.
- The feasible convex hull is imposed on Zhat_t(beta_0,Lambda_0) under the local law beta_T.
- Estimated-score equivalence is also assumed under beta_T.
- With numerator eta_q(Z+delta_h) and denominator eta_q^2, the quadratic EL limit is (Z+delta_h)^2 = chi-square_1(delta_h^2).
- The nuisance-break drift B_T does not appear because the true regime-specific projection removes the block nuisance component up to projection-bias remainder.

No local-noncentrality algebra error found.

### Lines 705-717: proof-dependency map

Status: correct.

The map accurately separates what the paper proves by projection algebra and scalar EL from what it assumes at high level. In particular, it correctly says estimated-partition inference depends on high-level estimated-score equivalence, not localization alone.

## Supplement: proof audit

### Lines 104-126: Lemma S1, block projection identities

Status: correct.

Line-by-line proof check:

- Define Pi = Q(Q'Q)^dagger Q'.
- Moore-Penrose projection theory gives Pi as the orthogonal projector onto col(Q).
- Therefore Pi' = Pi and Pi^2 = Pi.
- M = I - Pi is symmetric and idempotent.
- Since M projects onto col(Q) perpendicular, Q'M = 0.
- For any conformable c, (Qc)'Mw = c'Q'Mw = 0.

No invertibility assumption is needed.

### Lines 132-146: proof of Proposition 1

Status: correct.

The proof uses Y - beta_0 X_lag = mu + U under the null-imposed evaluation. Multiplication by T^{-1/2} M_{K,0} w gives the exact split into the stochastic score and omitted-break component. Because B_T uses exact mu, no sieve approximation remainder is introduced.

### Lines 148-163: proof of Theorem 1

Status: correct.

The assumed scalar EL expansion gives ell_T^st = (a_T+b_T)^2 + o_p(1). Divergence follows from |a_T+b_T| >= |b_T|-|a_T|. Stable convergence plus continuous mapping gives (Z+B)^2. Conditional noncentral chi-square follows because Z is conditionally standard normal.

### Lines 165-167: omitted-break divergence caveat

Status: correct.

This is the right caveat: if B_T is large under a fixed omitted break, scalar null-style EL expansion is not automatic. The theorem avoids claiming more than it assumes.

### Lines 171-184: true-partition residual decomposition

Status: correct.

Under the true partition, the nuisance vector decomposes as Q c_0 + r_0. Applying M_{K,Lambda_0} kills Q c_0 exactly and leaves M U + M r_0.

### Lines 186-199: Lemma S2, known-partition replacement statement

Status: correct.

The lemma states exactly the three replacements needed for scalar EL:

- numerator replacement;
- denominator replacement;
- maximum score bound.

### Lines 201-223: Lemma S2 numerator proof

Status: correct.

Line-by-line proof check:

- uhat_0(beta_0) = M(U+r_0) because M Q c_0 = 0.
- v_0 = M w.
- Since M is symmetric and idempotent, uhat_0' v_0 = (U+r_0)' v_0.
- The U part is the oracle numerator.
- The r_0 part is T^{-1/2} r_0' M w, which is o_p(1) by Assumption 3.

No Cauchy undersmoothing shortcut is used. This is the correct projection-bias repair.

### Lines 225-246: Lemma S2 denominator and maximum preliminary bounds

Status: correct, though compact.

Detailed check:

- v_0 = M w = w - Q(Q'Q)^dagger Q'w.
- w is bounded and row norms of Q are at most zeta_K.
- The true-partition Gram eigenvalues are bounded, so the fitted part has sup norm O_p(zeta_K), giving ||v_0||_infty = O_p(1+zeta_K).
- T^{-1}Q'r_0 = O_p(a_K) follows from ||r_0||_T=O_p(a_K) and bounded Gram: ||T^{-1}Q'r_0|| <= ||T^{-1}Q'Q||^{1/2} ||r_0||_T.
- e_0 = M r_0 satisfies ||e_0||_T <= ||r_0||_T = O_p(a_K), since M is an orthogonal projection.
- The fitted correction in e_0 has sup norm O_p(zeta_K a_K), while r_0 has sup norm O_p(a_K), giving ||e_0||_infty = O_p(a_K(1+zeta_K)).
- Therefore ||e_0 odot v_0||_T = O_p(a_K(1+zeta_K)) = o_p(1).
- Also ||e_0 odot v_0||_infty = O_p(a_K(1+zeta_K)^2) = o_p(sqrt(T)).

This proof is mathematically valid. It could be expanded in the paper for readability, but no error found.

### Lines 247-269: Lemma S2 denominator and maximum replacement

Status: correct.

Detailed check:

- Zhat_t = Utilde_{0,t} v_{t,0} + e_{0,t} v_{t,0}.
- Assumption 4 gives T^{-1} sum [(Utilde v)^2 - (U v)^2] = o_p(1).
- Since ||Utilde v||_T = O_p(1) and ||e_0 odot v_0||_T = o_p(1), the cross term 2 T^{-1} sum (Utilde v)(e v) is o_p(1).
- The square term T^{-1} sum (e v)^2 is o_p(1).
- Therefore the feasible denominator equals the oracle denominator plus o_p(1).
- For the maximum, |Zhat_t| <= |U_t v_t| + |(Utilde_t-U_t)v_t| + |e_t v_t|. Each term is o_p(sqrt(T)) by Assumption 4 and the rate above.

No denominator replacement error found.

### Lines 271-307: Lemma S3, scalar EL expansion

Status: correct.

Detailed check:

- The feasible convex hull gives existence of the scalar EL multiplier.
- Vhat_T^Z bounded away from zero implies Q_T^Z = sum Z_t^2 is order T.
- T^{-1/2} sum Z_t = O_p(1) implies S_T^Z = O_p(sqrt(T)).
- Monotonicity of the multiplier equation bounds lambda by O_p(T^{-1/2}).
- max_t |Z_t| = o_p(sqrt(T)) implies max_t |lambda Z_t| = o_p(1).
- Taylor expansion is therefore valid uniformly.
- The multiplier equation gives lambda = S_T^Z / Q_T^Z + o_p(T^{-1/2}).
- Substitution into the log EL expansion gives (T^{-1/2} sum Z_t)^2 / (T^{-1} sum Z_t^2) + o_p(1).

This is the standard scalar EL argument and is valid under the stated assumptions.

### Lines 309-319: proof of Theorem 2

Status: correct.

Proof chain:

- Lemma S2 reduces feasible true-partition scores to U_t v_{t,0}.
- Assumption 4 gives numerator stable convergence and denominator convergence to eta_q^2.
- Lemma S3 gives the quadratic EL expansion.
- The ratio [eta_q Z]^2 / eta_q^2 converges to Z^2, hence chi-square_1.

The random scale eta_q cancels. This is the correct Wilks argument under stable convergence.

### Lines 321-349: Lemma S4, fixed-order estimated-partition likelihood transfer

Status: correct conditional transfer proof.

Detailed check:

- Define S_T(Lambda) and V_T(Lambda) for feasible scores.
- Lemma S2 and Assumption 4 give S_T(Lambda_0)=O_p(1), V_T(Lambda_0) bounded away from zero, and max condition for Lambda_0.
- Assumption 6 gives S_T(Lambdatilde)-S_T(Lambda_0)=o_p(1), V_T(Lambdatilde)-V_T(Lambda_0)=o_p(1), max condition, and convex hull for Lambdatilde.
- Lemma S3 applies to both arrays.
- The two quadratic ratios differ by o_p(1) because numerators differ by o_p(1) and denominators differ by o_p(1) while bounded away from zero.

This proves likelihood transfer. It does not prove Assumption 6 from localization, and the paper correctly does not claim that.

### Lines 351-358: proof of Theorem 3 and Corollary 1

Status: correct.

Theorem 3 follows from Lemma S4 plus Theorem 2. The corollary follows because on the event q_hat(beta_0)=q_0, the unknown-order selected partition coincides with the fixed-order selected partition with probability tending to one.

### Lines 360-387: proof of Theorem 4, local alternatives

Status: correct conditional local-power proof.

Detailed check:

- Under beta_T = beta_0+d_T, Y-beta_0 X_lag = mu+U+d_T X_lag.
- True-partition projection removes the block-sieve approximation to mu up to the assumed local replacement error.
- The numerator becomes T^{-1/2} sum U_t v_t + T^{-1/2} sum g_t + o_p(1).
- The first term converges stably to eta_q Z; the second converges in probability to eta_q delta_h.
- The denominator of the shifted array is V_T^(q) + 2T^{-1}sum Uv g + T^{-1}sum g^2.
- The last term is o_p(1) by assumption.
- The cross term is o_p(1) by Cauchy-Schwarz: |T^{-1}sum Uv g| <= (T^{-1}sum U^2v^2)^{1/2}(T^{-1}sum g^2)^{1/2} = O_p(1)o_p(1).
- Therefore the denominator remains eta_q^2+o_p(1).
- The max and convex-hull assumptions let Lemma S3 apply.
- The known-partition local limit is (Z+delta_h)^2.
- Lemma S4, assumed under the local law, transfers the result to the estimated fixed-order partition.
- Correct order selection transfers it to the unknown-order BA statistic.

No local alternative proof error found.

## Dependency map checked manually

### Proposition 1

Needs: model at beta_0.

Conclusion: exact decomposition S_T^st = A_T^st + B_T.

Status: proved exactly.

### Theorem 1

Needs: Proposition 1, assumed stable EL quadratic expansion, V_T^st bounded away from zero, either divergence of b_T or joint stable convergence of (a_T,b_T).

Conclusion: divergence or noncentral/mixed noncentral chi-square.

Status: correct conditional theorem.

### Lemma 1 and Lemma S1

Needs: Moore-Penrose projection algebra.

Conclusion: Q'M=0, M'=M, M^2=M, and block-sieve nuisance cancellation.

Status: exact finite-sample result.

### Lemma S2

Needs: Assumption 3 and Assumption 4.

Conclusion: feasible true-partition score has the same numerator, denominator, and max behavior as U_t v_{t,0}.

Status: correct.

### Lemma S3

Needs: scalar convex hull, bounded normalized numerator, denominator bounded away from zero, max score o_p(sqrt(T)).

Conclusion: scalar EL quadratic expansion.

Status: correct.

### Theorem 2

Needs: Assumptions 1-4 and Lemmas S2-S3.

Conclusion: known-partition Wilks chi-square_1.

Status: correct conditional theorem.

### Assumption 5

Needs/proves: does not prove theorem directly; only states localization.

Status: correct and honestly limited.

### Assumption 6

Needs/proves: directly assumes fixed-order estimated-score equivalence.

Status: exactly the right high-level assumption for transfer.

### Theorem 3

Needs: Theorem 2 and Assumption 6.

Conclusion: fixed-order estimated-partition Wilks chi-square_1.

Status: correct conditional theorem.

### Corollary 1

Needs: Theorem 3 and P{q_hat=q_0}->1.

Conclusion: unknown-order BA statistic has chi-square_1 limit.

Status: correct.

### Theorem 4

Needs: local drift, local L2 and max conditions, shifted-array feasible replacement, local feasible convex hull, local estimated-score equivalence, and order consistency for the BA version.

Conclusion: local noncentral chi-square_1(delta_h^2).

Status: correct conditional theorem.

## Things that are high-level and not primitive proofs

These are the real mathematical limitations of the current paper:

1. Assumption 4 is not derived from primitive persistence conditions for X_t and U_t. It is an oracle score assumption.
2. The feasible convex-hull conditions are assumed, not proved.
3. Assumption 3 directly assumes projection-bias negligibility. The paper explains why this is weaker than sqrt(T) a_K -> 0, but does not prove a primitive product-rate theorem.
4. Assumption 6 is the key estimated-break inference assumption. It is not implied by r_T=o(sqrt(T)) localization alone.
5. Corollary 1 assumes order consistency of q_hat. It does not prove the penalty condition that guarantees it.
6. The omitted-break divergence result is conditional on the stable no-break EL quadratic expansion. The paper correctly warns that a fixed omitted break can invalidate a null-style expansion.
7. The local-power theorem assumes local replacement relative to U_t v_{t,0}+g_{t,T}. That is the right condition, but it is still assumed.

## Final mathematical conclusion

I found no fatal mistake in the projection algebra, feasible-oracle replacement, scalar EL expansion, Wilks theorem, estimated-partition likelihood transfer, omitted-break conditional theorem, or local noncentral chi-square calculation.

The current paper is mathematically correct as a conditional high-level proof. It is not yet a complete primitive proof that the proposed profile-sieve partition estimator satisfies all assumptions from only primitive smoothness, break, persistence, and penalty conditions.