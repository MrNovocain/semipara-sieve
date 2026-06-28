MC 这部分应该比 empirical study 更重要。**真实数据负责展示现象，MC 负责证明我们的 story 是真的。**

我们的 MC 不应该是普通 “size / power table”。它应该围绕最终 story 设计：

[
\boxed{
\text{profile-sieve EL preserves persistence-robust Wilks, while efficiency depends on score direction.}
}
]

也就是说，MC 要验证三件事：

1. **Wilks calibration**：加入未知 (m(W)) 后，bounded profile-sieve EL 仍然在不同 persistence regime 下有正确 size。
2. **Oracle equivalence**：feasible score 和 oracle score 一阶等价，不是我们纸上写出来而已。
3. **Robustness–efficiency tradeoff**：bounded saturated score 更稳，但 regular efficient score / raw-(X) score 在 stationary case 可能更有 power，却在 persistent/endogenous case 下更敏感。

---

# 1. Monte Carlo 的核心定位

MC 的一句话目标：

> We use simulations to examine whether the proposed profile-sieve EL statistic preserves the persistence-robust Wilks phenomenon after estimating the nonparametric nuisance, and to quantify the efficiency–robustness tradeoff between bounded saturated scores and semiparametric efficient scores.

不要写成：

> We compare forecasting methods.

我们不是做 forecasting horse race。我们做的是 inference。

---

# 2. 基础 DGP

推荐主 DGP 用 predictive regression 标准形式：

[
Y_t=m(W_{t-1})+\beta X_{t-1}+U_t,
]

[
X_t=\rho_T X_{t-1}+V_t,
]

[
W_t=aW_{t-1}+\eta_t.
]

令

[
\begin{pmatrix}
U_t\
V_t\
\eta_t
\end{pmatrix}
\sim
N\left(
0,
\begin{pmatrix}
1 & \kappa & 0\
\kappa & 1 & \xi\
0 & \xi & 1
\end{pmatrix}
\right).
]

这里：

* (\rho_T) 控制 predictor persistence；
* (\kappa) 控制 predictive-regression endogeneity / Stambaugh-type correlation；
* (\xi) 控制 (X_t) 和 (W_t) 的相关性；
* (m(W)) 是未知非参数 nuisance。

为了保证 conditional mean restriction，

[
E(U_t\mid \mathcal H_{t-1})=0
]

仍然成立，因为 (U_t) 只和 contemporaneous (V_t) 相关，不和过去信息相关。

---

# 3. Persistence regimes 必须覆盖这些

至少用四类：

[
\rho_T=0.5
\qquad
\text{stationary low persistence},
]

[
\rho_T=0.95
\qquad
\text{stationary high persistence},
]

[
\rho_T=1-\frac{5}{T}
\qquad
\text{local-to-unity},
]

[
\rho_T=1
\qquad
\text{unit root}.
]

如果想贴近 Cai/Hong lineage，可以加：

[
\rho_T=1+\frac{1}{T}
]

作为 mildly explosive / mildly above unity 的 robustness check。

但正文不要放太多。主表放四类就够，mildly explosive 放 appendix。

---

# 4. 非参数 nuisance (m(W)) 怎么设

需要至少三个 DGP：

## DGP A: no nuisance benchmark

[
m(W)=0.
]

作用：确认我们在 (m=0) 时不比原 Cai/Hong oracle setup 差。

## DGP B: smooth nonlinear nuisance

[
m(W)=0.5\sin(W)+0.3(W^2-1).
]

作用：测试 profile sieve 是否真的处理 nonlinear nuisance。

## DGP C: stronger nonlinear nuisance

[
m(W)=0.5\sin(2W)+0.4W\exp(-W^2/2).
]

作用：测试更复杂但仍 smooth 的 (m)。

如果只选一个主 nuisance，我建议用：

[
\boxed{
m(W)=0.5\sin(W)+0.3(W^2-1).
}
]

因为它既有非线性，又不会太怪。

---

# 5. Methods 要比较哪些

至少比较五类。

## Method 1: Oracle bounded EL

知道真实 (m(W))，使用

[
Z_t^{\mathrm{or}}(\beta_0)
==========================

U_t\widetilde w_t.
]

这是 oracle benchmark。

## Method 2: Proposed profile-sieve bounded EL

使用

[
\widehat Z_t(\beta)
===================

\widehat u_t(\beta)w_t^c,
]

其中

[
\widehat{\bm u}(\beta)=M_K(\bm Y-\beta\bm X),
]

[
\bm w^c=M_K\bm w.
]

这是主方法。

## Method 3: intercept-only bounded EL

忽略 (m(W))，只控制截距。

作用：展示如果不处理 nonparametric nuisance，size / coverage 会坏。

## Method 4: efficient-score benchmark

同方差时使用

[
g_t^\star
=========

X_{t-1}-E(X_{t-1}\mid W_{t-1}),
]

sample analog：

[
\widehat g_t^\star=(M_KX)_t.
]

EL score：

[
\widehat Z_t^{\mathrm{eff}}(\beta)
==================================

\widehat u_t(\beta)(M_KX)_t.
]

这是 conditional-mean semiparametric efficient benchmark。我们上传的 note 中已经推导了更一般的 (qV) efficient score；同方差下它退化为 residualized (X)。

## Method 5: conventional OLS / HAC / t-test

作为传统 benchmark。

不要让它成为主角，只用来展示普通方法对 persistence 和 nuisance 更敏感。

---

# 6. Weights 怎么设

主 bounded weight 可以用：

[
w_b(x)=\tanh(x/b).
]

取几个 (b)：

[
b\in{0.5,1,2,4}.
]

解释：

* 小 (b)：更快 saturation，更 robust，但效率可能低；
* 大 (b)：更接近 linear (X)，效率可能高，但 persistence robustness 可能变弱。

还可以加入：

[
w(x)=\frac{x}{\sqrt{1+x^2}}
]

作为 fixed bounded saturated alternative。

但正文主要用 (w_b(x)=\tanh(x/b))，因为它非常直观。

---

# 7. Sieve setup

使用 B-spline 或 polynomial sieve。为了 MC 简洁：

[
K\in{4,6,8}
]

或者

[
K=\lfloor cT^{1/5}\rfloor
]

并做 sensitivity。

我建议正文主表固定：

[
K=6
]

或

[
K=\lfloor T^{1/5}\rfloor+2.
]

然后 appendix 报告 (K) sensitivity。

不要让 MC 变成 tuning-parameter search。主要 choices 必须预先固定。

---

# 8. Sample sizes and replications

推荐：

[
T\in{100,250,500}.
]

Replications：

[
R=2000
]

够用。若计算不贵，可以

[
R=5000
]

用于 size table。

Nominal levels：

[
5%,\quad 10%.
]

主表重点放 5%。

---

# 9. Monte Carlo 要报告哪些指标

## A. Size

在

[
H_0:\beta=0
]

下报告 rejection frequency：

[
P{\ell_T(\beta_0)>\chi^2_{1,0.95}}.
]

这是最核心。

需要按：

[
\rho_T,\quad T,\quad \kappa,\quad m(W)
]

分组。

推荐主表：

| (\rho_T) | Oracle bounded EL | Proposed profile EL | Intercept-only EL | Efficient-score EL | OLS/HAC |
| -------- | ----------------: | ------------------: | ----------------: | -----------------: | ------: |

分别对 (T=100,250,500) 或者分三张表。

重点看：

[
\boxed{
\text{Proposed profile EL size close to oracle bounded EL.}
}
]

---

## B. Coverage and interval length

报告 (95%) confidence interval coverage 和 average length。

特别比较：

[
\text{proposed bounded profile EL}
\quad\text{vs}\quad
\text{efficient-score profile EL}.
]

预期故事：

* stationary 下 efficient-score interval 可能更短；
* high persistence / endogeneity 下 bounded score coverage 更稳定；
* proposed profile EL 接近 oracle bounded EL。

---

## C. Power

不要只用 fixed alternative。应该用 local alternative。

对于 stationary：

[
\beta_T=\frac{h}{\sqrt T}.
]

对于 unit root / local-to-unity：

[
\beta_T=\frac{h}{T}
]

或者更统一地写成 information scale：

[
\beta_T=\frac{h}{r_T}.
]

其中

[
r_T\asymp\sqrt T
]

for stationary，

[
r_T\asymp T
]

for unit root。

但为了展示结果，MC 表里可以直接给几组 (\beta)：

[
\beta\in{0,0.05,0.10}
]

for stationary，

unit-root 下用更小：

[
\beta\in{0,0.01,0.02}.
]

更严谨的是按 empirical information scale standardize：

[
\beta_T
=======

h
\left(
\sum_t \widehat g_t^2
\right)^{-1/2}.
]

但这会让 MC 复杂。主表可以用 simple alternatives，appendix 用 local scaling。

---

## D. Oracle-equivalence diagnostics

这是我们这篇 paper 特有的 MC，不要漏。

定义

[
D_{S,T}
=======

\frac{
T^{-1/2}\sum_t
{\widehat u_t(\beta_0)w_t^c-U_t\widetilde w_t}
}{
\left[
T^{-1}\sum_t U_t^2\widetilde w_t^2
\right]^{1/2}
}.
]

如果 theorem 对，应该：

[
D_{S,T}\to_p0.
]

也报告 denominator error：

[
D_{V,T}
=======

\frac{
T^{-1}\sum_t\widehat u_t(\beta_0)^2(w_t^c)^2
}{
T^{-1}\sum_tU_t^2\widetilde w_t^2
}
-1.
]

它应当趋近 0。

这两个 diagnostic 可以放一张表：

| (T) | mean (|D_S|) | sd (D_S) | mean (|D_V|) |
|---|---:|---:|---:|

这个非常有价值，因为它直接验证 “feasible = oracle”。

---

## E. Robustness–efficiency frontier

这是最有新意的 MC 图。

对

[
w_b(x)=\tanh(x/b)
]

计算：

[
\widehat{\operatorname{RE}}(b)
==============================

\widehat{\cos^2}\angle(g_{w_b},g^\star).
]

同方差下：

[
g^\star=M_KX,
]

[
g_{w_b}=M_Kw_b(X).
]

所以 sample relative efficiency 可以估为：

[
\boxed{
\widehat{\operatorname{RE}}(b)
==============================

\frac{
{(M_Kw_b)'(M_KX)}^2
}{
{(M_Kw_b)'(M_Kw_b)}
{(M_KX)'(M_KX)}
}.
}
]

然后画：

[
b\mapsto \widehat{\operatorname{RE}}(b),
]

以及：

[
b\mapsto \text{size distortion},
]

[
b\mapsto \text{power},
]

[
b\mapsto \text{CI length}.
]

这张图是 MC 的灵魂。

它展示：

[
\boxed{
\text{larger }b
\Rightarrow
\text{more efficient but less saturated;}
}
]

[
\boxed{
\text{smaller }b
\Rightarrow
\text{more robust but less efficient.}
}
]

这会非常好地服务我们最终 story。

---

# 10. MC 最小正文配置

如果篇幅有限，正文放 4 个核心结果就够：

## Table 1: Size across persistence regimes

比较 oracle bounded、proposed profile、intercept-only、efficient-score、HAC。

目的：证明 proposed profile EL size 稳。

## Table 2: Coverage and CI length

证明 proposed method inference 合理，且 nuisance adjustment 有影响。

## Table 3: Oracle-equivalence diagnostics

证明 feasible score 确实接近 oracle score。

## Figure 1: Robustness–efficiency frontier

展示 (b) 改变时，relative efficiency 和 size/power 的变化。

其他内容放 appendix。

---

# 11. 预期结果应该是什么

理想结果不是“我们所有方面都最好”。真实而高级的结果应该是：

### Under (H_0)

* Oracle bounded EL size close to nominal.
* Proposed profile-sieve bounded EL close to oracle.
* Intercept-only EL distorted when (m(W)) nonlinear.
* Efficient-score / raw-(X) score good in stationary case, but more sensitive under local-to-unity / unit root, especially when (\kappa\neq0).
* OLS/HAC most sensitive.

### Under alternatives

* Efficient-score method has better power in regular stationary case.
* Bounded EL loses some power but maintains better size under high persistence.
* Larger (b) improves power/relative efficiency but may worsen size robustness.
* Smaller (b) improves size stability but lowers power.

This is exactly the tradeoff story.

---

# 12. 最终 MC story

可以在 paper 里这样写：

> The simulations are designed around three questions. First, does the profile-sieve implementation preserve the oracle persistence-robust Wilks phenomenon after estimating the nonparametric nuisance? Second, does ignoring the nuisance component distort inference? Third, how large is the efficiency cost of using bounded saturated scores rather than the conditional-moment efficient direction? The results show that the proposed statistic closely tracks the oracle bounded EL statistic across persistence regimes, while the efficient raw-score benchmark delivers higher power in regular stationary designs but becomes more sensitive under highly persistent predictors. This confirms the theoretical robustness–efficiency tradeoff.

---

# 13. 最终建议

MC 不要做太散。主线就四句话：

[
\boxed{
\text{Proposed profile EL matches oracle bounded EL.}
}
]

[
\boxed{
\text{Ignoring }m(W)\text{ causes size distortion.}
}
]

[
\boxed{
\text{Raw efficient score gives power but is less robust under persistence.}
}
]

[
\boxed{
\text{Saturation parameter traces a visible robustness–efficiency frontier.}
}
]

这就是 Sinica-level 的 MC。它既支撑 theorem，又支撑 marketing。
