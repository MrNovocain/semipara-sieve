可以。先纠正一个关键表述：**kernel 不是数学上绝对不能做**。它可以借助 undersmoothing、bias correction、cross-fitting 或 Robinson-type residualization 来完成 semiparametric inference。真正的问题是，它不能在不大改现有证明的情况下，提供我们现在依赖的有限样本正交投影恒等式。

所以 novelty 不能写成“kernel 不行而 sieve 行”，而应写成：

[
\boxed{
\text{multiple unknown breaks in a nonparametric nuisance function can create spurious predictability under persistent regressors;}
}
]

[
\boxed{
\text{regime-specific sieve projection removes that break-induced score component and restores EL inference for }\beta.
}
]

下面是整篇论文的 writing plan。

---

# 一、论文最终身份

## 推荐题目

### 首选

[
\boxed{
\textbf{Multiple Nuisance Breaks and Spurious Predictability under Persistent Regressors}
}
]

### 方法导向备选

[
\textbf{Break-Adaptive Sieve Empirical Likelihood for Persistent Predictive Regressions}
]

第一版更好，因为它先卖**经济统计问题**，而不是工具组合。

## 一句话主旨

> We study inference on a stable predictive coefficient when the nonparametric nuisance component undergoes finitely many unknown structural breaks.

形式上：

[
Y_t
===

\beta_0X_{t-1}
+
m_{j(t)}(W_{t-1})
+
U_t,
]

其中 (X_t) persistent，(\beta_0) 在全样本保持稳定，而

[
j(t)=j
\quad\Longleftrightarrow\quad
k_{j,0}<t\le k_{j+1,0}
]

允许 (m_j) 在 (q_0) 个未知日期发生变化。

---

# 二、novelty 必须怎样定位

先把不能作为 novelty 的内容删掉：

* sieve 本身不是新；
* EL 本身不是新；
* multiple-break detection 不是新；
* SBS、NOT、WBS2 不是我们的贡献；
* persistent predictive regression 也不是新；
* “predictive regression + structural break + EL”本身也不是新。

Hong et al. 的公开摘要已经明确把贡献定义为：在 semiparametric partially linear regression 中，利用 profile weighted estimating equations 和 EL，对 stationary 与 nonstationary regressors 提供统一推断。

更接近我们的 Cai–Chang working paper 已经研究了 predictive regression 中一个未知结构突变，并用 weighted-score EL 检验 predictability；但它的 break 位于 parametric intercept/slope，且还允许 predictor 自身发生 break。

此外，Andersen–Varneskov 已经处理 persistent predictive systems 中多个未知参数结构突变，目标是 sup-Wald 型 instability/break testing。([NBER][1]) 最近也已有专门估计和检验 nonparametric regression 中 multiple structural breaks 的工作。([ResearchGate][2])

因此我们的 novelty 必须落在这个交集上：

[
\boxed{
\begin{gathered}
\text{persistent predictor}\
+\ \text{stable predictive coefficient }\beta\
+\ \text{multiple unknown breaks only in the nonparametric nuisance}\
+\ \text{break-induced false-predictivity theorem}\
+\ \text{valid post-break-estimation EL inference for }\beta.
\end{gathered}
}
]

## 最终四项贡献

### Contribution 1：新的 nuisance-instability 问题

现有 stable semiparametric model 是

[
Y_t=\beta_0X_{t-1}+m(W_{t-1})+U_t.
]

我们允许

[
m_t(W)
======

\sum_{j=0}^{q_0}
m_j(W)
1{k_{j,0}<t\le k_{j+1,0}}.
]

关键不是“允许 (m) 变化”本身，而是：

[
\boxed{
\beta_0\text{ 保持稳定，所有结构不稳定性都被视为 nuisance。}
}
]

因此我们的目标不是估计 regime-specific predictive slopes，也不是检验 predictive coefficient 是否 break，而是判断在允许 nuisance instability 后，(\beta_0) 是否仍然显著。

### Contribution 2：omitted-break failure theorem

证明 stable-nuisance procedure 在真实 nuisance 有 breaks 时，score 出现额外项：

[
S_T^{\mathrm{stable}}(\beta_0)
==============================

\frac1{\sqrt T}U'M_{K,0}w
+
\frac1{\sqrt T}\mu'M_{K,0}w
+
o_p(1),
]

其中

[
\mu_t=m_{j(t)}(W_{t-1}).
]

定义

[
B_T
===

\frac1{\sqrt T}\mu'M_{K,0}w.
]

当

[
B_T\not=o_p(1),
]

stable semiparametric score 就不再是纯粹关于 (\beta) 的 score。它包含一个 break-induced direction。

这才是我们的 Perron-style mechanism：

[
\boxed{
\text{persistent predictor does not itself create the signal;}
\quad
\text{it amplifies the alignment between an omitted low-frequency break and the score direction.}
}
]

必须明确：不是每个 break 都必然制造虚假拒绝。真正条件是 break residual 与 residualized persistent weight 的投影非忽略：

[
T^{-1/2}
\langle M_{K,0}\mu,M_{K,0}w\rangle
\not=o_p(1).
]

### Contribution 3：break-adaptive orthogonal score

给定 partition

[
\Lambda=(k_1,\ldots,k_q),
]

定义 block sieve matrix

[
Q_{K,\Lambda}
=============

\left[
D_0(\Lambda)P_K,,
D_1(\Lambda)P_K,,
\ldots,,
D_q(\Lambda)P_K
\right],
]

其中 (D_j(\Lambda)) 是第 (j) 个 regime 的对角 indicator matrix。

然后

[
M_{K,\Lambda}
=============

I-
Q_{K,\Lambda}
(Q_{K,\Lambda}'Q_{K,\Lambda})^{-1}
Q_{K,\Lambda}'.
]

最终 score 是

[
\widehat Z_t(\beta,\Lambda)
===========================

\left{
M_{K,\Lambda}(Y-\beta X)
\right}*t
\left{
M*{K,\Lambda}w
\right}_t.
]

这里的核心仍然是当前文章的双重 residualization：

[
\text{outcome residual}
\times
\text{weight residual}.
]

区别只是 nuisance space 从

[
\operatorname{span}{P_K(W)}
]

变成

[
\operatorname{span}
\left{
1(t\in I_j)P_K(W):
j=0,\ldots,q
\right}.
]

正交投影同时具有 minimum-distance 和 orthogonal-residual 两种刻画，这正是该构造的几何基础。

### Contribution 4：estimated-partition Wilks theorem

未知 break dates 不是外部预处理，而是在每个 null-imposed (\beta) 下 profile：

[
\widehat\Lambda_q(\beta)
========================

\arg\min_{\Lambda\in\mathcal L_q}
\left|
M_{K,\Lambda}(Y-\beta X)
\right|^2.
]

如果 (q) 也未知，则

[
\widehat q(\beta)
=================

\arg\min_{0\le q\le\bar q}
\left[
RSS_K{\beta,\widehat\Lambda_q(\beta)}
+
\operatorname{pen}_T(q,K)
\right].
]

然后证明

[
\ell_T^{BA}
\left{
\beta_0,
\widehat q(\beta_0),
\widehat\Lambda(\beta_0)
\right}
\Rightarrow\chi_1^2.
]

这才是主要 inference result。

---

# 三、为什么保留 sieve，而不改回 kernel

这一部分应该作为 method section 的 remark，而不是 introduction 的 novelty claim。

## 正确表述

> Kernel methods are possible in principle. We employ a sieve because the empirical-likelihood argument requires a projection structure that remains exact after introducing regime-specific nuisance functions.

sieve 给出：

[
Q_{K,\Lambda}'M_{K,\Lambda}=0,
\qquad
M_{K,\Lambda}'=M_{K,\Lambda},
\qquad
M_{K,\Lambda}^2=M_{K,\Lambda}.
]

因此任何 regime-specific sieve component

[
Q_{K,\Lambda}c
]

都满足

[
(Q_{K,\Lambda}c)'M_{K,\Lambda}w=0
]

，这是有限样本 exact cancellation。

一般 kernel smoother 写作

[
\widehat m=S_hY,
]

但通常

[
S_h'\neq S_h,
\qquad
S_h^2\neq S_h.
]

所以 (I-S_h) 不是 nuisance space 的正交 residual-maker。kernel 并非不能工作，但需要额外控制：

[
\text{kernel bias}
+
\text{bandwidth stochastic error}
+
\text{break-boundary error}
+
\text{estimated-partition error}.
]

sieve 把这些问题压缩成：

[
\text{sieve approximation error}
+
\text{partition estimation error}.
]

最好的表述是：

[
\boxed{
\text{sieve is not merely a nonparametric estimator here;}
\quad
\text{it is a geometry-compatible regularization of the nuisance space.}
}
]

---

# 四、正式模型

令

[
0=k_{0,0}<k_{1,0}<\cdots<k_{q_0,0}<k_{q_0+1,0}=T.
]

主模型：

[
Y_t
===

\beta_0X_{t-1}
+
\sum_{j=0}^{q_0}
m_j(W_{t-1})
1{k_{j,0}<t\le k_{j+1,0}}
+
U_t.
]

predictor 保持当前 persistence framework：

[
X_t=\theta+\rho_TX_{t-1}+\varepsilon_t,
]

覆盖 stationary、local-to-unity 或 unit-root class，但**主文不允许 (X_t) 自己 break**。否则会和 Cai–Chang 以及 parameter-instability literature 混在一起。

## 核心限制

[
q_0<\infty,
]

[
\min_j(k_{j+1,0}-k_{j,0})\ge \epsilon T,
]

[
m_j\in\mathcal H^s,
]

以及 break separation：

[
\Delta_T
========

\min_j
|m_{j+1}-m_j|_{L^2(P_W)}.
]

强 break theory 可以要求类似

[
T\Delta_T^2
\gg
K\log T.
]

具体指数以后根据 segmentation proof 确定。

## Identification

扩大 nuisance space 后，必须保留足够的 predictive direction：

[
\frac1T
w'M_{K,\Lambda_0}w
\overset{p}{\longrightarrow}
Q_w>0.
]

否则 (w) 几乎被 regime-specific nuisance space 吸收，(\beta) 无法识别。

---

# 五、break detection 的设计

## 理论对象：global profile-sieve segmentation

对任意区间 ([s,e])，定义

[
\mathcal C_K(s,e;\beta)
=======================

\min_{c\in\mathbb R^K}
\sum_{t=s}^e
\left[
Y_t-\beta X_{t-1}
-P_K(W_{t-1})'c
\right]^2.
]

给定 (q)：

[
RSS_K(\beta,\Lambda)
====================

\sum_{j=0}^{q}
\mathcal C_K(k_j+1,k_{j+1};\beta).
]

理论上定义：

[
\widehat\Lambda_q(\beta)
========================

\arg\min_{\Lambda\in\mathcal L_q(\epsilon)}
RSS_K(\beta,\Lambda).
]

这是 proof object。

## 计算对象：multiscale candidate generation

实现时不必穷举所有 partitions。可以使用 SBS 或 NOT 产生 candidate breaks，再以 profile-sieve RSS 与 IC pruning 做最终选择。SBS 的 seeded intervals 和 NOT 的 localized intervals 都允许把用户自定义的 split-gain statistic 嵌进去。([arXiv][3])

局部 split gain：

[
G_K(s,e,b;\beta)
================

## \mathcal C_K(s,e;\beta)

## \mathcal C_K(s,b;\beta)

\mathcal C_K(b+1,e;\beta).
]

但文章必须说清：

[
\boxed{
\text{SBS/NOT are computational devices, not the statistical novelty.}
}
]

## 为什么在每个 (\beta) 下重新 profile partition

不要先估一个 (\widehat\beta)，再固定它检测 breaks。persistent (X_t) 下，preliminary-(\beta) error 乘上 (X_t) 可能污染 break criterion。

更干净的是：

[
\beta=\beta_0
]

下估计 partition，然后检验该 (\beta_0)。

置信区间通过 inversion：

[
\mathcal C_{1-\alpha}
=====================

\left{
\beta:
\ell_T^{BA}(\beta)
\le
\chi^2_{1,1-\alpha}
\right}.
]

这样 segmentation 与 inference 都是 null-imposed profile。

---

# 六、理论章节的 theorem roadmap

## Proposition 1：stable-score decomposition

在 (H_0:\beta=\beta_0) 下：

[
S_T^{stable}(\beta_0)
=====================

A_T+B_T+o_p(1),
]

其中

[
A_T
===

\frac1{\sqrt T}U'M_{K,0}w,
]

[
B_T
===

\frac1{\sqrt T}\mu'M_{K,0}w.
]

这是全文最重要的 decomposition。

## Theorem 1：false-predictivity result

分两种情形。

### Fixed break

若

[
|B_T|\overset{p}{\longrightarrow}\infty,
]

则 stable EL test 满足

[
P{
\ell_T^{stable}(\beta_0)>c_\alpha
}
\to1,
]

即使 (\beta_0=0)。

### Local break

若

[
B_T\Rightarrow B,
\qquad
P(B\neq0)>0,
]

则 stable EL 极限不再是 central (\chi^2)，而是 noncentral 或 random-noncentral mixture。

这条 theorem 是“旧方法为什么失败”。

## Theorem 2：known-partition Wilks

先假设 (\Lambda_0) 已知：

[
\ell_T(\beta_0,\Lambda_0)
\Rightarrow\chi_1^2.
]

你现在文章中的几何 Wilks theorem 应当降级为这条 theorem 的核心 lemma，而不是整篇文章唯一主结果。

## Theorem 3：break estimator consistency

在 fixed (q_0) 下证明：

[
\widehat q=q_0
]

或先假设 (q_0) known，并证明

[
\max_{j\le q_0}
|\widehat k_j-k_{j,0}|
======================

O_p(r_T),
]

至少需要

[
r_T=o(\sqrt T).
]

固定幅度 break 最理想是 (O_p(1))。

## Theorem 4：estimated-partition Wilks

证明

[
\ell_T
{\beta_0,\widehat\Lambda(\beta_0)}
----------------------------------

# \ell_T(\beta_0,\Lambda_0)

o_p(1),
]

从而

[
\ell_T^{BA}(\beta_0)
\Rightarrow\chi_1^2.
]

这里不要强行证明

[
|M_{\widehat\Lambda}-M_{\Lambda_0}|_{op}\to0,
]

因为 projection spaces 只要换几行，operator norm 未必容易控制。更好的证明是直接控制被误分类 observations 的贡献：

[
\mathcal M_T
============

{t:j_{\widehat\Lambda}(t)\neq j_{\Lambda_0}(t)},
]

并利用

[
|\mathcal M_T|
==============

O_p
\left(
\sum_j|\widehat k_j-k_{j,0}|
\right)
=======

o_p(\sqrt T).
]

## Corollary：unknown (q_0)

先用高层条件：

[
P(\widehat q=q_0)\to1.
]

则 plug-in EL 仍然是 (\chi_1^2)。

不要第一版就做 (q_T\to\infty)。

## Theorem 5：local power

在适合 persistent weighted score 的 local alternative 下：

[
\beta_T=\beta_0+\frac{h}{r_T},
]

证明 proposed statistic 有 noncentral limit，而其 noncentrality 来自 (h)，不是 nuisance break。

这正式区分：

[
\text{break-induced score drift}
\quad\text{和}\quad
\text{genuine predictive drift}.
]

---

# 七、当前 proof 必须修改的 subtle points

## 1. Oracle weight 不能再只是 global-centred weight

当前文章使用

[
\widetilde w_t=w_t-\bar w.
]

但 multiple-break nuisance space 包含 (q_0+1) 个 regime intercepts。于是

[
M_{K,\Lambda_0}w
]

更接近的是 regime-specific residual：

[
v_{t,0}
=======

w_t-
\Pi_{\mathcal N_{\Lambda_0}}w_t,
]

而不是 global centering。

oracle score 应改成

[
A_T^{(q)}
=========

\frac1{\sqrt T}
\sum_{t=1}^T
U_tv_{t,0}.
]

这不是符号修改，而是 theorem 的 oracle object 发生了变化。

## 2. 当前 (K/\sqrt T\to0) 改成 block dimension

固定 (q_0) 时，相关维数是

[
K_q=(q_0+1)K.
]

因此 rate condition 写成

[
\frac{(q_0+1)K}{\sqrt T}\to0,
]

以及

[
\zeta_K
\sqrt{
\frac{(q_0+1)K}{T}
}
\to0.
]

固定 (q_0) 时和原 rate 同阶，但必须写清楚。

## 3. 第一版保留当前 approximation rate

最少改动版本继续用

[
\sqrt T,a_K\to0.
]

这样可以最大程度复用现有 proof。

更高级的第二版可以利用 outcome nuisance 与 weight nuisance 的双重正交性，将条件放松为 product-rate，例如

[
\sqrt T,a_{m,K}a_{w,K}\to0.
]

但这需要重新定义 population residualization 和证明 Neyman orthogonality。第一稿不要同时做。

## 4. 不要轻易声称 semiparametric efficiency

目前最稳妥的名称是：

[
\text{orthogonalized weighted score},
]

不是 efficient score。

只有在明确 working likelihood、conditional variance structure 和 optimal weight 后，才能证明 efficiency bound。

---

# 八、整篇文章的 section plan

## Section 1. Introduction

建议七段。

### Paragraph 1：predictability 与 persistence

说明 persistent predictors 让标准 predictive-regression inference 非标准。

### Paragraph 2：stable semiparametric control

介绍 Hong-type contribution：允许 stable nonparametric control，同时统一 stationary/nonstationary predictor inference。

### Paragraph 3：遗漏的问题

指出现有 semiparametric inference 通常假设：

[
m_t(W)\equiv m(W).
]

但长期样本中，控制变量对 outcome 的 response surface 可能发生 regime changes。

### Paragraph 4：false-predictivity mechanism

直接放核心 decomposition：

[
S_T^{stable}=A_T+B_T+o_p(1).
]

强调 (B_T) 可能被误判为 (\beta)-signal。

### Paragraph 5：方法

介绍 null-imposed profile-sieve segmentation、regime-specific double residualization 和 EL。

### Paragraph 6：贡献

写四条贡献，不要写“we combine”。

### Paragraph 7：literature differentiation

清楚区分 Hong、Cai–Chang、persistent break tests 和 nonparametric break detection。

## Section 2. Model, Identification, and Break-Induced Predictivity

内容：

1. (q_0)-break nuisance model；
2. persistence class；
3. minimum spacing 和 break strength；
4. identification；
5. stable nuisance misspecification；
6. score decomposition；
7. formal definition of break-induced spurious predictivity。

## Section 3. Regime-Specific Sieve Geometry

内容：

1. block sieve space；
2. projection matrix；
3. exact annihilation；
4. known-partition profile residual；
5. orthogonalized weight；
6. EL construction；
7. “Why sieve rather than kernel” remark。

几何是方法的 backbone，但不再作为 introduction 的主 selling point。

## Section 4. Learning Multiple Nuisance Breaks

内容：

1. segment cost；
2. global profile criterion；
3. IC for (q)；
4. multiscale candidate generation；
5. null-imposed estimation；
6. computational algorithm。

算法框可以写成：

[
\beta
\to
\widehat q(\beta),\widehat\Lambda(\beta)
\to
M_{K,\widehat\Lambda(\beta)}
\to
\widehat Z_t(\beta)
\to
\ell_T^{BA}(\beta).
]

## Section 5. Asymptotic Theory

顺序：

1. stable-method failure theorem；
2. known-partition Wilks；
3. partition consistency；
4. estimated-partition Wilks；
5. unknown-(q) corollary；
6. local power。

## Section 6. Monte Carlo Evidence

必须围绕“false versus surviving predictability”，而不只是汇报 size/power 表。

## Section 7. Empirical Application

比较：

[
\ell_T^{stable}(0)
\quad\text{和}\quad
\ell_T^{BA}(0).
]

并报告 score decomposition：

[
\widehat S_T^{stable}
\approx
\widehat S_T^{BA}
+
\widehat B_T.
]

## Section 8. Conclusion

只总结：

* omitted nuisance breaks create score contamination；
* regime-specific projection removes it；
* inference for (\beta) remains valid。

---

# 九、simulation plan

## DGP dimensions

### Persistence

[
\rho=0.5,\ 0.9,\ 0.98,\ 1-c/T,\ 1.
]

### Break number

[
q_0=0,1,3,5.
]

### Break form

Level break：

[
m_{j+1}(w)-m_j(w)=a_j.
]

Shape break：

[
m_{j+1}(w)-m_j(w)=\Delta_j(w),
\qquad E\Delta_j(W)=0.
]

Mixed break：

[
a_j+\Delta_j(w).
]

### Break size

* fixed；
* local；
* near-detection-boundary。

### Dependence

* (X) 与 (W) 独立；
* (X) 与 (W) correlated；
* conditional heteroskedastic (U_t)。

## Competitors

1. Stable sieve EL；
2. Oracle-break sieve EL；
3. Proposed estimated-break sieve EL；
4. Parametric-break EL；
5. Kernel-profile break procedure；
6. No-nuisance benchmark。

kernel competitor 很重要：不是为了证明 kernel 不可能，而是展示 bandwidth 和 boundary sensitivity。

## 报告指标

* test size；
* power；
* confidence-set coverage；
* average length；
* (\widehat q) accuracy；
* break-date error；
* stable-to-adaptive rejection gap。

最重要的图：

[
\text{false rejection rate}
\quad\text{against}\quad
(\rho,\Delta).
]

这能直接展示 persistence 如何放大 omitted nuisance break。

---

# 十、empirical application plan

当前 commodity-return/weather motivation 可以保留，但要满足足够样本量。

模型：

[
R_t
===

\beta X_{t-1}
+
m_{j(t)}(\text{weather}_{t-1})
+
U_t.
]

结果至少报告：

1. stable model 的 predictability p-value；
2. break-adaptive p-value；
3. (\widehat q) 和 break dates；
4. 各 regime 的 estimated weather response surface；
5. estimated break-induced score component；
6. residualized predictor strength：

[
T^{-1}w'M_{K,\widehat\Lambda}w.
]

不能仅仅报告：

> stable rejects, adaptive does not reject, so predictability is false.

更严谨的语言是：

> The evidence of predictability is not robust to allowing multiple breaks in the nuisance response.

---

# 十一、abstract 草稿框架

> We study inference on a stable predictive coefficient in a semiparametric predictive regression where the predictor may be highly persistent and the nonparametric nuisance component may undergo finitely many structural breaks at unknown dates. We show that imposing a stable nuisance function leaves a break-induced component in the orthogonalized score, which can generate spurious evidence of predictability. We propose a null-imposed profile-sieve segmentation procedure to estimate the nuisance partition and construct an empirical likelihood statistic after residualizing both the outcome and the score weight against the resulting regime-specific sieve space. The sieve formulation yields exact sample orthogonality within each candidate regime and separates sieve approximation error from break-date estimation error. Under finite-break, spacing, detectability, and sieve-growth conditions, the empirical likelihood ratio evaluated at the estimated partition has a standard chi-square limit. Simulations document the interaction between predictor persistence and omitted nuisance breaks, and an empirical application examines whether conventional evidence of predictability survives break-adaptive adjustment.

---

# 十二、当前稿件怎样改

你现在的稿件不要废掉。它可以作为新论文的 **known-partition geometric core**。

具体迁移：

* 当前 title 和 abstract：全部重写；
* 当前 Introduction：压缩后移入 Section 3；
* 当前 stable model：替换为 (q_0)-regime model；
* 当前 (P_K)：改成 (Q_{K,\Lambda})；
* 当前 (M_K)：改成 (M_{K,\Lambda})；
* 当前 global-centred oracle weight：改成 regime-residualized oracle weight；
* 当前 projection lemma：改成 block projection lemma；
* 当前 Wilks theorem：成为 known-partition theorem；
* 新增 failure theorem；
* 新增 segmentation theorem；
* 新增 estimated-partition equivalence theorem；
* weather interpretation：移到完整 empirical section。

---

# 十三、最合理的实际写作顺序

第一步先写 Section 2 的 score decomposition。只要这一步不能明确产生 (B_T)，整篇文章就没有足够强的故事。

第二步把当前 proof 全部改成 known-(\Lambda_0) 的 block-sieve version。

第三步证明：

[
\max_j|\widehat k_j-k_{j,0}|=o_p(\sqrt T)
]

足以得到 plug-in Wilks。

第四步再决定是否在主文证明 (\widehat q) consistency；如果太重，就先把 fixed (q_0)、unknown dates 做成主 theorem，把 unknown (q_0) 做成 corollary/high-level extension。

第五步才接 SBS/NOT implementation、simulations 和 empirical application。

整篇文章最终必须围绕这个闭环：

[
\boxed{
\text{stable nuisance misspecification}
\Longrightarrow
\text{break-induced score drift}
\Longrightarrow
\text{spurious predictability}
}
]

[
\boxed{
\text{multiple-break profile sieve}
\Longrightarrow
\text{exact regime-wise orthogonalization}
\Longrightarrow
\text{restored Wilks inference}.
}
]

这才是 reviewer 能够识别、也能被 theorem 支撑的 novelty。

[1]: https://www.nber.org/system/files/working_papers/w28570/w28570.pdf "Testing for Parameter Instability and Structural Change in Persistent Predictive Regressions"
[2]: https://www.researchgate.net/publication/398491878_Estimating_and_Testing_Multiple_Structural_Breaks_in_Nonparametric_Regressions "Estimating and Testing Multiple Structural Breaks in Nonparametric Regressions | Request PDF"
[3]: https://arxiv.org/abs/2002.06633?utm_source=chatgpt.com "Seeded Binary Segmentation: A general methodology for fast and optimal change point detection"
