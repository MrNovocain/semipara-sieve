你说得对。上一版像“研究想法清单”，不是 reproducible implementation plan。下面我按 **CLEAN + SOLID** 重写成一个真正能落地、能复现、能扩展的工程方案。

我这里把 CLEAN 解释成：

[
\boxed{
\text{Config-driven, Logged, Explicit, Automated, No hidden state.}
}
]

也就是：所有实验由 config 驱动；每次运行有完整 lineage；输入输出 schema 明确；自动检查；不靠 notebook 手动操作。

---

# 0. 核心原则

整个 MC repo 只允许一种运行方式：

```bash
python -m pseel.run configs/mc/size_main.yaml
```

或者：

```bash
python -m pseel.run configs/mc/frontier_main.yaml
```

不能靠手动改代码、手动改 notebook、手动复制参数。

每次 run 必须生成：

```text
results/
  <run_id>/
    config.yaml
    config_hash.txt
    git_commit.txt
    environment.txt
    manifest.json
    raw_replications.parquet
    summary.csv
    diagnostics.json
    logs.txt
```

这样半年后你看到一个 table，也能知道它来自哪个 config、哪个 seed、哪个 commit、哪个 DGP。

---

# 1. Repo structure

建议结构：

```text
profile-sieve-el/
  pyproject.toml
  README.md
  requirements.lock

  configs/
    mc/
      size_main.yaml
      oracle_equiv_main.yaml
      frontier_main.yaml
      power_main.yaml
      ci_main.yaml
    dgp/
      baseline.yaml
      nonlinear_mild.yaml
      nonlinear_strong.yaml
    methods/
      bounded_profile.yaml
      oracle_bounded.yaml
      efficient_profile.yaml
      intercept_only.yaml

  src/
    pseel/
      __init__.py
      contracts.py
      registry.py

      dgp.py
      basis.py
      residualize.py
      weights.py
      el.py
      methods.py
      metrics.py

      experiment.py
      runner.py
      summarize.py
      plotting.py
      io.py
      checks.py

  tests/
    test_el_solver.py
    test_residualize.py
    test_dgp_shapes.py
    test_method_identities.py
    test_reproducibility.py

  scripts/
    make_tables.py
    make_figures.py

  results/
    .gitkeep
```

重点：`notebooks/` 可以有，但只能用于看图和 debug，不能生成 final result。

---

# 2. SOLID design

## S — Single Responsibility

每个模块只做一件事。

```text
dgp.py          只负责生成数据
basis.py        只负责生成 sieve basis
residualize.py  只负责 projection / residualization
weights.py      只负责 w(X) 或 g(X,W)
el.py           只负责 scalar empirical likelihood
methods.py      只负责把 DGP + basis + weight 变成 statistic
metrics.py      只负责 size / power / diagnostics
runner.py       只负责跑 replications
summarize.py    只负责从 raw results 做 summary
```

不要在 `methods.py` 里模拟数据。不要在 `dgp.py` 里算 EL。不要在 notebook 里生成主表。

---

## O — Open/Closed

以后加 DGP 或 method，不改 runner，只注册新类。

例如：

```python
# registry.py
DGP_REGISTRY = {}
METHOD_REGISTRY = {}
WEIGHT_REGISTRY = {}

def register_dgp(name):
    def wrapper(cls):
        DGP_REGISTRY[name] = cls
        return cls
    return wrapper
```

以后加一个 mildly explosive DGP，只写新 class：

```python
@register_dgp("predictive_ar1")
class PredictiveAR1DGP:
    ...
```

runner 不需要变。

---

## L — Liskov Substitution

所有 DGP 都必须返回同一个 `Dataset` schema。所有 method 都必须接受同一个 `Dataset`，返回同一个 `MethodResult`。

---

## I — Interface Segregation

不要搞一个巨大的 `SimulationObject`。用小接口。

```python
class DGP(Protocol):
    def simulate(self, seed: int) -> Dataset: ...

class Method(Protocol):
    def evaluate(self, data: Dataset, beta0: float) -> MethodResult: ...

class Weight(Protocol):
    def transform(self, x: np.ndarray) -> np.ndarray: ...

class Basis(Protocol):
    def make(self, w: np.ndarray) -> np.ndarray: ...
```

---

## D — Dependency Inversion

runner 不依赖具体 DGP 或具体 method，只依赖接口：

```python
class ExperimentRunner:
    def __init__(
        self,
        dgp: DGP,
        methods: list[Method],
        metrics: list[Metric],
        seed_manager: SeedManager,
    ):
        ...
```

这会让你的 MC 可扩展，不会一加 method 就改全局代码。

---

# 3. Data contracts

先定义不可变数据结构。

```python
@dataclass(frozen=True)
class Dataset:
    y: np.ndarray
    x_lag: np.ndarray
    w_lag: np.ndarray
    u: np.ndarray
    m_w: np.ndarray
    meta: dict
```

强制 shape：

[
Y_t=m(W_{t-1})+\beta X_{t-1}+U_t.
]

所以所有数组长度必须相同：

```python
len(y) == len(x_lag) == len(w_lag) == len(u) == T
```

方法输出：

```python
@dataclass(frozen=True)
class MethodResult:
    method_name: str
    el_stat: float
    reject_5: bool
    beta0: float
    feasible: bool
    lambda_hat: float | None
    diagnostics: dict
```

replication 输出行必须 long format：

```text
run_id
rep
seed
T
rho_type
rho_value
beta
kappa
xi
m_type
method
el_stat
reject_5
feasible
DS
DV
RE
ci_lower
ci_upper
ci_length
coverage
```

这样后面 summary 全部从 raw parquet 生成。

---

# 4. Config-driven experiment

一个 config 必须完整描述实验。

例如 `configs/mc/size_main.yaml`：

```yaml
experiment:
  name: size_main
  task: size
  n_replications: 2000
  global_seed: 20260701
  alpha: 0.05

dgp:
  name: predictive_ar1
  params:
    T_values: [100, 250, 500]
    rho_designs:
      - {label: stationary_low, formula: fixed, value: 0.5}
      - {label: stationary_high, formula: fixed, value: 0.95}
      - {label: local_to_unity, formula: local, c: 5}
      - {label: unit_root, formula: fixed, value: 1.0}
    beta: 0.0
    a_w: 0.5
    kappa: 0.5
    xi: 0.3
    burnin: 300
    m:
      name: sinus_quad
      params:
        a1: 0.5
        a2: 0.3

sieve:
  basis: polynomial
  include_intercept: true
  standardize_w: true
  K_values: [6]

weights:
  bounded_main:
    name: tanh
    params:
      b: 1.0

methods:
  - oracle_bounded
  - profile_bounded
  - intercept_only_bounded
  - profile_efficient

el:
  solver:
    tolerance: 1.0e-10
    max_iter: 100
    infeasible_value: .inf

outputs:
  save_raw: true
  save_diagnostics: true
```

注意：所有 DGP 参数都必须在 config 里，不允许藏在代码默认值里。

---

# 5. Seed management

不要 `np.random.seed()` 到处乱设。用 `SeedSequence`。

```python
class SeedManager:
    def __init__(self, global_seed: int):
        self.seed_seq = np.random.SeedSequence(global_seed)

    def spawn(self, n: int) -> list[int]:
        children = self.seed_seq.spawn(n)
        return [int(c.generate_state(1)[0]) for c in children]
```

每个 replication 的 seed 存进 raw output。

这样你可以精确复现第 1374 次 replication：

```bash
python -m pseel.debug_rep --config configs/mc/size_main.yaml --rep 1374
```

---

# 6. DGP implementation contract

DGP 必须明确 indexing。

```python
@register_dgp("predictive_ar1")
class PredictiveAR1DGP:
    def __init__(self, params: DGPParams):
        self.params = params

    def simulate(self, seed: int, T: int, rho: float, beta: float) -> Dataset:
        rng = np.random.default_rng(seed)

        # simulate length T + burnin + 1
        # construct X_t, W_t
        # output y_t using x_{t-1}, w_{t-1}
        return Dataset(...)
```

Innovation covariance：

[
\Sigma=
\begin{pmatrix}
1&\kappa&0\
\kappa&1&\xi\
0&\xi&1
\end{pmatrix}.
]

必须有 check：

```python
assert np.all(np.linalg.eigvalsh(Sigma) > 1e-10)
```

否则有些 (\kappa,\xi) 组合不是 positive definite。

---

# 7. Projection implementation

不要显式构造 (M_K)。

```python
class Residualizer:
    def __init__(self, P: np.ndarray, ridge: float = 0.0):
        self.P = P
        self.gram = P.T @ P
        self.ridge = ridge

    def residualize(self, z: np.ndarray) -> np.ndarray:
        G = self.gram + self.ridge * np.eye(self.gram.shape[0])
        coef = np.linalg.solve(G, self.P.T @ z)
        return z - self.P @ coef
```

测试必须包括：

```python
r = residualizer.residualize(z)
assert np.linalg.norm(P.T @ r) < 1e-8
```

---

# 8. EL solver

Scalar EL solver 单独写，单独测试。

```python
def empirical_likelihood_scalar(z: np.ndarray, tol=1e-10) -> ELResult:
    z = np.asarray(z)
    if np.min(z) > 0 or np.max(z) < 0:
        return ELResult(np.inf, None, False)

    lower = max([-1 / zi for zi in z if zi > 0], default=-np.inf)
    upper = min([-1 / zi for zi in z if zi < 0], default=np.inf)

    # solve sum z_i / (1 + lambda z_i)=0
    ...
```

Tests：

1. If `mean(z)=0`, then (\lambda\approx0), EL stat (\approx0).
2. Scaling invariance:
   [
   EL(z)=EL(cz)
   ]
   for (c\neq0).
3. Convex hull infeasible when all (z>0).
4. Quadratic approximation:
   [
   EL(z)\approx \frac{(\sum z)^2}{\sum z^2}
   ]
   when mean small.

---

# 9. Method implementations

所有 method 继承同一个接口。

## Oracle bounded

```python
class OracleBoundedEL(Method):
    def evaluate(self, data, beta0):
        w = tanh_weight(data.x_lag, b)
        w_centered = w - w.mean()
        z = data.u * w_centered
        return el(z)
```

## Profile bounded

```python
class ProfileBoundedEL(Method):
    def evaluate(self, data, beta0):
        P = basis.make(data.w_lag)
        R = Residualizer(P)

        uhat = R.residualize(data.y - beta0 * data.x_lag)
        w = weight.transform(data.x_lag)
        wc = R.residualize(w)

        z = uhat * wc

        diagnostics = {
            "orth_u": norm(P.T @ uhat),
            "orth_w": norm(P.T @ wc),
        }

        return el(z)
```

## Intercept-only bounded

Same as profile bounded, but (P=\mathbf 1).

## Efficient profile

```python
class ProfileEfficientEL(Method):
    def evaluate(self, data, beta0):
        P = basis.make(data.w_lag)
        R = Residualizer(P)

        uhat = R.residualize(data.y - beta0 * data.x_lag)
        g_eff = R.residualize(data.x_lag)

        z = uhat * g_eff
        return el(z)
```

注意：efficient profile 是 benchmark，不是 main robust method。

---

# 10. Metrics

metrics 不应该散落在 runner 里。单独写。

## Size metric

```python
reject_5 = el_stat > chi2.ppf(0.95, 1)
```

## Oracle equivalence diagnostic

只对 proposed profile bounded 和 oracle bounded 计算：

[
D_S=
\frac{
T^{-1/2}\sum(\widehat u_tw_t^c-U_t\widetilde w_t)
}{
\sqrt{T^{-1}\sum U_t^2\widetilde w_t^2}
}.
]

[
D_V=
\frac{
T^{-1}\sum \widehat u_t^2(w_t^c)^2
}{
T^{-1}\sum U_t^2\widetilde w_t^2
}
-1.
]

这些 diagnostics 必须由 method 返回必要 components，而不是重新算一遍。

所以 `MethodResult` 可以附加：

```python
components = {
    "uhat": ...,
    "instrument": ...,
    "z": ...
}
```

但 raw parquet 不保存大数组。大数组只在 debug mode 保存。

---

# 11. Frontier experiment config

单独一个 config，不和 size main 混在一起。

```yaml
experiment:
  name: frontier_main
  task: frontier
  n_replications: 2000
  global_seed: 20260702

frontier:
  b_values: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

methods:
  - profile_bounded_frontier
  - profile_efficient
```

对每个 (b) 算：

[
g_b=M_Kw_b(X),
\qquad
g^\star=M_KX.
]

[
\widehat{RE}(b)=
\frac{(g_b'g^\star)^2}{(g_b'g_b)(g^{\star\prime}g^\star)}.
]

输出 long format：

```text
rep, T, rho_label, b, RE, el_stat, reject_5, feasible
```

这样 figure 很容易画。

---

# 12. Automated checks

每个 run 前自动检查：

1. Config schema valid.
2. All method names registered.
3. All DGP covariance matrices positive definite.
4. (K<T/5) 或其他安全限制。
5. Basis includes intercept.
6. EL solver tolerance positive.
7. Output dir does not already exist unless `--overwrite`.

每个 replication 内部检查：

```python
assert np.all(np.isfinite(data.y))
assert len(data.y) == T
assert np.linalg.matrix_rank(P) == K
```

如果 failure，记录，不要 silently skip。

---

# 13. Result lineage

每个 run_id 由 config hash + timestamp 生成：

```text
size_main_20260623_142011_a82f9c1
```

`manifest.json`：

```json
{
  "run_id": "size_main_20260623_142011_a82f9c1",
  "config_hash": "a82f9c1...",
  "git_commit": "...",
  "python_version": "...",
  "numpy_version": "...",
  "pandas_version": "...",
  "n_replications": 2000,
  "started_at": "...",
  "finished_at": "...",
  "status": "completed"
}
```

这就是 reproducibility 的底线。

---

# 14. Testing plan

最少要有这些 unit tests。

## `test_residualize.py`

* residual is orthogonal to basis；
* residualizing a basis column gives zero；
* residualizing constant gives zero if intercept included。

## `test_el_solver.py`

* scaling invariance；
* zero mean gives zero statistic；
* infeasible convex hull caught；
* quadratic approximation correct。

## `test_dgp_shapes.py`

* output shape；
* no NaN；
* empirical innovation covariance roughly matches target for large T。

## `test_method_identities.py`

* profile residual orthogonal to P；
* profile instrument orthogonal to P；
* if (m=0) and (K=1), profile bounded equals intercept-only bounded；
* oracle diagnostic improves as T increases in a smoke simulation。

## `test_reproducibility.py`

* same config + same seed gives identical raw results；
* different seed changes results。

---

# 15. Reproducible run order

不要一口气跑全量。按这个顺序：

## Run 1: Smoke test

```yaml
T_values: [50]
rho_designs: [0.95]
n_replications: 20
methods:
  - oracle_bounded
  - profile_bounded
```

目的：代码不崩。

## Run 2: Oracle sanity

```yaml
T_values: [100, 250]
rho_designs: [0.95, 1]
n_replications: 500
methods:
  - oracle_bounded
```

如果 oracle EL 自己 size 不对，先停。

## Run 3: Profile equivalence

```yaml
methods:
  - oracle_bounded
  - profile_bounded
metrics:
  - DS
  - DV
```

看 (D_S,D_V) 是否随 T 下降。

## Run 4: Main size

完整方法，(R=2000)。

## Run 5: Frontier

不同 (b)。

## Run 6: CI / power

最后做，因为最慢。

---

# 16. Definition of done

第一阶段完成的标准不是“代码能跑”，而是得到这些 artifact：

```text
results/<run_id>/
  raw_replications.parquet
  summary_size.csv
  summary_oracle_equiv.csv
  figure_frontier.pdf
  manifest.json
```

并且：

1. 同一个 config 重跑，结果 bitwise identical 或 numerical identical；
2. `pytest` 全部通过；
3. 每张表都能由 `scripts/make_tables.py --run_id <run_id>` 自动生成；
4. 不需要打开 notebook。

---

# 17. 最终 paper table mapping

代码输出和论文表格一一对应。

| Paper object               | Source config            | Source script                         |
| -------------------------- | ------------------------ | ------------------------------------- |
| Table 1 Size               | `size_main.yaml`         | `make_tables.py --table size`         |
| Table 2 Oracle equivalence | `oracle_equiv_main.yaml` | `make_tables.py --table oracle_equiv` |
| Figure 1 Frontier          | `frontier_main.yaml`     | `make_figures.py --fig frontier`      |
| Table 3 Coverage           | `ci_main.yaml`           | `make_tables.py --table coverage`     |

这叫 reproducible paper pipeline。

---

# 18. 最小 implementation milestone

先做这个，不要贪：

```text
M1:
  DGP
  polynomial basis
  residualizer
  scalar EL solver
  oracle bounded method
  profile bounded method
  size + DS/DV metrics
  YAML runner
  raw parquet output
  pytest
```

M1 完成后，你就已经能验证主 theorem。

M2 再加：

```text
intercept-only
efficient-profile
frontier b-grid
relative efficiency metric
```

M3 最后加：

```text
CI inversion
power
HAC benchmark
empirical study
```

---

# 19. 最终压缩版

真正的 reproducible implementation plan 是：

[
\boxed{
\text{One config } \to \text{ one run_id } \to \text{ raw parquet } \to \text{ deterministic tables/figures}.
}
]

工程上：

[
\boxed{
\text{DGP, basis, residualizer, weight, EL solver, method, metric, runner 全部分离。}
}
]

统计上：

[
\boxed{
\text{先验证 oracle bounded，再验证 feasible bounded，再做 efficiency frontier。}
}
]

这样才符合 CLEAN + SOLID。
