# Sobol 方差敏感度分析说明

## 1. 背景与目的

本文档对应脚本 `sobol_sensitivity.m`，对双涵道涡扇发动机性能模型进行全局灵敏度分析。分析目标是定量回答：

> **13 个效率/压力恢复参数中，哪些参数对发动机输出量（比推力 $R_{ud}$、比油耗 $C_{ud}$）的不确定性贡献最大？**

这对于指导 TMCMC 贝叶斯反演中哪些参数需要精确先验、哪些参数可以适当放宽，具有直接的工程意义。

---

## 2. 前向模型简介

前向模型 `engine_forward` 来自 `tmcmc13.m`，描述地面静止工况（$M = 0$）下大涵道比涡扇发动机的热力循环：

| 工况参数 | 值 |
|---|---|
| 大气总温 $T_H$ | 288 K |
| 飞行马赫数 $M$ | 0（地面静止）|
| 涵道比 $m$ | 10.0 |
| 压气机压比 $\pi_k$ | 33.0 |
| 涡轮前总温 $T_g$ | 1700 K |

**输出量：**

- $R_{ud}$ — 单位空气流量推力（比推力）$[\text{N·s/kg}]$
- $C_{ud}$ — 单位推力耗油量（比油耗）$[\text{kg/(N·h)}]$

**热力循环主要步骤：**

1. 计算进口气流速度与总压恢复（$\tau_v$、$T_B$）
2. 压气机出口温度 $T_k$，耗油比 $g_T$
3. 热恢复系数 $\lambda_{heat}$（热端/冷端功率匹配）
4. 涡轮膨胀项与单位自由能 $L_{sv}$
5. 最优自由能分配系数 $x_{pc}$（内/外涵道最优分配）
6. 内涵喷管速度 $V_{j1}$、外涵速度 $V_{j2}$，合成推力 $R_{ud}$
7. 燃油流量换算得比油耗 $C_{ud}$

---

## 3. 待分析参数与先验分布

所有 13 个参数均假设服从**独立均匀先验分布** $\theta_i \sim \mathcal{U}(lb_i,\ ub_i)$，与 `tmcmc13.m` 中 TMCMC 的先验设定一致。

| 序号 | 参数 | 含义 | 先验下界 $lb$ | 先验上界 $ub$ |
|---|---|---|---|---|
| 1 | $\eta_k$ | 压气机绝热效率 | 0.84 | 0.86 |
| 2 | $\eta_t$ | 涡轮绝热效率 | 0.86 | 0.92 |
| 3 | $\eta_T$ | 热效率修正系数 | 0.97 | 0.99 |
| 4 | $\eta_m$ | 机械效率 | 0.98 | 0.995 |
| 5 | $\lambda$ | 速度系数比（内/外涵） | 1.00 | 1.06 |
| 6 | $\eta_v$ | 外涵速度系数 | 0.84 | 0.90 |
| 7 | $\eta_{tv}$ | 涡轮速度系数 | 0.88 | 0.96 |
| 8 | $\eta_{c1}$ | 内涵喷管效率 | 0.90 | 0.98 |
| 9 | $\eta_{c2}$ | 外涵喷管效率 | 0.90 | 0.98 |
| 10 | $\sigma_{cc}$ | 燃烧室总压恢复系数 | 0.96 | 1.04 |
| 11 | $\sigma_{kan}$ | 进气道总压恢复系数 | 0.95 | 1.01 |
| 12 | $\sigma_{kask}$ | 混合器总压恢复系数 | 0.96 | 1.02 |
| 13 | $\sigma_{ks}$ | 喷管总压恢复系数 | 0.93 | 0.99 |

> 参数区间宽度差异悬殊（例如 $\eta_k$ 仅宽 0.02，而 $\eta_t$ 宽 0.06），这直接影响绝对方差贡献，因此 Sobol 指标体现的是**在先验范围内**的相对重要性。

---

## 4. Sobol 方差敏感度方法

### 4.1 方差分解（ANOVA-HDMR）

设模型输出 $Y = f(\boldsymbol{\theta})$，参数独立。根据 Sobol 函数分解，总方差可写为：

$$
\operatorname{Var}(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \cdots + V_{1,2,\ldots,k}
$$

其中
$$
V_i = \operatorname{Var}_{\theta_i}\!\left[\mathbb{E}_{\boldsymbol{\theta}_{\sim i}}(Y \mid \theta_i)\right]
$$

### 4.2 一阶主效应指标 $S_i$

$$
S_i = \frac{V_i}{\operatorname{Var}(Y)}
\quad \in [0,1]
$$

**含义**：参数 $\theta_i$ 单独变化（其余参数取期望）时对总方差的贡献比例。反映**独立主效应**。

$$
\sum_{i=1}^{k} S_i \leq 1
$$

等号成立当且仅当参数间无任何交互项。

### 4.3 总效应指标 $ST_i$

$$
ST_i = \frac{\mathbb{E}_{\boldsymbol{\theta}_{\sim i}}\!\left[\operatorname{Var}_{\theta_i}(Y \mid \boldsymbol{\theta}_{\sim i})\right]}{\operatorname{Var}(Y)}
= 1 - \frac{\operatorname{Var}_{\boldsymbol{\theta}_{\sim i}}\!\left[\mathbb{E}_{\theta_i}(Y \mid \boldsymbol{\theta}_{\sim i})\right]}{\operatorname{Var}(Y)}
$$

**含义**：包含参数 $\theta_i$ 的**所有阶次交互效应**之和，即"固定 $\theta_i$ 可消除的方差"。

$$
ST_i \geq S_i, \quad \sum_{i=1}^{k} ST_i \geq 1
$$

$ST_i - S_i$ 越大，说明该参数参与的高阶交互效应越显著。

### 4.4 Saltelli (2010) / Jansen (1999) 估计量

给定两个独立样本矩阵 $\mathbf{A}$、$\mathbf{B}$（各 $N \times k$），构造

$$
\mathbf{AB}^{(i)} = \mathbf{A} \text{ 但第 } i \text{ 列替换为 } \mathbf{B}_{:,i}
$$

估计量：

$$
\hat{S}_i = \frac{\frac{1}{N}\sum_{j=1}^{N} f(\mathbf{B})_j \cdot \left[f(\mathbf{AB}^{(i)})_j - f(\mathbf{A})_j\right]}{\widehat{\operatorname{Var}}(Y)}
$$

$$
\hat{ST}_i = \frac{\frac{1}{2N}\sum_{j=1}^{N} \left[f(\mathbf{A})_j - f(\mathbf{AB}^{(i)})_j\right]^2}{\widehat{\operatorname{Var}}(Y)}
$$

其中 $\widehat{\operatorname{Var}}(Y) = \operatorname{Var}([f(\mathbf{A});\ f(\mathbf{B})])$。

每次完整分析共需 $N \times (k + 2)$ 次模型评估，$k=13$、$N=5000$ 时约 **75,000 次**。

---

## 5. 采样策略

代码采用**拉丁超立方采样（Latin Hypercube Sampling, LHS）**生成矩阵 $\mathbf{A}$ 和 $\mathbf{B}$：

- LHS 将每个参数的取值范围均匀分成 $N$ 份，保证样本在高维空间内分布更均匀；
- 相比纯随机蒙特卡洛，LHS 在相同 $N$ 下通常可将估计误差降低 20%–40%；
- 需要 MATLAB Statistics and Machine Learning Toolbox（`lhsdesign`），若不可用，代码自动回退至伪随机均匀采样。

**收敛建议：**

| $N$ | 总调用次数 | 适用场景 |
|---|---|---|
| 1 000 | ~15 000 | 快速预览 |
| 5 000 | ~75 000 | 默认配置（较好精度） |
| 10 000 | ~150 000 | 高精度，建议发表用结果 |
| 50 000 | ~750 000 | 极高精度 |

---

## 6. 代码结构

```
sobol_sensitivity.m
│
├── 第 1 节  参数定义（与 tmcmc13.m 一致）
├── 第 2 节  采样（LHS，生成矩阵 A、B）
├── 第 3 节  模型批量评估（A、B、AB_i）
├── 第 4 节  Sobol 指标计算
├── 第 5 节  结果打印（表格）
├── 第 6 节  可视化（4 幅图）
│
├── eval_model_batch()       批量调用 engine_forward
├── compute_sobol_indices()  Saltelli/Jansen 估计量
├── plot_sobol()             4 幅图：分组柱图×2、水平柱图×2、雷达图
├── radar_plot()             极坐标蜘蛛网辅助函数
│
└── 前向模型函数（与 tmcmc13.m 保持一致）
    ├── piecewise_kT()
    ├── piecewise_RT()
    ├── delta_cooling()
    └── engine_forward()
```

---

## 7. 输出说明

### 7.1 命令行输出

```
参数           S_i(R)   ST_i(R)    S_i(C)   ST_i(C)
------------------------------------------------------------
eta_k          0.XXXX   0.XXXX     0.XXXX   0.XXXX
...
求和           X.XXXX   X.XXXX     X.XXXX   X.XXXX
```

- `S_i(R)` / `S_i(C)`：比推力 / 比油耗的一阶主效应指标
- `ST_i(R)` / `ST_i(C)`：比推力 / 比油耗的总效应指标

### 7.2 图形输出

| 图号 | 内容 |
|---|---|
| 图 1 | 比推力 $R_{ud}$ 的 13 个参数 Sobol 指标（分组柱状图）|
| 图 2 | 比油耗 $C_{ud}$ 的 13 个参数 Sobol 指标（分组柱状图）|
| 图 3 | 两个输出量的总效应指标降序排列对比（水平柱状图）|
| 图 4 | 两个输出量的 Sobol 指标雷达图（蜘蛛网图）|

---

## 8. 结果解读指南

| 情况 | 含义 | 工程建议 |
|---|---|---|
| $S_i$ 大、$ST_i \approx S_i$ | 独立主效应显著，几乎无交互 | 优先精化该参数的先验或测量 |
| $S_i$ 小、$ST_i$ 大 | 主要通过与其他参数的交互影响输出 | 关注参数组合的联合不确定性 |
| $S_i \approx 0$、$ST_i \approx 0$ | 对输出影响可忽略 | 可考虑在反演中固定该参数 |
| $\sum S_i \ll 1$ | 高阶交互效应显著 | 模型非线性较强，主效应分析不足 |

### 特别注意

- **参数区间宽度**影响结果：区间宽（不确定性大）的参数天然有更高的 Sobol 指标，这是先验信息不足的体现，而非参数本身"物理敏感"。
- $\eta_t$（涡轮效率，区间宽 0.06）和 $\eta_k$（压气机效率，区间宽 0.02）的区间差异达 3 倍，Sobol 指标也会有相应差异。
- 对于地面静止工况（$M=0$），$V_{flight}=0$，与飞行速度相关的项消失，$x_{pc}$ 退化为纯效率比值的函数，$\eta_{tv}$、$\eta_v$ 的交互效应因此可能更突出。

---

## 9. 运行方法

**环境要求：**
- MATLAB R2020b 或更新版本
- 推荐：Statistics and Machine Learning Toolbox（用于 `lhsdesign`）

**运行步骤：**

```matlab
% 在 MATLAB 命令窗口中：
cd('path/to/sensitivity')
sobol_sensitivity
```

脚本自包含，无需额外依赖文件（前向模型内嵌）。

**调整样本数：**

```matlab
% 修改第 2 节中的 N 值
N = 10000;   % 提高精度
```

---

## 10. 参考文献

1. Sobol', I.M. (1993). Sensitivity analysis for non-linear mathematical models. *Mathematical Modelling and Computational Experiment*, 1(4), 407–414.
2. Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259–270.
3. Jansen, M.J.W. (1999). Analysis of variance designs for model output. *Computer Physics Communications*, 117(1–2), 35–43.
4. Iman, R.L., & Conover, W.J. (1980). Small sample sensitivity analysis techniques for computer models, with an application to risk assessment. *Communications in Statistics – Theory and Methods*, 9(17), 1749–1842. （LHS 方法基础）
