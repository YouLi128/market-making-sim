# Deep Dive — What's Worth Improving
# 深挖方向

> 按优先级排列，越靠前对面试/论文价值越高。

---

## 1. 接入真实数据 / Real Data Integration

**现状：** 价格全部用 GBM 模拟，和真实市场行为差距较大。

**可以做：**
- 接入 Lumid 平台 BTC 实时数据
- 或用 Binance 公开 API 拉历史分钟数据（免费）

**价值：** 真实数据有跳空、fat tail、非对称波动，会暴露模型在 GBM 下隐藏的问题，论文说服力大幅提升。

**难度：** ⭐⭐（主要是数据清洗）

---

## 2. 升级毒性检测 — VPIN / Better Adverse Selection

**现状：** 用价格方向一致性（rolling up-fraction）作为毒性代理，很简单，滞后明显。

**可以做：** 实现 **VPIN（Volume-synchronized Probability of Informed Trading）**
- 把成交量分成固定大小的 bucket
- 每个 bucket 里估算买方发起 vs 卖方发起的比例
- VPIN = |buy_volume - sell_volume| / total_volume（滚动平均）

**为什么更好：**
- 基于成交量而不是纯价格，对方向性流量更敏感
- 是学术界和业界广泛使用的标准指标
- Flash Crash 2010 研究中 VPIN 提前预警

**参考论文：** Easley et al. (2012), *Flow Toxicity and Liquidity in a High Frequency World*

**难度：** ⭐⭐⭐

---

## 3. 波动率预测 — GARCH / Volatility Forecasting

**现状：** 波动率用固定值（80%/yr），Phase 4 只做了简单的滚动实现波动率检测。

**可以做：** 用 **GARCH(1,1)** 模型实时预测下一步波动率
```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
```
把预测波动率输入到 AS 模型的 reservation price 和最优价差公式里，实现真正的动态参数。

**价值：** BTC 波动率聚集效应非常明显（大波动后跟大波动），GARCH 能捕捉这个特性。

**难度：** ⭐⭐⭐（可用 `arch` 库简化实现）

---

## 4. 完整的 AS 理论公式 / Exact AS Implementation

**现状：** 我们用的是简化版 AS（gamma 和 kappa 手动标定，时间用归一化 tau）。

**可以做：** 按原论文实现完整公式：
```
r = mid − γ · σ² · q · (T − t)
δ* = γ · σ² · (T − t) / 2  +  (1/γ) · ln(1 + γ/κ)
```
其中 σ 用每步真实波动率，κ 从历史成交数据标定，γ 用效用函数推导。

**价值：** 论文中对标原始文献，能展示你真正读过这篇 paper。

**参考：** Avellaneda & Stoikov (2008), Quantitative Finance 8(3)

**难度：** ⭐⭐⭐（主要是单位标定）

---

## 5. P&L 归因精确化 / Exact P&L Attribution

**现状：** Spread P&L 用 `delta × lot_size × n_fills` 近似，忽略了 reservation price 偏移的影响。

**可以做：** 精确分解每笔成交的利润来源：
```
每次买入成交：spread_gain = mid_at_fill - bid_price
每次卖出成交：spread_gain = ask_price - mid_at_fill
inventory_gain = inventory × (mid_now - mid_prev)
```
这样 spread P&L 和 inventory P&L 的分界更精确，不再有近似误差。

**难度：** ⭐⭐

---

## 6. 多品种对冲 / Multi-Asset Extension

**现状：** 只做 BTC 单品种做市。

**可以做：** 同时做市 BTC + ETH，利用两者的相关性对冲库存
- BTC 多头 + ETH 空头互相抵消部分方向性风险
- 联合 reservation price 考虑跨品种相关系数

**价值：** 真实做市商（Wintermute、Jump）都是多品种同时运行的，展示这个维度很加分。

**难度：** ⭐⭐⭐⭐

---

## 7. 回测框架 / Backtesting Framework

**现状：** 每次运行是单条路径的前向模拟，没有统计意义上的评估。

**可以做：** 跑 1000 条 Monte Carlo 路径，对每个模型输出：
- P&L 分布（均值、标准差、Sharpe ratio）
- 最大回撤分布
- 库存超限频率

用统计方法证明 Phase 4 在**所有路径上**都优于 Phase 1，而不只是一条示例路径。

**难度：** ⭐⭐（代码改动不大，主要是循环 + 统计汇总）

---

## 优先级总结

| 优先级 | 方向 | 理由 |
|--------|------|------|
| ⭐ 最高 | 接入真实数据 | 让整个项目脱离"玩具"层面 |
| ⭐ 最高 | Monte Carlo 回测 | 结论有统计支撑，面试最硬 |
| ⭐⭐ 高 | VPIN 毒性检测 | 有学术出处，替换简单方法 |
| ⭐⭐ 高 | GARCH 波动率 | BTC 场景下效果明显 |
| ⭐⭐⭐ 中 | 精确 P&L 归因 | 论文细节加分 |
| ⭐⭐⭐ 中 | 完整 AS 公式 | 对标原始文献 |
| ⭐⭐⭐⭐ 低 | 多品种对冲 | 工程量大，时间不够别做 |

---

*优先做前两个，两周内能完成，面试说服力翻倍。*
