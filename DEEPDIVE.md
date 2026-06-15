# Deep Dive — What's Worth Improving
# 深挖方向

> 按优先级排列，越靠前对面试/论文价值越高。

---

## ~~1. 接入真实数据 / Real Data Integration~~ ✅ 已完成

Binance 公开 API，BTC/USDT 1分钟数据，支持指定日期或拉最近 N 天。
7天真实数据回测：四个模型全部每天盈利，Phase 4 七天总 P&L +$2,963。
详见 `simulator/data_loader.py`、`run_real_data.py`、`run_realdata_backtest.py`。

---

## ~~2. 升级毒性检测 — VPIN~~ ✅ 已完成

基于成交量的 VPIN 实现，BVC 分类每根 K 线买卖量，滚动 50 个 bucket 平均。
真实数据结果：VPIN 更平滑、库存标准差从 0.067 → 0.058，P&L +$333 → +$389。
详见 `simulator/vpin.py`、`run_vpin.py`。
参考：Easley et al. (2012), *Flow Toxicity and Liquidity in a High Frequency World*

---

## ~~3. 波动率预测 — GARCH~~ ✅ 已完成

GARCH(1,1) 拟合真实 BTC 数据，条件方差实时更新 AS 模型的 gamma 和 delta_0。
vol ratio = σ_garch / σ_baseline，高波动时自动加宽价差、加强库存偏移。
注意：GARCH 在突发跳空行情前可能低估波动率，单日表现不如固定 σ，但跨多日更稳定。
详见 `simulator/garch.py`、`run_garch.py`。

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

## ~~7. 回测框架 / Backtesting Framework~~ ✅ 已完成

500 条 Monte Carlo 路径，跨路径 Sharpe：Baseline 1.09 → Phase 4 3.19。
详见 `run_montecarlo.py` 和 `montecarlo_results.png`。

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
| ~~⭐⭐ 高~~ | ~~Monte Carlo 回测~~ | ✅ 已完成 |
| ⭐⭐⭐⭐ 低 | 多品种对冲 | 工程量大，时间不够别做 |

---

*接下来优先：接入真实数据 + VPIN。*
