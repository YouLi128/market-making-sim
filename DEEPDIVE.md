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

## ~~4. 完整的 AS 理论公式~~ ✅ 已完成

原论文公式实现，解析标定法自动求解 (γ, κ)：
- 给定目标价差 $50 和目标库存偏移 $25，一次求解无需手调参数
- σ 用真实数据实现波动率（$/√step）
- 成交概率 P = exp(-κ·dist)，κ=0.040，成交率 13.5%（简化版 ~30%）

结果对比：精确 AS 成交次数更少（677 vs 857），但每笔价差收益更高（更宽价差），
Inventory P&L 更接近零（+$1.65 vs -$），最终 P&L +$240 vs +$188。
详见 `simulator/exact_as.py`、`run_exact_as.py`。

---

## ~~5. P&L 归因精确化 / Exact P&L Attribution~~ ✅ 已完成

每笔成交精确归因：买单 `(mid − bid) × lot`，卖单 `(ask − mid) × lot`。
库存 P&L 直接计算 `Σ inventory × Δmid`（不再用残差法）。
恒等式 `spread_pnl + inventory_pnl == mtm_pnl` 对所有模型精确成立（误差 < $0.01）。
Baseline 误差 $0；AS 误差约 −$0.81（−0.2%），因 reservation price 使 `mid − bid = δ + γ·q·τ ≠ δ`。
AS 系列模型新增 `spread_pnl_approx` 列供对比。详见 `run_exact_attribution.py`。

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
| ~~⭐ 最高~~ | ~~接入真实数据~~ | ✅ 已完成 |
| ~~⭐ 最高~~ | ~~Monte Carlo 回测~~ | ✅ 已完成 |
| ~~⭐⭐ 高~~ | ~~VPIN 毒性检测~~ | ✅ 已完成 |
| ~~⭐⭐ 高~~ | ~~GARCH 波动率~~ | ✅ 已完成 |
| ~~⭐⭐⭐ 中~~ | ~~精确 P&L 归因~~ | ✅ 已完成 |
| ~~⭐⭐⭐ 中~~ | ~~完整 AS 公式~~ | ✅ 已完成 |
| ⭐⭐⭐⭐ 低 | 多品种对冲 | 工程量大，时间不够别做 |

---

*全部计划方向已完成。唯一剩余扩展：多品种对冲（BTC + ETH）。*
