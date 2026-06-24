# Market Making Simulator — NUS MComp GT Capstone
# 做市商模拟器 — NUS 计算机硕士毕业项目

> **Last updated / 最后更新:** 2026-06-24 (Multi-asset BTC+ETH added)  
> **Stack / 技术栈:** Python · pandas · numpy · matplotlib · requests · arch  
> **GitHub:** https://github.com/YouLi128/market-making-sim

---

## Project Goal / 项目目标

**EN:** Build a market making simulator on real-time price data, starting from a naive fixed-spread baseline and progressively improving toward the Avellaneda-Stoikov (AS) model. The goal is to demonstrate, step by step, *why* each improvement matters — inventory drift, adverse selection, and risk regime — using BTC as the primary asset (24/7, high vol, inventory risk visible immediately).

**中文:** 在真实价格数据上构建一个做市商模拟器，从最简单的固定价差基线策略出发，逐步演进到 Avellaneda-Stoikov (AS) 最优做市模型。目的是用 BTC 作为标的（24小时交易、高波动、库存风险直观可见），一步步展示每次改进的必要性——库存漂移、逆向选择、风险控制。

---

## Roadmap / 阶段计划

| Phase | 阶段 | 内容 | Status |
|-------|------|------|--------|
| **1** | 基线做市商 | Fixed spread δ around mid · track cash / inventory / MtM P&L | ✅ Done |
| **2** | Avellaneda-Stoikov 模型 | Dynamic reservation price · optimal spread based on inventory & volatility | ✅ Done |
| **3** | 逆向选择检测 | Detect informed order flow · widen spread on toxic flow | ✅ Done |
| **4** | 风险控制 | Inventory hard limits · volatility regime switching | ✅ Done |
| **5** | P&L 归因分析 | Decompose P&L → spread capture / inventory risk / adverse selection loss | ✅ Done |
| **6** | Monte Carlo 回测 | 500 条路径统计验证，Sharpe ratio 对比 | ✅ Done |
| **7** | 真实数据接入 | Binance API · 7天真实 BTC 回测 | ✅ Done |
| **8** | VPIN 逆向选择 | 成交量同步毒性检测，替换简单价格方向法 | ✅ Done |
| **9** | GARCH 动态波动率 | GARCH(1,1) 实时预测 σ，动态更新 AS 公式参数 | ✅ Done |
| **10** | 精确 AS 公式 | 原论文公式 + 解析标定，对标 Avellaneda & Stoikov (2008) | ✅ Done |
| **11** | 精确 P&L 归因 | Per-fill exact decomposition · identity spread+inv==MtM verified | ✅ Done |
| **12** | 多品种对冲 | BTC + ETH 联合做市 · 跨资产组合方差最小化 · 库存相关性 −0.875 | ✅ Done |

---

## Progress Log / 进度记录

### 2026-06-24 — Multi-Asset Market Making Complete (Phase 12)

**EN:** Extended the AS model to simultaneous BTC + ETH market making with joint portfolio-variance reservation prices. The cross-asset term ρ·q_ETH·σ_BTC·σ_ETH modifies each asset's quote skew based on the other asset's inventory: when BTC is long and ETH is short (a natural hedge under positive correlation), both reservation prices shift less aggressively — the model recognises the hedge and holds both positions more patiently. Compared to two independent single-asset AS models (ρ=0.85, seed=42): portfolio risk −23.2%, BTC-ETH inventory correlation −0.875 (hedged) vs +0.947 (both trend together), total MtM P&L +$710 vs +$284, zero drawdown vs −$48. This is the same logic used by professional market makers (Wintermute, Jump): manage inventory at the portfolio level, not per-asset.

**中文:** 将 AS 模型扩展到 BTC + ETH 双资产联合做市，使用组合方差梯度计算每个资产的保留价格：`r_BTC = mid_BTC − γ·τ·(q_BTC·σ_BTC² + ρ·q_ETH·σ_BTC·σ_ETH)`，`r_ETH` 对称。交叉项的含义：BTC 多头 + ETH 空头时，两个保留价格的偏移量都减弱——模型识别到对冲关系，不再急于平仓。对比两个独立单资产 AS 模型（ρ=0.85，seed=42）：组合风险降低 23.2%，BTC-ETH 库存相关性 −0.875（对冲）vs +0.947（同向漂移），总 MtM P&L +$710 vs +$284，最大回撤 $0 vs −$48。新增 `simulator/multi_asset_mm.py`、`run_multi_asset.py`、`generate_correlated_btc_eth()` 和 `plot_multi_asset()`。

---

### 2026-06-24 — Exact P&L Attribution Complete

**EN:** Replaced the approximate spread P&L formula (`δ × lot × n_fills`) with per-fill exact attribution across all five simulator classes. Buy fills: `spread_gain = (mid − bid) × lot`; sell fills: `spread_gain = (ask − mid) × lot`. Inventory P&L is now tracked directly as `Σ inventory × Δmid` each step rather than as a residual. The key identity `spread_pnl + inventory_pnl == mtm_pnl` holds exactly (verified to within $0.01). AS-based models also expose a `spread_pnl_approx` column for comparison. Baseline error is exactly $0 since `mid − bid = δ` always. AS error is ~−$0.81 (−0.2%) on seed=7: the reservation price skew makes `(mid − bid) = δ + γ·q·τ ≠ δ` when inventory is non-zero, so the approx slightly overestimates spread P&L.

**中文:** 将五个模拟器类中近似价差公式（`δ × lot × n_fills`）全部替换为逐笔精确归因。买单成交：`spread_gain = (mid − bid) × lot`；卖单成交：`spread_gain = (ask − mid) × lot`。库存 P&L 改为每步直接计算 `Σ inventory × Δmid`，不再用残差法。恒等式 `spread_pnl + inventory_pnl == mtm_pnl` 对所有模型精确成立（误差 < $0.01，已验证）。AS 系列模型同时保留 `spread_pnl_approx` 列供对比分析。基线误差为 $0（`mid − bid = δ` 恒成立）；AS 误差约 −$0.81（−0.2%）——reservation price 偏移使 `mid − bid = δ + γ·q·τ ≠ δ`，导致近似法略微高估价差收益。新增 `run_exact_attribution.py` 入口脚本和 `plot_exact_attribution()` 可视化函数。

---

### 2026-06-16 — Exact AS Formula Complete

**EN:** Implemented the original Avellaneda-Stoikov (2008) formulas without simplification, with an analytical calibration method that solves for (γ, κ) given two intuitive targets: target half-spread at session start and target reservation price skew at mid-session. σ is estimated from real BTC price data per-step. Key difference from simplified AS: fill probability = exp(-κ·dist) with κ=0.040 gives 13.5% fill rate at $50 distance vs simplified model's ~30%. Result: exact AS trades less (677 vs 857 fills) but earns similar spread P&L with tighter inventory control. The `calibrate_as_params()` function makes the model fully parameter-free given human-interpretable targets.

**中文:** 实现原始 AS 论文公式，并提供解析标定函数 `calibrate_as_params()`——给定直观的目标价差和目标库存偏移，自动解出 (γ, κ)，无需手动调参。σ 从真实 BTC 数据估算（实现波动率，$/√step）。与简化版 AS 的核心区别：成交概率 = exp(-κ·dist)，κ=0.040，$50 处成交率 13.5%（简化版约 30%）。精确版成交次数更少但价差收益相当，库存控制更紧。

---

### 2026-06-15 — VPIN + GARCH Complete

**EN:** Two signal upgrades added on top of the existing pipeline. (1) VPIN replaces the simple rolling up-fraction toxicity score — uses Bulk Volume Classification to estimate buy/sell volume per bar, accumulates into fixed-size buckets, and computes rolling |OI|/V as the toxicity signal. On real data: inventory std reduced 0.067→0.058, P&L improved +$333→+$389. (2) GARCH(1,1) replaces the fixed σ=0.80 in the AS reservation price and spread formulas — fits on the full price series, then scales gamma and delta_0 by vol_ratio = σ_garch/σ_baseline at each step. Known limitation: GARCH is lagged on sudden jumps, performing worse on gap-up/down days but better over multi-day periods.

**中文:** 两个信号升级。(1) VPIN 替换简单价格方向毒性——BVC 估算每根 K 线买卖量，固定大小 bucket 积累，滚动 |OI|/V 作为毒性分数。真实数据：库存标准差 0.067→0.058，P&L +$333→+$389。(2) GARCH(1,1) 替换 AS 公式里的固定 σ——拟合整段价格序列，每步用 vol_ratio = σ_garch/σ_baseline 缩放 gamma 和 delta_0。已知局限：GARCH 对突发跳空滞后，跳空当天不如固定 σ，但多日累积表现更稳定。

---

### 2026-06-11 — Monte Carlo Backtest Complete

**EN:** 500-path Monte Carlo backtest implemented across all four models. Key result: Phase 4 achieves Sharpe ratio 4.28 vs baseline 1.46 — nearly 3× more consistent P&L across different market conditions. Baseline has higher mean P&L (+$448) but std of $306, meaning it's essentially gambling on price direction. Phase 4 mean +$357 with std only $83, making it predictable and robust. AS also shows a strong Sharpe of 3.31, confirming that inventory management alone (without risk controls) already dramatically improves consistency. This is the statistical proof that each improvement is meaningful, not just luck on one price path.

**中文:** 500 条路径 Monte Carlo 回测完成。核心结论：Phase 4 的跨路径 Sharpe 为 4.28，基线为 1.46，相差近 3 倍。基线均值 P&L 虽高（+$448），但标准差 $306，本质上是在赌价格方向。Phase 4 均值 +$357，标准差只有 $83，稳定可预期。AS 模型 Sharpe 达到 3.31，说明仅靠库存管理已能大幅提升稳定性。这是统计层面的证明——每个阶段的改进是真实有效的，而不是某条路径的运气。

---

### 2026-06-10 — Phase 5 Complete (All phases done)

**EN:** Performance attribution complete. All four models run on the same regime-switching price path (seed=7). Key results: Baseline earns highest spread P&L (+$432) but has widest inventory swings (std=0.10 BTC); AS tightens inventory (std=0.043) while maintaining solid spread P&L (+$367); Phase 3 recovers spread P&L (+$376) while reducing inventory std further (0.046); Phase 4 achieves tightest inventory (std=0.036) with good spread capture (+$372) and best inventory P&L control (-$17). The three-panel chart (P&L curves / inventory curves / decomposition bars) is the thesis conclusion figure.

**中文:** 归因分析完成，四个模型跑在同一条价格路径上。核心结论：基线价差收益最高但库存漂移最大；AS 大幅收紧库存（标准差 0.043 BTC）同时维持良好价差收益（+$367）；Phase 3 进一步压缩库存波动；Phase 4 库存最稳定（标准差 0.036 BTC），价差收益接近基线。三格对比图（P&L 曲线/库存曲线/分解柱状图）是论文结论页的核心图表。

---

### 2026-06-10 — Phase 4 Complete

**EN:** Risk controls implemented with three stacking layers: (1) Hard inventory limit — if inventory exceeds ±0.10 BTC, the over-limit direction is blocked entirely; (2) Volatility regime detection — rolling 30-step realized vol compared to 1.2× baseline threshold, spread multiplied by 1.3× when high vol detected (19% of session); (3) Emergency liquidation — BTC-appropriate: triggers when same-side inventory held >120 steps OR inventory floating loss exceeds −$150 (not session-close based, since BTC trades 24/7). Result vs Phase 3: max inventory reduced 0.15→0.11 BTC, inventory std dev −25% (0.067→0.050 BTC), final P&L +$321 vs +$306, max drawdown −$0.86 vs −$0.86.

**中文:** 风险控制完成，三层叠加保护：(1) 库存硬限制——库存超过 ±0.10 BTC 时自动封锁该方向报价；(2) 波动率切换——滚动窗口实现波动率超过基准 1.2 倍时价差乘以 1.3 倍（约 19% 的时间触发）；(3) 紧急清仓——针对 BTC 24/7 特性改进：触发条件为"同方向持仓超过 120 步"或"库存浮动亏损超过 -$150"（不再依赖收盘时间）。对比 Phase 3：最大库存 0.15→0.11 BTC，库存标准差降 25%（0.067→0.050 BTC），最终 P&L +$321 vs +$306，最大回撤 $0.86 vs $0.86。

---

### 2026-06-10 — Phase 3 Complete

**EN:** Adverse selection detection implemented. Added regime-switching price generator (normal random walk ↔ trending periods that simulate informed traders). Toxicity score tracks directional consistency of recent price moves — when price keeps going one way, informed flow is likely active. When toxicity exceeds threshold, spread widens automatically: `delta_eff = delta_base × (1 + 2.0 × toxicity)`. Fill probability uses absolute exponential decay from mid, so wider spreads = fewer fills = less exposure to informed flow. Result vs Phase 2 AS: P&L +$306 vs +$417, inventory std dev 0.067 vs 0.063 BTC, max drawdown −$0.86 vs −$2.46. Note: on this specific regime-switching path Phase 3 shows lower total P&L than Phase 2 because toxicity detection reduces fills more than the adverse selection it avoids. The advantage of Phase 3 is clearer in the attribution comparison (seed=7) and statistically across 500 Monte Carlo paths.

**中文:** 逆向选择检测完成。新增两部分：(1) 带趋势阶段的价格生成器——模拟知情交易者活跃时的市场（正常随机游走 ↔ 单边漂移阶段）；(2) 毒性分数实时检测——滚动窗口统计最近价格方向一致性，持续单边运动 = 知情流活跃 = 自动加宽价差。价差加宽后报价远离中间价，成交概率指数下降，减少在知情流面前的暴露。对比 Phase 2 AS：P&L +$306 vs +$417，库存标准差 0.067 vs 0.063 BTC，最大回撤 $0.86 vs $2.46。注意：在这条特定路径上 Phase 3 总 P&L 低于 Phase 2，因为毒性检测减少的成交量多于它避免的逆向选择损失。Phase 3 的优势在 attribution 对比图（seed=7）和 500 条 Monte Carlo 路径的统计结果中更明显。

---

### 2026-06-10 — Phase 2 Complete

**EN:** Avellaneda-Stoikov model implemented. Two mechanisms added: (1) reservation price `r = mid − γ·q·τ` skews quotes away from mid proportionally to inventory — when long, the ask gets cheaper and fills more readily; (2) time-decaying spread narrows from $50 → $5 as the session ends, increasing urgency to flatten inventory. Fill probability uses absolute exponential decay from mid: `p = exp(−k × dist)`, calibrated so p=0.30 at $50. Result on same price path: max inventory reduced (0.47 → 0.12 BTC), inventory std dev reduced 49% (0.122 → 0.063 BTC). Note: on this particular upward-trending path, baseline outperforms in total P&L because it got lucky holding long inventory during a BTC rally — AS intentionally avoids that directional bet, which hurts on uptrends but protects on downtrends.

**中文:** Avellaneda-Stoikov 模型完成。新增两个机制：(1) 保留价格 `r = mid − γ·q·τ` 根据库存量偏离中间价——持多头时卖价变便宜、更容易成交，库存自然回归；(2) 价差随时间收窄（$50 → $5），临近收盘时做市商被迫更积极地平仓。成交概率使用绝对指数衰减 `p = exp(−k × dist)`（$50 处成交率 30%），与 Phase 3/4 及原论文一致。相同价格路径上的结果：最大库存大幅压缩（0.47 → 0.12 BTC），库存标准差下降 49%（0.122 → 0.063 BTC）。注意：本次路径 BTC 整体上涨，基线因为碰巧积累多头而赚得更多——这是运气，不是策略优势；AS 主动规避了这个方向性赌注，在上涨路径中吃亏，但在下跌路径中得到保护。

---

### 2026-06-10 — Phase 1 Complete

**EN:** Baseline market maker implemented and tested. Key result: with fixed δ=$50, 1 day of 1-min BTC bars, the maker captures ~$0.30/min in spread P&L, but inventory drifts randomly (max 0.47 BTC) and can fully erase spread gains. This is the exact failure mode that motivates the AS model.

**中文:** 基线做市商完成并通过测试。核心结论：固定价差 δ=$50，模拟 1 天 BTC 分钟数据，每分钟稳定捕捉约 $0.30 的价差收益，但库存随机漂移（最大 0.47 BTC），方向性敞口可以完全吃掉价差利润。这正是 AS 模型要解决的核心问题。

---

## File Structure / 文件结构

```
market-making-sim/
│
├── PROJECT.md                ← 本文档 / this document
│
├── simulator/
│   ├── __init__.py
│   ├── data_gen.py               ← 合成价格生成 (GBM + regime-switching)
│   ├── data_loader.py            ← Binance API 真实数据 (price + OHLCV)
│   ├── baseline_mm.py            ← Phase 1: 固定价差做市商
│   ├── avellaneda_stoikov.py     ← Phase 2: AS 模型 (支持 GARCH sigma 注入)
│   ├── adverse_selection.py      ← Phase 3: 简单毒性检测 (支持外部信号覆盖)
│   ├── risk_controls.py          ← Phase 4: 风险控制三层保护
│   ├── exact_as.py               ← Phase 10: 原论文精确 AS 公式 + 解析标定
│   ├── vpin.py                   ← Phase 8: VPIN 成交量毒性检测
│   ├── garch.py                  ← Phase 9: GARCH(1,1) 条件波动率
│   └── visualize.py              ← 所有画图函数
│
├── run_baseline.py               ← Phase 1 入口
├── run_as.py                     ← Phase 2 入口
├── run_compare.py                ← Baseline vs AS 对比图
├── run_phase3.py                 ← Phase 3 入口
├── run_phase4.py                 ← Phase 4 入口
├── run_attribution.py            ← Phase 5 归因对比入口
├── run_montecarlo.py             ← Phase 6 Monte Carlo 入口
├── run_real_data.py              ← Phase 7 真实数据单日
├── run_realdata_backtest.py      ← Phase 7 多日回测
├── run_vpin.py                   ← Phase 8 VPIN 入口
├── run_garch.py                  ← Phase 9 GARCH 入口
├── run_exact_as.py               ← Phase 10 精确 AS 入口
├── run_exact_attribution.py      ← Phase 11 精确归因入口
├── run_multi_asset.py            ← Phase 12 多品种对冲入口
│
├── baseline_results.png          ← Phase 1 输出图
├── as_results.png                ← Phase 2 输出图
├── comparison_results.png        ← Phase 1 vs 2 对比图
├── phase3_results.png            ← Phase 3 输出图
├── phase4_results.png            ← Phase 4 输出图
├── attribution_results.png       ← Phase 5 归因对比图（论文结论图）
├── montecarlo_results.png        ← Monte Carlo 500条路径统计分布图
├── real_data_results.png         ← 真实数据单日运行图
├── realdata_backtest.png         ← 7天真实数据回测图
├── vpin_results.png              ← VPIN vs 简单毒性对比图
├── garch_results.png             ← GARCH vs 固定 σ 对比图
├── exact_as_results.png          ← Phase 10 精确 AS 输出图
├── exact_attribution_results.png ← Phase 11 精确归因对比图
├── multi_asset_results.png       ← Phase 12 多品种对冲对比图
└── requirements.txt              ← numpy · pandas · matplotlib
```

---

## Code Walkthrough / 代码说明

### `simulator/data_gen.py` — 价格模拟器

**EN:** Generates a synthetic BTC price path using **Geometric Brownian Motion (GBM)**. GBM is the standard model for asset prices: the log-return at each step is drawn from a normal distribution parameterised by drift `mu` and volatility `sigma`.

**中文:** 用**几何布朗运动 (GBM)** 生成合成 BTC 价格路径。GBM 是资产价格的标准随机过程：每步对数收益率服从正态分布，由漂移率 `mu` 和波动率 `sigma` 参数化。

**核心公式 / Key formula:**

```
log_return[t] = (μ - σ²/2) · dt  +  σ · √dt · Z[t]       Z ~ N(0,1)
price[t]      = price[0] · exp( Σ log_return[1..t] )
```

**参数说明 / Parameters:**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `S0` | 50,000 | 初始价格 / initial BTC price (USD) |
| `mu` | 0.0 | 年化漂移率 / annualised drift |
| `sigma` | 0.80 | 年化波动率 / annualised vol (80% is realistic for BTC) |
| `n_steps` | 1440 | 时间步数 / number of bars (1440 = 1 day of 1-min bars) |
| `seed` | 42 | 随机种子 / reproducibility seed |

**每步价格标准差 / Per-step std dev:**
```
σ_step = 50,000 × 0.80 × √(1/(365×24×60)) ≈ $55 per minute
```

---

### `simulator/baseline_mm.py` — 基线做市商

**EN:** The core logic. At every timestep the maker posts two quotes symmetrically around the mid price. Fills arrive independently on each side with probability `prob_fill` — this models random uninformed order flow. The key design choice here is that **fill probability is independent of price direction**, so all inventory risk comes purely from holding positions while prices drift.

**中文:** 核心逻辑。每个时间步，做市商在中间价两侧对称报价。每侧以概率 `prob_fill` 独立成交——模拟随机的非知情订单流。关键设计：**成交概率与价格方向无关**，因此所有库存风险纯粹来自持仓期间的价格漂移。

**报价逻辑 / Quote logic:**
```
bid = mid_price - δ     (我们愿意买的价格 / we buy here)
ask = mid_price + δ     (我们愿意卖的价格 / we sell here)
```

**P&L 分解 / P&L decomposition:**

```
MtM P&L      = cash + inventory × mid − initial_cash   ← 总收益
Spread P&L   = δ × lot_size × total_fills              ← 价差收益（稳定正）
Inventory P&L = MtM P&L − Spread P&L                  ← 方向性敞口（波动大）
```

**为什么这样拆 / Why this decomposition matters:**

每次成交，无论买还是卖，我们都从价差里赚 δ。这是做市商的"稳定收益源"。  
库存 P&L 则反映：持有仓位时价格走反了多少。基线策略不管库存，所以这一项完全不受控。AS 模型的目标就是让库存 P&L 接近零。

*Every fill earns δ regardless of direction — that's the maker's structural edge. Inventory P&L captures the cost of holding an unhedged position. The baseline ignores inventory, so this term is uncontrolled. The AS model is designed to drive it toward zero.*

**`Trade` dataclass:** records every fill with `timestamp`, `side`, `price`, `qty` — used for post-trade analysis in later phases.

---

### `simulator/visualize.py` — 可视化

**EN:** Produces a 3-panel figure using matplotlib:

**中文:** 用 matplotlib 生成三格图：

| Panel | 内容 | 说明 |
|-------|------|------|
| 1 | Price + Quotes | 中间价、买一价、卖一价随时间变化 |
| 2 | Inventory | 库存漂移；橙色=多头，蓝色=空头；基线不回归是核心缺陷 |
| 3 | P&L Decomposition | 紫色=总 MtM，绿虚线=价差收益，红点线=库存收益 |

Panel 3 is the most important for the capstone narrative: it shows how spread P&L grows steadily while inventory P&L fluctuates wildly — motivating every subsequent improvement.

Panel 3 是毕业论文叙事中最重要的图：价差收益稳定增长，库存收益剧烈波动，这直接引出后续每一步改进的动机。

---

### `run_baseline.py` — 运行入口

**EN:** CLI wrapper. Parses arguments, calls `generate_btc_price` → `BaselineMarketMaker.run` → `plot_simulation`, prints a summary table, and saves `baseline_results.png`.

**中文:** 命令行入口。解析参数，依次调用价格生成 → 做市商模拟 → 画图，打印汇总表，保存结果图。

**常用命令 / Common commands:**

```bash
# 默认运行 / default run
python run_baseline.py

# 换个随机路径看看 / try a different random path
python run_baseline.py --seed 7

# 更紧的价差，测试成交率影响 / tighter spread
python run_baseline.py --delta 25

# 加入上涨趋势，展示库存爆仓 / add upward drift to show inventory blowup
python run_baseline.py --mu 0.5

# 只保存图片不弹窗 / save PNG without opening window
python run_baseline.py --no-show
```

**所有参数 / All arguments:**

| 参数 | 默认 | 含义 |
|------|------|------|
| `--S0` | 50000 | 初始价格 |
| `--mu` | 0.0 | 年化漂移率 |
| `--sigma` | 0.80 | 年化波动率 |
| `--n-steps` | 1440 | 时间步数 |
| `--delta` | 50.0 | 单侧价差 (USD) |
| `--lot-size` | 0.01 | 每次成交 BTC 数量 |
| `--prob-fill` | 0.30 | 每侧每步成交概率 |
| `--seed` | 42 | 随机种子 |
| `--no-show` | False | 不弹图窗，只存文件 |

---

### `simulator/avellaneda_stoikov.py` — AS 模型

**EN:** The AS model replaces the fixed symmetric quote with two improvements:

**中文:** AS 模型用两个改进替换掉固定对称报价：

**① 保留价格 / Reservation price**

```
r = mid − gamma × q × tau
```

- `q` = 当前库存（BTC）
- `tau` = 剩余时间占比 τ ∈ [0, 1]
- `gamma` = 风险厌恶系数（$/BTC），默认 50

持多头（q > 0）→ r < mid → 卖价靠近中间价 → 更容易成交 → 库存主动回归零  
持空头（q < 0）→ r > mid → 买价靠近中间价 → 更容易成交 → 库存主动回归零

**② 动态价差 / Time-decaying spread**

```
delta(tau) = delta_0 × tau + delta_min
```

开盘时 δ≈$50（和基线相同），收盘前收窄到 $5——临近结束时做市商接受更低利润来尽快平仓。

**③ 成交概率响应报价位置 / Fill probability responds to quote placement**

```
p_fill = prob_fill × exp(k × (delta(tau) − |quote_dist_from_mid|))
```

- 报价比当前价差更靠近中间价：p > prob_fill（更容易成交）
- 报价比当前价差更远：p < prob_fill（更难成交）
- 库存为零时报价对称：p = prob_fill（和基线完全一样）

这第三点是 AS 模型能让库存回归的真正原因——不是报价位置本身，而是报价位置改变了两侧的成交概率，让卖出比买入更容易发生（当持多时）。

*This third mechanism is what actually drives inventory mean-reversion: the skewed quotes create an asymmetry in fill rates that systematically pushes inventory back toward zero.*

**参数说明 / Parameters:**

| 参数 | 默认 | 含义 |
|------|------|------|
| `gamma` | 50.0 | $/BTC：每 BTC 库存在 τ=1 时的报价偏移量 |
| `delta_0` | 45.0 | 开盘时额外价差（加上 delta_min = $50 总计） |
| `delta_min` | 5.0 | 收盘前最低价差 |
| `T` | 1440 | 总步数（会话长度） |
| `fill_k` | 0.024 | 成交概率对距离的敏感度（1/$） |

**常用命令 / Common commands:**

```bash
python run_as.py                        # 默认运行
python run_as.py --seed 7               # 换个路径
python run_as.py --gamma 100            # 更激进的库存偏移
python run_compare.py                   # Baseline vs AS 对比
python run_compare.py --mu 0.5          # 上涨趋势下的对比（基线占优）
python run_compare.py --mu -0.5         # 下跌趋势下的对比（AS 占优）
```

---

## Key Concepts / 核心概念

### 什么是做市商 / What is a Market Maker?

做市商同时在买卖两侧挂单，赚取买卖价差。它不预测价格方向，而是通过高频双边成交积累利润。风险在于：如果库存在价格不利方向积累，方向性亏损会超过价差收益。

*A market maker posts both a bid and an ask simultaneously, earning the spread on each round trip. It does not predict direction — it profits from high-frequency two-sided flow. The risk: if inventory accumulates while the price moves adversely, directional losses exceed spread gains.*

### 为什么选 BTC / Why BTC?

- 24/7 交易，没有开收盘跳空 / No overnight gaps
- 波动率高（~80%/yr），库存风险极其显著 / High vol makes inventory risk obvious immediately
- 价差宽，做市利润直观 / Wide spreads make P&L easier to understand
- 数据容易获取 / Data readily available

### 基线策略的核心缺陷 / Why the Baseline Fails

1. **库存不回归 / No inventory mean-reversion** — 报价始终对称，不会因为持仓多就把买价降低来吸引对手方。
2. **价差固定 / Fixed spread** — 市场剧烈波动时也不加宽，逆向选择损失放大。
3. **无风险控制 / No risk controls** — 库存可以无限累积。

这三点对应后续三个改进阶段，是毕业论文的论证主线。

*These three failure modes map directly onto the next three improvement phases — they are the narrative backbone of the capstone.*

---

## What's Next / 下一步

**Phase 1–12 全部完成，项目可直接作为毕设提交。**

---

## Known Limitations / 已知局限性

代码审查后发现的设计问题，供论文写作参考。

**~~1. Phase 2 与 Phase 3/4 的成交模型不一致~~ ✅ 已修复**

Phase 2 已统一为绝对衰减 `p = exp(−k × dist)`，与 Phase 3/4 及原论文一致。
修复后 Phase 2 Sharpe 从 ~1 提升至 3.31（Monte Carlo），Phase 2→3 的 P&L 对比现在纯粹反映毒性检测本身的效果。

**2. tau 对 BTC 24/7 的适用性**

Reservation price 和价差公式仍使用 `tau`（会话剩余比例）。在单次模拟（1440步=1天）中合理，但若做市商持续运行多天，需要将 tau 改为滚动窗口重置机制，否则价差永远接近最小值。

**3. Spread P&L 是近似值**

每笔成交的价差收益记录为 `delta × lot_size`，忽略了 reservation price 与 mid 偏差带来的影响。实际上买入时的价差收益和卖出时略有不同。在库存小的情况下误差可忽略。

**4. 毒性检测滞后**

滚动窗口（默认30步）只能在知情流已经持续一段时间后才能检测到。理论上前30步内已经受到了完整的逆向选择损失，保护措施才刚刚启动。

---

*This document is updated manually at the end of each phase. / 本文档在每个阶段结束后手动更新。*
