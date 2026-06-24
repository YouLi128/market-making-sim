import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


def plot_simulation(results: pd.DataFrame, title: str = "Baseline Market Maker") -> plt.Figure:
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # --- panel 1: price + quotes ------------------------------------------
    ax1.plot(results.index, results["mid_price"], color="steelblue", lw=1.2, label="Mid price")
    ax1.plot(results.index, results["bid"], color="forestgreen", lw=0.6, alpha=0.55, label="Bid (mid − δ)")
    ax1.plot(results.index, results["ask"], color="firebrick", lw=0.6, alpha=0.55, label="Ask (mid + δ)")
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.set_title("Price and Quotes", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: inventory ------------------------------------------------
    ax2.plot(results.index, results["inventory"], color="darkorange", lw=1.2)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.fill_between(results.index, results["inventory"], 0,
                     where=results["inventory"] > 0, alpha=0.25, color="darkorange", label="Long")
    ax2.fill_between(results.index, results["inventory"], 0,
                     where=results["inventory"] < 0, alpha=0.25, color="steelblue", label="Short")
    ax2.set_ylabel("Inventory (BTC)", fontsize=9)
    ax2.set_title("Inventory Drift  (baseline does not skew quotes — key weakness)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: P&L decomposition ----------------------------------------
    ax3.plot(results.index, results["mtm_pnl"],
             color="purple", lw=1.4, label="Total MtM P&L", zorder=3)
    ax3.plot(results.index, results["spread_pnl"],
             color="forestgreen", lw=1.0, linestyle="--", label="Spread P&L (captured edge)")
    ax3.plot(results.index, results["inventory_pnl"],
             color="firebrick", lw=1.0, linestyle=":", label="Inventory P&L (directional risk)")
    ax3.axhline(0, color="black", lw=0.8, linestyle="--")
    ax3.fill_between(results.index, results["mtm_pnl"], 0,
                     where=results["mtm_pnl"] >= 0, alpha=0.12, color="green")
    ax3.fill_between(results.index, results["mtm_pnl"], 0,
                     where=results["mtm_pnl"] < 0, alpha=0.12, color="red")
    ax3.set_ylabel("P&L (USD)", fontsize=9)
    ax3.set_title("P&L Decomposition  (spread capture vs inventory risk)", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_as_simulation(results: pd.DataFrame, title: str = "Avellaneda-Stoikov Market Maker") -> plt.Figure:
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # --- panel 1: price + reservation price + skewed quotes ---------------
    ax1.plot(results.index, results["mid_price"], color="steelblue", lw=1.2, label="Mid price")
    ax1.plot(results.index, results["reservation_price"], color="gold", lw=1.0,
             linestyle="--", label="Reservation price r")
    ax1.plot(results.index, results["bid"], color="forestgreen", lw=0.6, alpha=0.55, label="Bid")
    ax1.plot(results.index, results["ask"], color="firebrick", lw=0.6, alpha=0.55, label="Ask")
    ax1.fill_between(results.index, results["mid_price"], results["reservation_price"],
                     alpha=0.15, color="gold", label="Skew (mid − r)")
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.set_title("Price, Reservation Price, and Skewed Quotes", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=5)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: inventory — should mean-revert vs baseline --------------
    ax2.plot(results.index, results["inventory"], color="darkorange", lw=1.2)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.fill_between(results.index, results["inventory"], 0,
                     where=results["inventory"] > 0, alpha=0.25, color="darkorange", label="Long")
    ax2.fill_between(results.index, results["inventory"], 0,
                     where=results["inventory"] < 0, alpha=0.25, color="steelblue", label="Short")
    ax2.set_ylabel("Inventory (BTC)", fontsize=9)
    ax2.set_title("Inventory  (AS skews quotes to push inventory back toward zero)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: P&L decomposition ---------------------------------------
    ax3.plot(results.index, results["mtm_pnl"],
             color="purple", lw=1.4, label="Total MtM P&L", zorder=3)
    ax3.plot(results.index, results["spread_pnl"],
             color="forestgreen", lw=1.0, linestyle="--", label="Spread P&L (captured edge)")
    ax3.plot(results.index, results["inventory_pnl"],
             color="firebrick", lw=1.0, linestyle=":", label="Inventory P&L (directional risk)")
    ax3.axhline(0, color="black", lw=0.8, linestyle="--")
    ax3.fill_between(results.index, results["mtm_pnl"], 0,
                     where=results["mtm_pnl"] >= 0, alpha=0.12, color="green")
    ax3.fill_between(results.index, results["mtm_pnl"], 0,
                     where=results["mtm_pnl"] < 0, alpha=0.12, color="red")
    ax3.set_ylabel("P&L (USD)", fontsize=9)
    ax3.set_title("P&L Decomposition", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_comparison(
    baseline: pd.DataFrame,
    as_model: pd.DataFrame,
    title: str = "Baseline vs Avellaneda-Stoikov — Same Price Path",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.93, bottom=0.06)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # --- panel 1: inventory comparison ------------------------------------
    ax1.plot(baseline.index, baseline["inventory"],
             color="firebrick", lw=1.2, alpha=0.85, label="Baseline (drifts)")
    ax1.plot(as_model.index, as_model["inventory"],
             color="steelblue", lw=1.2, label="Avellaneda-Stoikov (mean-reverts)")
    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    ax1.fill_between(as_model.index, as_model["inventory"], 0, alpha=0.12, color="steelblue")

    b_std = baseline["inventory"].std()
    a_std = as_model["inventory"].std()
    ax1.set_title(
        f"Inventory  —  Baseline std={b_std:.3f} BTC   AS std={a_std:.3f} BTC   "
        f"(reduction: {(1 - a_std/b_std)*100:.0f}%)",
        fontsize=10,
    )
    ax1.set_ylabel("Inventory (BTC)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: spread comparison (baseline fixed vs AS dynamic) --------
    ax2.plot(baseline.index, [50.0] * len(baseline),
             color="firebrick", lw=1.2, linestyle="--", label=f"Baseline half-spread (fixed $50)")
    ax2.plot(as_model.index, as_model["half_spread"],
             color="steelblue", lw=1.0, label="AS half-spread δ* (dynamic)")
    ax2.set_ylabel("Half-spread (USD)", fontsize=9)
    ax2.set_title("Quote Spread  —  Baseline fixed vs AS dynamic (widens with vol, collapses at end)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: total P&L comparison ------------------------------------
    ax3.plot(baseline.index, baseline["mtm_pnl"],
             color="firebrick", lw=1.2, alpha=0.85, label="Baseline MtM P&L")
    ax3.plot(as_model.index, as_model["mtm_pnl"],
             color="steelblue", lw=1.2, label="AS MtM P&L")
    ax3.axhline(0, color="black", lw=0.8, linestyle="--")
    ax3.fill_between(as_model.index, as_model["mtm_pnl"], baseline["mtm_pnl"],
                     where=as_model["mtm_pnl"] >= baseline["mtm_pnl"],
                     alpha=0.15, color="steelblue", label="AS outperforms")
    ax3.fill_between(as_model.index, as_model["mtm_pnl"], baseline["mtm_pnl"],
                     where=as_model["mtm_pnl"] < baseline["mtm_pnl"],
                     alpha=0.15, color="firebrick", label="Baseline outperforms")
    ax3.set_ylabel("P&L (USD)", fontsize=9)
    ax3.set_title("Total MtM P&L Comparison", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_phase3(
    results: pd.DataFrame,
    regimes: pd.Series,
    as_results: pd.DataFrame = None,
    title: str = "Phase 3 — Adverse Selection Detection",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 1, hspace=0.45, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    def shade_regimes(ax):
        in_trend, start = False, None
        for t, r in regimes.items():
            if r == 1 and not in_trend:
                in_trend, start = True, t
            elif r == 0 and in_trend:
                ax.axvspan(start, t, alpha=0.12, color="firebrick", zorder=0)
                in_trend = False
        if in_trend:
            ax.axvspan(start, regimes.index[-1], alpha=0.12, color="firebrick", zorder=0)

    # --- panel 1: price + regime shading ----------------------------------
    ax1.plot(results.index, results["mid_price"], color="steelblue", lw=1.2, label="Mid price")
    ax1.plot(results.index, results["reservation_price"], color="gold", lw=0.8,
             linestyle="--", alpha=0.8, label="Reservation price")
    shade_regimes(ax1)
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.set_title("Price  (red shading = trending regime — informed flow active)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: toxicity score ------------------------------------------
    ax2.plot(results.index, results["toxicity"], color="firebrick", lw=1.0, label="Toxicity score")
    ax2.axhline(0.4, color="darkorange", lw=0.8, linestyle="--", label="Threshold 0.4")
    ax2.fill_between(results.index, results["toxicity"], 0.4,
                     where=results["toxicity"] > 0.4, alpha=0.25, color="firebrick",
                     label="High toxicity")
    shade_regimes(ax2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Toxicity [0–1]", fontsize=9)
    ax2.set_title("Toxicity Score  —  spikes during trend regime, triggers spread widening", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: spread comparison ---------------------------------------
    ax3.plot(results.index, results["delta_eff"],
             color="firebrick", lw=1.0, label="Effective spread δ_eff (toxicity-adjusted)")
    ax3.plot(results.index, results["delta_base"],
             color="steelblue", lw=0.8, linestyle="--", label="Base AS spread δ_base")
    shade_regimes(ax3)
    ax3.set_ylabel("Half-spread (USD)", fontsize=9)
    ax3.set_title("Quote Spread  —  widens automatically during toxic periods", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # --- panel 4: P&L comparison ------------------------------------------
    ax4.plot(results.index, results["mtm_pnl"],
             color="firebrick", lw=1.2, label="Phase 3 (adverse selection aware)")
    if as_results is not None:
        ax4.plot(as_results.index, as_results["mtm_pnl"],
                 color="steelblue", lw=1.0, linestyle="--", alpha=0.7,
                 label="Phase 2 AS (no detection)")
    ax4.axhline(0, color="black", lw=0.8, linestyle="--")
    shade_regimes(ax4)
    ax4.set_ylabel("P&L (USD)", fontsize=9)
    ax4.set_title("MtM P&L  —  Phase 3 reduces drawdowns during informed flow", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_phase4(
    p4: pd.DataFrame,
    p3: pd.DataFrame,
    regimes: pd.Series,
    max_inventory: float = 0.30,
    title: str = "Phase 4 — Risk Controls",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 1, hspace=0.45, top=0.93, bottom=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    def shade_regimes(ax):
        in_trend, start = False, None
        for t, r in regimes.items():
            if r == 1 and not in_trend:
                in_trend, start = True, t
            elif r == 0 and in_trend:
                ax.axvspan(start, t, alpha=0.10, color="firebrick", zorder=0)
                in_trend = False
        if in_trend:
            ax.axvspan(start, regimes.index[-1], alpha=0.10, color="firebrick", zorder=0)

    def shade_high_vol(ax, results):
        in_hv, start = False, None
        for t, v in results["high_vol"].items():
            if v == 1 and not in_hv:
                in_hv, start = True, t
            elif v == 0 and in_hv:
                ax.axvspan(start, t, alpha=0.15, color="purple", zorder=0)
                in_hv = False
        if in_hv:
            ax.axvspan(start, results.index[-1], alpha=0.15, color="purple", zorder=0)

    # --- panel 1: inventory — Phase 3 vs Phase 4 with hard limit lines ----
    ax1.plot(p3.index, p3["inventory"],
             color="steelblue", lw=1.0, alpha=0.7, label="Phase 3 (no hard limit)")
    ax1.plot(p4.index, p4["inventory"],
             color="firebrick", lw=1.2, label="Phase 4 (hard limit ± {:.2f} BTC)".format(max_inventory))
    ax1.axhline(+max_inventory, color="black", lw=1.0, linestyle="--", label=f"+limit = +{max_inventory} BTC")
    ax1.axhline(-max_inventory, color="black", lw=1.0, linestyle="--", label=f"−limit = −{max_inventory} BTC")
    ax1.axhline(0, color="grey", lw=0.5)
    shade_regimes(ax1)
    ax1.set_ylabel("Inventory (BTC)", fontsize=9)
    ax1.set_title("Inventory  —  Phase 4 hard limit prevents runaway accumulation", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: realised vol vs threshold -------------------------------
    ax2.plot(p4.index, p4["realised_vol"],
             color="purple", lw=1.0, label="Realised vol (annualised)")
    threshold_line = p4["realised_vol"].copy()
    baseline_sigma = 0.80
    vol_threshold  = 1.5
    ax2.axhline(baseline_sigma * vol_threshold, color="darkorange", lw=0.8,
                linestyle="--", label=f"High-vol threshold ({baseline_sigma*vol_threshold:.1f}×)")
    ax2.fill_between(p4.index, p4["realised_vol"], baseline_sigma * vol_threshold,
                     where=p4["realised_vol"] > baseline_sigma * vol_threshold,
                     alpha=0.20, color="purple", label="High-vol regime")
    shade_regimes(ax2)
    ax2.set_ylabel("Annual vol", fontsize=9)
    ax2.set_title("Realised Volatility  —  purple shading = spread multiplied by 1.5×", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: spread layers -------------------------------------------
    ax3.plot(p4.index, p4["delta_final"],
             color="firebrick", lw=1.0, label="Final spread (tox + vol adjusted)")
    ax3.plot(p4.index, p4["delta_tox"],
             color="darkorange", lw=0.8, linestyle="--", label="After toxicity adjustment")
    ax3.plot(p4.index, p4["delta_base"],
             color="steelblue", lw=0.7, linestyle=":", label="Base AS spread")
    shade_high_vol(ax3, p4)
    shade_regimes(ax3)
    ax3.set_ylabel("Half-spread (USD)", fontsize=9)
    ax3.set_title("Spread Layers  —  base → toxicity → vol-adjusted (three-layer protection)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # --- panel 4: P&L comparison ------------------------------------------
    ax4.plot(p3.index, p3["mtm_pnl"],
             color="steelblue", lw=1.0, linestyle="--", alpha=0.7, label="Phase 3 P&L")
    ax4.plot(p4.index, p4["mtm_pnl"],
             color="firebrick", lw=1.2, label="Phase 4 P&L (risk controls)")
    ax4.axhline(0, color="black", lw=0.8, linestyle="--")
    shade_regimes(ax4)
    ax4.set_ylabel("P&L (USD)", fontsize=9)
    ax4.set_title("MtM P&L  —  smoother equity curve with bounded inventory", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_attribution(
    results: dict,
    regimes: pd.Series,
    title: str = "Phase 5 — Performance Attribution",
) -> plt.Figure:
    COLORS = {
        "Baseline": "grey",
        "AS":       "steelblue",
        "Phase3":   "darkorange",
        "Phase4":   "firebrick",
    }
    LABELS = {
        "Baseline": "Phase 1: Baseline",
        "AS":       "Phase 2: AS",
        "Phase3":   "Phase 3: Adverse Selection",
        "Phase4":   "Phase 4: Risk Controls",
    }

    fig = plt.figure(figsize=(14, 13))
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.93, bottom=0.06,
                           height_ratios=[2, 2, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    def shade_regimes(ax):
        in_trend, start = False, None
        for t, r in regimes.items():
            if r == 1 and not in_trend:
                in_trend, start = True, t
            elif r == 0 and in_trend:
                ax.axvspan(start, t, alpha=0.07, color="firebrick", zorder=0)
                in_trend = False
        if in_trend:
            ax.axvspan(start, regimes.index[-1], alpha=0.07, color="firebrick", zorder=0)

    # --- panel 1: P&L curves -----------------------------------------------
    for name, df in results.items():
        ax1.plot(df.index, df["mtm_pnl"],
                 color=COLORS[name], lw=1.2, label=LABELS[name])
    ax1.axhline(0, color="black", lw=0.6, linestyle="--")
    shade_regimes(ax1)
    ax1.set_ylabel("MtM P&L (USD)", fontsize=9)
    ax1.set_title("Total MtM P&L — all phases on the same price path", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: inventory curves -----------------------------------------
    for name, df in results.items():
        ax2.plot(df.index, df["inventory"],
                 color=COLORS[name], lw=1.0, label=LABELS[name], alpha=0.85)
    ax2.axhline(0, color="black", lw=0.6, linestyle="--")
    shade_regimes(ax2)
    ax2.set_ylabel("Inventory (BTC)", fontsize=9)
    ax2.set_title("Inventory — each phase tightens control", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: P&L decomposition bar chart ------------------------------
    names  = list(results.keys())
    x      = np.arange(len(names))
    width  = 0.32

    spread_vals = [results[n]["spread_pnl"].iloc[-1]    for n in names]
    inv_vals    = [results[n]["inventory_pnl"].iloc[-1] for n in names]
    total_vals  = [results[n]["mtm_pnl"].iloc[-1]       for n in names]

    ax3.bar(x - width/2, spread_vals, width,
            color="forestgreen", alpha=0.8, label="Spread P&L")
    ax3.bar(x + width/2, inv_vals, width,
            color=[("firebrick" if v < 0 else "steelblue") for v in inv_vals],
            alpha=0.8, label="Inventory P&L")

    for i, (s, inv, tot) in enumerate(zip(spread_vals, inv_vals, total_vals)):
        ypos = max(s, max(inv, 0)) + 5
        ax3.text(i, ypos, f"Total\n${tot:+,.0f}",
                 ha="center", fontsize=7.5, fontweight="bold",
                 color=COLORS[names[i]])

    ax3.axhline(0, color="black", lw=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels([LABELS[n] for n in names], fontsize=8)
    ax3.set_ylabel("P&L (USD)", fontsize=9)
    ax3.set_title("P&L Decomposition — spread capture (green) vs inventory risk (red=negative / blue=positive)",
                  fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_multi_asset(
    multi: pd.DataFrame,
    indep_btc: pd.DataFrame,
    indep_eth: pd.DataFrame,
    corr: float = 0.85,
    corr_series: pd.Series = None,
    title: str = "Multi-Asset Market Making — BTC + ETH",
) -> plt.Figure:
    """
    Four or five-panel comparison: multi-asset MM vs two independent single-asset AS models.

    Panel 1: Rolling ρ (only shown if corr_series provided)
    Panel 2: BTC inventory — multi vs independent
    Panel 3: ETH inventory — multi vs independent
    Panel 4: Portfolio variance over time — multi vs independent sum
    Panel 5: Combined MtM P&L — multi vs independent sum
    """
    n_panels = 5 if corr_series is not None else 4
    fig = plt.figure(figsize=(14, 4 * n_panels))
    ratios = ([1] + [1.5] * 4) if corr_series is not None else ([1.5] * 4)
    gs = gridspec.GridSpec(n_panels, 1, hspace=0.45, top=0.93, bottom=0.05,
                           height_ratios=ratios)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    if corr_series is not None:
        ax_corr = axes[0]
        ax1, ax2, ax3, ax4 = axes[1], axes[2], axes[3], axes[4]
    else:
        ax_corr = None
        ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    idx = multi.index

    # --- panel 0: rolling ρ (only when corr_series provided) ----------------
    if ax_corr is not None:
        ax_corr.plot(corr_series.index, corr_series.values,
                     color="mediumpurple", lw=1.2, label="Rolling 60-min ρ")
        ax_corr.axhline(corr, color="grey", lw=0.8, linestyle="--",
                        label=f"Fixed ρ={corr:.2f} (synthetic baseline)")
        ax_corr.fill_between(corr_series.index, corr_series.values, corr,
                             where=corr_series.values > corr,
                             alpha=0.18, color="mediumpurple")
        ax_corr.fill_between(corr_series.index, corr_series.values, corr,
                             where=corr_series.values < corr,
                             alpha=0.18, color="firebrick")
        ax_corr.set_ylim(-1.05, 1.05)
        ax_corr.axhline(0, color="black", lw=0.5, linestyle=":")
        ax_corr.set_ylabel("ρ", fontsize=9)
        avg_corr = float(corr_series.dropna().mean())
        ax_corr.set_title(
            f"BTC-ETH Rolling Correlation ρ  (60-min window, avg={avg_corr:.3f})",
            fontsize=10,
        )
        ax_corr.legend(loc="upper right", fontsize=8)
        ax_corr.grid(True, alpha=0.2)
        ax_corr.tick_params(labelbottom=False)

    # --- panel 1: BTC inventory -------------------------------------------
    ax1.plot(idx, multi["inv_btc"],     color="steelblue", lw=1.2, label="Multi-asset")
    ax1.plot(idx, indep_btc["inventory"], color="grey", lw=1.0, linestyle="--",
             alpha=0.8, label="Independent AS")
    ax1.axhline(0, color="black", lw=0.6, linestyle="--")
    ax1.fill_between(idx, multi["inv_btc"], 0,
                     where=multi["inv_btc"] > 0, alpha=0.15, color="darkorange")
    ax1.fill_between(idx, multi["inv_btc"], 0,
                     where=multi["inv_btc"] < 0, alpha=0.15, color="steelblue")
    b_std_m = multi["inv_btc"].std()
    b_std_i = indep_btc["inventory"].std()
    ax1.set_title(
        f"BTC Inventory  — Multi std={b_std_m:.4f}  Indep std={b_std_i:.4f}  "
        f"(reduction {(1-b_std_m/b_std_i)*100:.0f}%)",
        fontsize=10,
    )
    ax1.set_ylabel("BTC", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: ETH inventory -------------------------------------------
    ax2.plot(idx, multi["inv_eth"],     color="darkorange", lw=1.2, label="Multi-asset")
    ax2.plot(idx, indep_eth["inventory"], color="grey", lw=1.0, linestyle="--",
             alpha=0.8, label="Independent AS")
    ax2.axhline(0, color="black", lw=0.6, linestyle="--")
    ax2.fill_between(idx, multi["inv_eth"], 0,
                     where=multi["inv_eth"] > 0, alpha=0.15, color="darkorange")
    ax2.fill_between(idx, multi["inv_eth"], 0,
                     where=multi["inv_eth"] < 0, alpha=0.15, color="steelblue")
    e_std_m = multi["inv_eth"].std()
    e_std_i = indep_eth["inventory"].std()
    inv_corr = multi["inv_btc"].corr(multi["inv_eth"])
    ax2.set_title(
        f"ETH Inventory  — Multi std={e_std_m:.4f}  Indep std={e_std_i:.4f}  "
        f"|  BTC-ETH inv corr (multi): {inv_corr:+.3f}",
        fontsize=10,
    )
    ax2.set_ylabel("ETH", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: portfolio variance ------------------------------------
    # Independent: compute combined variance with realized positions
    dt_year = 1.0 / (365.0 * 24.0 * 60.0)
    sigma_btc_abs = 50_000.0 * 0.80 * np.sqrt(dt_year)
    sigma_eth_abs =  3_000.0 * 1.20 * np.sqrt(dt_year)

    port_var_multi = multi["port_var"]
    q_b_i = indep_btc["inventory"]
    q_e_i = indep_eth["inventory"]
    port_var_indep = (
        (q_b_i * sigma_btc_abs) ** 2
        + (q_e_i * sigma_eth_abs) ** 2
        + 2 * corr * q_b_i * sigma_btc_abs * q_e_i * sigma_eth_abs
    )

    ax3.plot(idx, np.sqrt(port_var_multi), color="steelblue", lw=1.2,
             label="Multi-asset (joint quotes)")
    ax3.plot(idx, np.sqrt(port_var_indep), color="grey", lw=1.0, linestyle="--",
             alpha=0.8, label="Independent AS (realized positions)")
    ax3.fill_between(idx,
                     np.sqrt(port_var_multi), np.sqrt(port_var_indep),
                     where=np.sqrt(port_var_multi) <= np.sqrt(port_var_indep),
                     alpha=0.15, color="steelblue", label="Multi-asset lower risk")
    ax3.set_ylabel("Portfolio σ ($/√step)", fontsize=9)
    pv_m = np.sqrt(port_var_multi).mean()
    pv_i = np.sqrt(port_var_indep).mean()
    ax3.set_title(
        f"Portfolio Risk σ  — Multi avg={pv_m:.3f}  Indep avg={pv_i:.3f}  "
        f"(reduction {(1-pv_m/pv_i)*100:.0f}%)",
        fontsize=10,
    )
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # --- panel 4: combined P&L -------------------------------------------
    indep_combined_pnl = indep_btc["mtm_pnl"] + indep_eth["mtm_pnl"]
    ax4.plot(idx, multi["mtm_pnl"],      color="steelblue", lw=1.2,
             label="Multi-asset MtM P&L")
    ax4.plot(idx, indep_combined_pnl,    color="grey", lw=1.0, linestyle="--",
             alpha=0.8, label="Independent AS (BTC + ETH combined)")
    ax4.axhline(0, color="black", lw=0.6, linestyle="--")
    ax4.fill_between(idx, multi["mtm_pnl"], indep_combined_pnl,
                     where=multi["mtm_pnl"] >= indep_combined_pnl,
                     alpha=0.12, color="steelblue", label="Multi-asset outperforms")
    ax4.fill_between(idx, multi["mtm_pnl"], indep_combined_pnl,
                     where=multi["mtm_pnl"] < indep_combined_pnl,
                     alpha=0.12, color="firebrick", label="Independent outperforms")
    ax4.set_ylabel("P&L (USD)", fontsize=9)
    final_m = multi["mtm_pnl"].iloc[-1]
    final_i = indep_combined_pnl.iloc[-1]
    ax4.set_title(
        f"Combined MtM P&L  — Multi ${final_m:+,.0f}   Independent ${final_i:+,.0f}",
        fontsize=10,
    )
    ax4.legend(loc="upper left", fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_exact_attribution(
    results: dict,
    regimes: pd.Series,
    title: str = "Phase 5+ — Exact P&L Attribution",
) -> plt.Figure:
    """
    Three-panel comparison: exact vs approximate P&L decomposition.

    Panel 1: Approximation error (exact - approx spread P&L) over time.
    Panel 2: Exact spread P&L curves for all models.
    Panel 3: Side-by-side bar chart — exact vs approx spread/inventory split.
    """
    COLORS = {
        "Baseline": "grey",
        "AS":       "steelblue",
        "Phase3":   "darkorange",
        "Phase4":   "firebrick",
    }
    LABELS = {
        "Baseline": "Phase 1: Baseline",
        "AS":       "Phase 2: AS",
        "Phase3":   "Phase 3: Adverse Selection",
        "Phase4":   "Phase 4: Risk Controls",
    }

    fig = plt.figure(figsize=(14, 13))
    gs = gridspec.GridSpec(3, 1, hspace=0.48, top=0.93, bottom=0.06,
                           height_ratios=[1.5, 2, 1.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    def shade_regimes(ax):
        in_trend, start = False, None
        for t, r in regimes.items():
            if r == 1 and not in_trend:
                in_trend, start = True, t
            elif r == 0 and in_trend:
                ax.axvspan(start, t, alpha=0.07, color="firebrick", zorder=0)
                in_trend = False
        if in_trend:
            ax.axvspan(start, regimes.index[-1], alpha=0.07, color="firebrick", zorder=0)

    # --- panel 1: approximation error over time --------------------------------
    for name, df in results.items():
        if "spread_pnl_approx" not in df.columns:
            continue  # baseline: exact = approx, skip
        error = df["spread_pnl"] - df["spread_pnl_approx"]
        ax1.plot(df.index, error, color=COLORS[name], lw=1.1, label=LABELS[name])

    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    ax1.fill_between(results["AS"].index,
                     results["AS"]["spread_pnl"] - results["AS"]["spread_pnl_approx"], 0,
                     where=(results["AS"]["spread_pnl"] - results["AS"]["spread_pnl_approx"]) >= 0,
                     alpha=0.10, color="steelblue")
    ax1.fill_between(results["AS"].index,
                     results["AS"]["spread_pnl"] - results["AS"]["spread_pnl_approx"], 0,
                     where=(results["AS"]["spread_pnl"] - results["AS"]["spread_pnl_approx"]) < 0,
                     alpha=0.10, color="firebrick")
    shade_regimes(ax1)
    ax1.set_ylabel("Error (USD)", fontsize=9)
    ax1.set_title(
        "Spread P&L error = exact − approx  (approx uses δ × lot × fills; exact uses mid − fill_price)",
        fontsize=10,
    )
    ax1.legend(loc="lower left", fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # --- panel 2: exact spread P&L curves ------------------------------------
    for name, df in results.items():
        ax2.plot(df.index, df["spread_pnl"],
                 color=COLORS[name], lw=1.1, label=f"{LABELS[name]} (exact)")
        if "spread_pnl_approx" in df.columns:
            ax2.plot(df.index, df["spread_pnl_approx"],
                     color=COLORS[name], lw=0.7, linestyle=":", alpha=0.55,
                     label=f"{LABELS[name]} (approx)")
    shade_regimes(ax2)
    ax2.set_ylabel("Cumulative Spread P&L (USD)", fontsize=9)
    ax2.set_title("Spread P&L — solid=exact, dotted=approx  (Baseline: no difference)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # --- panel 3: exact vs approx decomposition bar chart --------------------
    names  = list(results.keys())
    x      = np.arange(len(names))
    width  = 0.20

    exact_spread  = [results[n]["spread_pnl"].iloc[-1]    for n in names]
    approx_spread = [
        results[n]["spread_pnl_approx"].iloc[-1]
        if "spread_pnl_approx" in results[n].columns
        else results[n]["spread_pnl"].iloc[-1]
        for n in names
    ]
    exact_inv     = [results[n]["inventory_pnl"].iloc[-1] for n in names]
    total_vals    = [results[n]["mtm_pnl"].iloc[-1]       for n in names]

    ax3.bar(x - width * 1.5, exact_spread,  width, color="forestgreen", alpha=0.85, label="Spread P&L (exact)")
    ax3.bar(x - width * 0.5, approx_spread, width, color="forestgreen", alpha=0.35, label="Spread P&L (approx)", hatch="//")
    ax3.bar(x + width * 0.5, exact_inv,     width,
            color=[("firebrick" if v < 0 else "steelblue") for v in exact_inv],
            alpha=0.85, label="Inventory P&L (exact)")

    for i, (es, ap, inv, tot) in enumerate(zip(exact_spread, approx_spread, exact_inv, total_vals)):
        err = es - ap
        ypos = max(es, max(inv, 0)) + 4
        ax3.text(i, ypos,
                 f"Total ${tot:+,.0f}\nErr {err:+,.1f}",
                 ha="center", fontsize=7, fontweight="bold",
                 color=COLORS[names[i]])

    ax3.axhline(0, color="black", lw=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels([LABELS[n] for n in names], fontsize=8)
    ax3.set_ylabel("P&L (USD)", fontsize=9)
    ax3.set_title(
        "Exact vs Approx decomposition — dark green=exact, hatched=approx  (Err = exact − approx)",
        fontsize=10,
    )
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.2, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def plot_montecarlo(
    stats: dict,
    n_paths: int,
    title: str = "Monte Carlo Backtest",
) -> plt.Figure:
    COLORS = {"Baseline": "grey", "AS": "steelblue", "Phase3": "darkorange", "Phase4": "firebrick"}
    LABELS = {"Baseline": "Phase 1\nBaseline", "AS": "Phase 2\nAS",
              "Phase3": "Phase 3\nAdv. Sel.", "Phase4": "Phase 4\nRisk Ctrl"}

    names  = list(stats.keys())
    colors = [COLORS[n] for n in names]

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                            top=0.92, bottom=0.07, left=0.08, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    def violin(ax, key, ylabel, title_str):
        data  = [stats[n][key] for n in names]
        parts = ax.violinplot(data, positions=range(len(names)),
                              showmedians=True, showextrema=True)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        for k in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[k].set_color("black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABELS[n] for n in names], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title_str, fontsize=10)
        ax.axhline(0, color="black", lw=0.6, linestyle="--")
        ax.grid(True, alpha=0.2, axis="y")

    violin(ax1, "final_pnl",    "USD",  "Final MtM P&L distribution")
    violin(ax2, "max_drawdown", "USD",  "Max Drawdown distribution")
    violin(ax3, "inv_std",      "BTC",  "Inventory Std Dev distribution")

    # Sharpe bar chart (cross-path: mean / std of final P&L)
    sharpes = []
    for n in names:
        arr = np.array(stats[n]["final_pnl"])
        sharpes.append(arr.mean() / (arr.std() + 1e-9))

    bars = ax4.bar(range(len(names)), sharpes, color=colors, alpha=0.75, width=0.5)
    for bar, val in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 val + (0.02 if val >= 0 else -0.12),
                 f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([LABELS[n] for n in names], fontsize=8)
    ax4.set_ylabel("Sharpe  (mean / std  across paths)", fontsize=9)
    ax4.set_title("Cross-path Sharpe — P&L consistency across market conditions", fontsize=10)
    ax4.axhline(0, color="black", lw=0.6, linestyle="--")
    ax4.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"{title}  ({n_paths} paths)", fontsize=13, fontweight="bold")
    return fig
