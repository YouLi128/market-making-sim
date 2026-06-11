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
