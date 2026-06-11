"""
Run all four models on real BTC/USDT 1-minute data from Binance.

Usage:
    python run_real_data.py                    # latest 1440 bars
    python run_real_data.py --date 2024-06-01  # specific date
    python run_real_data.py --n-steps 480      # last 8 hours
    python run_real_data.py --no-show
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

from simulator.data_loader import fetch_btc_price
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date",    type=str, default=None,  help="YYYY-MM-DD (UTC), default=latest")
    p.add_argument("--n-steps", type=int, default=1440)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def plot_real_data(results: dict, prices: pd.Series, title: str) -> plt.Figure:
    COLORS = {"Baseline": "grey", "AS": "steelblue", "Phase3": "darkorange", "Phase4": "firebrick"}
    LABELS = {"Baseline": "Phase 1: Baseline", "AS": "Phase 2: AS",
              "Phase3": "Phase 3: Adv. Sel.", "Phase4": "Phase 4: Risk Ctrl"}

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4, top=0.93, bottom=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel 1: real price
    ax1.plot(prices.index, prices.values, color="steelblue", lw=1.2)
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.set_title(f"Real BTC/USDT — 1-min close  |  range ${prices.min():,.0f}–${prices.max():,.0f}  "
                  f"|  move {(prices.iloc[-1]/prices.iloc[0]-1)*100:+.2f}%", fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # Panel 2: P&L curves
    for name, df in results.items():
        ax2.plot(df.index, df["mtm_pnl"],
                 color=COLORS[name], lw=1.2, label=LABELS[name])
    ax2.axhline(0, color="black", lw=0.6, linestyle="--")
    ax2.set_ylabel("MtM P&L (USD)", fontsize=9)
    ax2.set_title("MtM P&L — all models on real data", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: inventory curves
    for name, df in results.items():
        ax3.plot(df.index, df["inventory"],
                 color=COLORS[name], lw=1.0, label=LABELS[name], alpha=0.85)
    ax3.axhline(0, color="black", lw=0.6, linestyle="--")
    ax3.set_ylabel("Inventory (BTC)", fontsize=9)
    ax3.set_title("Inventory — each phase tightens control", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


def print_table(results: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {'Model':<22} {'Spread P&L':>12} {'Inv P&L':>12} "
          f"{'Total P&L':>12} {'MaxDD':>10} {'Inv Std':>9}")
    print("=" * 78)
    for name, df in results.items():
        print(f"  {name:<22} "
              f"{df['spread_pnl'].iloc[-1]:>+12.2f} "
              f"{df['inventory_pnl'].iloc[-1]:>+12.2f} "
              f"{df['mtm_pnl'].iloc[-1]:>+12.2f} "
              f"{df['mtm_pnl'].min():>+10.2f} "
              f"{df['inventory'].std():>9.4f}")
    print("=" * 78)


def main():
    args = parse_args()

    print("Fetching real BTC/USDT data from Binance…")
    prices = fetch_btc_price(date=args.date, n_steps=args.n_steps)
    start  = prices.index[0].strftime("%Y-%m-%d %H:%M UTC")
    end    = prices.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    print(f"  {len(prices):,} bars  |  {start} → {end}")
    print(f"  price range: ${prices.min():,.2f} – ${prices.max():,.2f}")
    print(f"  total move:  {(prices.iloc[-1]/prices.iloc[0]-1)*100:+.2f}%")

    shared = dict(lot_size=0.01, initial_cash=prices.iloc[0] * 2, seed=0)

    print("\nRunning all four models on real data…")
    n = len(prices)

    all_results = {
        "Baseline": BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared).run(prices),
        "AS":       AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                      T=n, **shared).run(prices),
        "Phase3":   AdverseSelectionMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                       T=n, toxicity_window=30, toxicity_mult=2.0,
                                       **shared).run(prices),
        "Phase4":   RiskControlMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                   T=n, toxicity_window=30, toxicity_mult=2.0,
                                   max_inventory=0.10, vol_threshold=1.2,
                                   vol_spread_mult=1.3, max_holding_steps=120,
                                   max_inventory_loss=150.0, **shared).run(prices),
    }

    print_table(all_results)

    date_str = args.date or prices.index[0].strftime("%Y-%m-%d")
    fig = plot_real_data(
        all_results, prices,
        title=f"Real BTC/USDT Data — {date_str}  ({n} bars)",
    )

    out_path = os.path.join(os.path.dirname(__file__), "real_data_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
