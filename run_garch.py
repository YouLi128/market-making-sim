"""
Compare AS with fixed sigma vs GARCH(1,1) dynamic sigma on real BTC data.

Usage:
    python run_garch.py
    python run_garch.py --date 2024-06-10
    python run_garch.py --no-show
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simulator.data_loader import fetch_btc_price
from simulator.garch import compute_garch_sigma
from simulator.avellaneda_stoikov import AvellanedaStoikov


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date",    type=str, default=None)
    p.add_argument("--n-steps", type=int, default=1440)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Fetching real BTC/USDT data from Binance…")
    prices = fetch_btc_price(date=args.date, n_steps=args.n_steps)
    print(f"  {len(prices):,} bars  |  ${prices.min():,.0f}–${prices.max():,.0f}  "
          f"|  move {(prices.iloc[-1]/prices.iloc[0]-1)*100:+.2f}%")

    print("Fitting GARCH(1,1)…")
    sigma_series = compute_garch_sigma(prices, baseline_sigma=0.80)
    print(f"  sigma range: {sigma_series.min():.2f}–{sigma_series.max():.2f}  "
          f"|  mean: {sigma_series.mean():.2f}  |  baseline: 0.80")

    shared = dict(gamma=50.0, delta_0=45.0, delta_min=5.0, T=len(prices),
                  lot_size=0.01, initial_cash=prices.iloc[0] * 2, seed=0)

    print("\nRunning AS with fixed sigma (baseline 0.80)…")
    as_fixed = AvellanedaStoikov(**shared)
    fixed_results = as_fixed.run(prices)

    print("Running AS with GARCH dynamic sigma…")
    as_garch = AvellanedaStoikov(**shared)
    garch_results = as_garch.run_garch(prices, sigma_series)

    # Summary
    print("\n" + "=" * 60)
    print(f"  {'':30} {'Fixed σ':>12} {'GARCH σ':>12}")
    print("=" * 60)
    for label, key in [("Final P&L",     "mtm_pnl"),
                        ("Spread P&L",    "spread_pnl"),
                        ("Inventory P&L", "inventory_pnl"),
                        ("Max drawdown",  None),
                        ("Inventory std", None)]:
        if key:
            fv = fixed_results[key].iloc[-1]
            gv = garch_results[key].iloc[-1]
            print(f"  {label:<30} {fv:>+12.2f} {gv:>+12.2f}")
    print(f"  {'Max drawdown':<30} {fixed_results['mtm_pnl'].min():>+12.2f} "
          f"{garch_results['mtm_pnl'].min():>+12.2f}")
    print(f"  {'Inventory std':<30} {fixed_results['inventory'].std():>12.4f} "
          f"{garch_results['inventory'].std():>12.4f}")
    print("=" * 60)

    # Plot
    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(4, 1, hspace=0.42, top=0.93, bottom=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # Panel 1: price
    ax1.plot(prices.index, prices.values, color="steelblue", lw=1.2)
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.set_title("Real BTC/USDT price", fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # Panel 2: GARCH sigma vs fixed
    ax2.plot(sigma_series.index, sigma_series.values,
             color="firebrick", lw=1.2, label="GARCH σ (dynamic)")
    ax2.axhline(0.80, color="steelblue", lw=1.0, linestyle="--", label="Fixed σ = 0.80")
    ax2.fill_between(sigma_series.index, sigma_series.values, 0.80,
                     where=sigma_series.values > 0.80,
                     alpha=0.2, color="firebrick", label="High vol regime")
    ax2.fill_between(sigma_series.index, sigma_series.values, 0.80,
                     where=sigma_series.values < 0.80,
                     alpha=0.2, color="steelblue", label="Low vol regime")
    ax2.set_ylabel("Annual σ", fontsize=9)
    ax2.set_title("GARCH(1,1) Conditional Volatility — captures vol clustering", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: spread comparison
    ax3.plot(fixed_results.index, fixed_results["half_spread"],
             color="steelblue", lw=1.0, label="Fixed σ spread")
    ax3.plot(garch_results.index, garch_results["half_spread"],
             color="firebrick", lw=1.0, label="GARCH spread (widens in high vol)")
    ax3.set_ylabel("Half-spread (USD)", fontsize=9)
    ax3.set_title("Spread — GARCH automatically widens during volatile periods", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # Panel 4: P&L
    ax4.plot(fixed_results.index, fixed_results["mtm_pnl"],
             color="steelblue", lw=1.2, label="AS fixed σ")
    ax4.plot(garch_results.index, garch_results["mtm_pnl"],
             color="firebrick", lw=1.2, label="AS + GARCH σ")
    ax4.axhline(0, color="black", lw=0.6, linestyle="--")
    ax4.set_ylabel("MtM P&L (USD)", fontsize=9)
    ax4.set_title("P&L Comparison", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    date_str = args.date or prices.index[0].strftime("%Y-%m-%d")
    fig.suptitle(f"AS Fixed σ vs GARCH(1,1) Dynamic σ — Real BTC  {date_str}",
                 fontsize=13, fontweight="bold")

    out_path = os.path.join(os.path.dirname(__file__), "garch_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
