"""
Compare simple rolling toxicity (Phase 3) vs VPIN-enhanced model on real BTC data.

Usage:
    python run_vpin.py
    python run_vpin.py --date 2024-06-10
    python run_vpin.py --no-show
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simulator.data_loader import fetch_btc_ohlcv
from simulator.vpin import compute_vpin
from simulator.adverse_selection import AdverseSelectionMM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date",    type=str, default=None)
    p.add_argument("--n-steps", type=int, default=1440)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("Fetching real BTC/USDT OHLCV data from Binance…")
    ohlcv  = fetch_btc_ohlcv(date=args.date, n_steps=args.n_steps)
    prices = ohlcv["close"].rename("mid_price")
    print(f"  {len(ohlcv):,} bars  |  ${prices.min():,.0f}–${prices.max():,.0f}  "
          f"|  move {(prices.iloc[-1]/prices.iloc[0]-1)*100:+.2f}%")

    print("Computing VPIN…")
    vpin = compute_vpin(ohlcv, n_buckets=50, n_window=50)
    print(f"  VPIN range: {vpin.min():.3f}–{vpin.max():.3f}  "
          f"|  mean: {vpin.mean():.3f}  |  >0.4: {(vpin > 0.4).mean():.1%}")

    shared = dict(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=len(prices), lot_size=0.01,
        toxicity_window=30, toxicity_mult=2.0,
        initial_cash=prices.iloc[0] * 2, seed=0,
    )

    print("\nRunning Phase 3 (simple rolling toxicity)…")
    p3 = AdverseSelectionMM(**shared)
    p3_results = p3.run(prices)

    print("Running VPIN-enhanced model…")
    pv = AdverseSelectionMM(**shared)
    vpin_aligned = vpin.reindex(prices.index).fillna(0.0)
    records = [pv.step(ts, p, toxicity_override=float(vpin_aligned.loc[ts]))
               for ts, p in prices.items()]
    import pandas as pd
    vpin_results = pd.DataFrame(records, index=prices.index)

    # Summary
    print("\n" + "=" * 60)
    print(f"  {'':30} {'Phase 3':>12} {'VPIN':>12}")
    print("=" * 60)
    for label, key in [("Final P&L", "mtm_pnl"), ("Spread P&L", "spread_pnl"),
                        ("Inventory P&L", "inventory_pnl")]:
        print(f"  {label:<30} {p3_results[key].iloc[-1]:>+12.2f} "
              f"{vpin_results[key].iloc[-1]:>+12.2f}")
    print(f"  {'Max drawdown':<30} {p3_results['mtm_pnl'].min():>+12.2f} "
          f"{vpin_results['mtm_pnl'].min():>+12.2f}")
    print(f"  {'Inventory std dev':<30} {p3_results['inventory'].std():>12.4f} "
          f"{vpin_results['inventory'].std():>12.4f}")
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

    # Panel 2: VPIN vs simple toxicity
    ax2.plot(p3_results.index, p3_results["toxicity"],
             color="steelblue", lw=0.8, alpha=0.7, label="Simple toxicity (rolling up-frac)")
    ax2.plot(vpin.index, vpin.values,
             color="firebrick", lw=1.2, label="VPIN (volume-based)")
    ax2.axhline(0.4, color="black", lw=0.7, linestyle="--", label="Threshold 0.4")
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Toxicity / VPIN", fontsize=9)
    ax2.set_title("VPIN vs Simple Toxicity  —  VPIN is smoother and volume-anchored", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: effective spread comparison
    ax3.plot(p3_results.index, p3_results["delta_eff"],
             color="steelblue", lw=1.0, label="Phase 3 spread (simple toxicity)")
    ax3.plot(vpin_results.index, vpin_results["delta_eff"],
             color="firebrick", lw=1.0, label="VPIN spread")
    ax3.set_ylabel("Half-spread (USD)", fontsize=9)
    ax3.set_title("Effective Spread  —  VPIN triggers differently", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # Panel 4: P&L comparison
    ax4.plot(p3_results.index, p3_results["mtm_pnl"],
             color="steelblue", lw=1.2, label="Phase 3 (simple toxicity)")
    ax4.plot(vpin_results.index, vpin_results["mtm_pnl"],
             color="firebrick", lw=1.2, label="VPIN-enhanced")
    ax4.axhline(0, color="black", lw=0.6, linestyle="--")
    ax4.set_ylabel("MtM P&L (USD)", fontsize=9)
    ax4.set_title("P&L Comparison", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True, alpha=0.2)

    date_str = args.date or prices.index[0].strftime("%Y-%m-%d")
    fig.suptitle(f"VPIN vs Simple Toxicity — Real BTC Data  {date_str}",
                 fontsize=13, fontweight="bold")

    out_path = os.path.join(os.path.dirname(__file__), "vpin_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
