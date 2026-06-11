"""
Phase 5 — Performance Attribution.

Runs all four models on the same regime-switching price path and produces:
  1. Console summary table
  2. attribution_results.png — 3-panel comparison chart

Usage:
    python run_attribution.py
    python run_attribution.py --seed 7
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from simulator.data_gen import generate_btc_price_regime
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM
from simulator.visualize import plot_attribution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",      type=float, default=50_000.0)
    p.add_argument("--n-steps", type=int,   default=1440)
    p.add_argument("--seed",    type=int,   default=7)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def print_summary_table(results: dict) -> None:
    print("\n" + "=" * 90)
    print(f"  {'Model':<28} {'Fills':>6} {'Spread P&L':>12} {'Inv P&L':>12} {'Total P&L':>12} {'MaxDD':>10} {'Inv Std':>9}")
    print("=" * 90)
    for name, df in results.items():
        fills     = len([t for t in df.index if True])   # proxy: just count rows
        fills_col = None
        spread    = df["spread_pnl"].iloc[-1]
        inv_pnl   = df["inventory_pnl"].iloc[-1]
        total     = df["mtm_pnl"].iloc[-1]
        maxdd     = df["mtm_pnl"].min()
        inv_std   = df["inventory"].std()
        print(f"  {name:<28} {'—':>6} {spread:>+12,.2f} {inv_pnl:>+12,.2f} {total:>+12,.2f} {maxdd:>+10,.2f} {inv_std:>9.4f}")
    print("=" * 90)


def main() -> None:
    args = parse_args()

    print("Generating regime-switching BTC price path…")
    prices, regimes = generate_btc_price_regime(
        S0=args.S0, n_steps=args.n_steps, seed=args.seed,
    )
    print(f"  {len(prices):,} bars  |  trending: {regimes.mean()*100:.1f}%  "
          f"|  range ${prices.min():,.0f}–${prices.max():,.0f}")

    shared = dict(lot_size=0.01, initial_cash=100_000.0, seed=args.seed)

    print("\nRunning all four models…")

    b = BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared)
    b_res = b.run(prices)

    a = AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
                          T=args.n_steps, **shared)
    a_res = a.run(prices)

    p3 = AdverseSelectionMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                             T=args.n_steps, toxicity_window=30, toxicity_mult=2.0,
                             **shared)
    p3_res = p3.run(prices)

    p4 = RiskControlMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                       T=args.n_steps, toxicity_window=30, toxicity_mult=2.0,
                       max_inventory=0.10, vol_threshold=1.2, vol_spread_mult=1.3,
                       max_holding_steps=120, max_inventory_loss=150.0,
                       **shared)
    p4_res = p4.run(prices)

    all_results = {
        "Baseline": b_res,
        "AS":       a_res,
        "Phase3":   p3_res,
        "Phase4":   p4_res,
    }

    print_summary_table(all_results)

    fig = plot_attribution(
        all_results, regimes,
        title=f"Phase 5 — Performance Attribution  (seed={args.seed})",
    )

    out_path = os.path.join(os.path.dirname(__file__), "attribution_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
