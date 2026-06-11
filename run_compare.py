"""
Run baseline and AS on the *same* price path and plot a side-by-side comparison.

Usage:
    python run_compare.py             # default seed=42
    python run_compare.py --seed 7
    python run_compare.py --mu 0.5   # trending market — shows AS advantage more clearly
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.visualize import plot_comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",        type=float, default=50_000.0)
    p.add_argument("--mu",        type=float, default=0.0)
    p.add_argument("--sigma",     type=float, default=0.80)
    p.add_argument("--n-steps",   type=int,   default=1440)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no-show",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating synthetic BTC price path…")
    prices = generate_btc_price(
        S0=args.S0, mu=args.mu, sigma=args.sigma,
        n_steps=args.n_steps, dt_minutes=1.0, seed=args.seed,
    )
    print(f"  {len(prices):,} bars  |  range ${prices.min():,.0f} – ${prices.max():,.0f}")

    shared = dict(lot_size=0.01, prob_fill=0.30, initial_cash=100_000.0, seed=args.seed)

    print("\nRunning Baseline…")
    baseline = BaselineMarketMaker(delta=50.0, **shared)
    b_results = baseline.run(prices)
    baseline.summary(b_results)

    print("\nRunning Avellaneda-Stoikov…")
    as_mm = AvellanedaStoikov(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=args.n_steps, **shared,
    )
    a_results = as_mm.run(prices)
    as_mm.summary(a_results)

    print("\n=== Head-to-head ===")
    print(f"  Inventory std dev  —  Baseline: {b_results['inventory'].std():.4f} BTC  |  AS: {a_results['inventory'].std():.4f} BTC")
    print(f"  Final MtM P&L      —  Baseline: ${b_results['mtm_pnl'].iloc[-1]:+,.2f}  |  AS: ${a_results['mtm_pnl'].iloc[-1]:+,.2f}")
    print(f"  Max |inventory|    —  Baseline: {b_results['inventory'].abs().max():.4f} BTC  |  AS: {a_results['inventory'].abs().max():.4f} BTC")

    fig = plot_comparison(
        b_results, a_results,
        title=f"Baseline vs Avellaneda-Stoikov — BTC (seed={args.seed}, μ={args.mu:.1f}, σ={args.sigma:.0%}/yr)",
    )

    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
