"""
Phase 1 — Baseline market maker.

Usage:
    python run_baseline.py           # default params (1 day of 1-min BTC bars)
    python run_baseline.py --seed 7  # try a different random path
    python run_baseline.py --delta 100 --prob-fill 0.25
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price
from simulator.baseline_mm import BaselineMarketMaker
from simulator.visualize import plot_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",        type=float, default=50_000.0, help="Initial BTC price")
    p.add_argument("--mu",        type=float, default=0.0,      help="Annual drift")
    p.add_argument("--sigma",     type=float, default=0.80,     help="Annual volatility")
    p.add_argument("--n-steps",   type=int,   default=1440,     help="Number of 1-min bars")
    p.add_argument("--delta",     type=float, default=50.0,     help="Half-spread in USD")
    p.add_argument("--lot-size",  type=float, default=0.01,     help="BTC per fill")
    p.add_argument("--prob-fill", type=float, default=0.30,     help="Fill probability per side per step")
    p.add_argument("--seed",      type=int,   default=42,       help="Random seed")
    p.add_argument("--no-show",   action="store_true",          help="Save PNG only, don't open window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating synthetic BTC price path (GBM)…")
    prices = generate_btc_price(
        S0=args.S0,
        mu=args.mu,
        sigma=args.sigma,
        n_steps=args.n_steps,
        dt_minutes=1.0,
        seed=args.seed,
    )
    print(f"  {len(prices):,} bars  |  range ${prices.min():,.0f} – ${prices.max():,.0f}")

    print("Running baseline market maker…")
    mm = BaselineMarketMaker(
        delta=args.delta,
        lot_size=args.lot_size,
        prob_fill=args.prob_fill,
        initial_cash=100_000.0,
        seed=args.seed,
    )
    results = mm.run(prices)
    mm.summary(results)

    subtitle = (
        f"δ = ${args.delta:.0f}  |  lot = {args.lot_size} BTC  |  "
        f"p_fill = {args.prob_fill:.0%}  |  σ = {args.sigma:.0%}/yr  |  seed = {args.seed}"
    )
    fig = plot_simulation(results, title=f"Baseline Market Maker — BTC\n{subtitle}")

    out_path = os.path.join(os.path.dirname(__file__), "baseline_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
