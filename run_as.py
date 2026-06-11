"""
Phase 2 — Avellaneda-Stoikov model.

Usage:
    python run_as.py                  # default params
    python run_as.py --seed 7
    python run_as.py --gamma 5e-5     # more aggressive inventory skew
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.visualize import plot_as_simulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",        type=float, default=50_000.0)
    p.add_argument("--mu",        type=float, default=0.0)
    p.add_argument("--sigma",     type=float, default=0.80)
    p.add_argument("--n-steps",   type=int,   default=1440)
    p.add_argument("--gamma",      type=float, default=50.0,  help="$/BTC inventory skew sensitivity")
    p.add_argument("--delta-0",   type=float, default=45.0,  help="Max half-spread at session start")
    p.add_argument("--delta-min", type=float, default=5.0,   help="Min half-spread at session end")
    p.add_argument("--lot-size",  type=float, default=0.01)
    p.add_argument("--prob-fill", type=float, default=0.30)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--no-show",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating synthetic BTC price path (GBM)…")
    prices = generate_btc_price(
        S0=args.S0, mu=args.mu, sigma=args.sigma,
        n_steps=args.n_steps, dt_minutes=1.0, seed=args.seed,
    )
    print(f"  {len(prices):,} bars  |  range ${prices.min():,.0f} – ${prices.max():,.0f}")

    print("Running Avellaneda-Stoikov model…")
    mm = AvellanedaStoikov(
        gamma=args.gamma,
        delta_0=args.delta_0,
        delta_min=args.delta_min,
        T=args.n_steps,
        lot_size=args.lot_size,
        prob_fill=args.prob_fill,
        initial_cash=100_000.0,
        seed=args.seed,
    )
    results = mm.run(prices)
    mm.summary(results)

    subtitle = (
        f"γ={args.gamma:.0f} $/BTC  |  δ=[{args.delta_min:.0f},{args.delta_0+args.delta_min:.0f}]  |  "
        f"lot={args.lot_size} BTC  |  p_fill={args.prob_fill:.0%}  |  σ={args.sigma:.0%}/yr  |  seed={args.seed}"
    )
    fig = plot_as_simulation(results, title=f"Avellaneda-Stoikov — BTC\n{subtitle}")

    out_path = os.path.join(os.path.dirname(__file__), "as_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
