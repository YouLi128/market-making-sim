"""
Phase 3 — Adverse Selection Detection.

Runs Phase 3 (adverse selection aware) and Phase 2 (AS baseline) on the same
regime-switching price path, then plots a 4-panel comparison.

Usage:
    python run_phase3.py
    python run_phase3.py --seed 7
    python run_phase3.py --toxicity-mult 3.0   # more aggressive spread widening
    python run_phase3.py --toxicity-window 20  # shorter detection window
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price_regime
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.visualize import plot_phase3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",             type=float, default=50_000.0)
    p.add_argument("--n-steps",        type=int,   default=1440)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--toxicity-window",type=int,   default=30)
    p.add_argument("--toxicity-mult",  type=float, default=2.0)
    p.add_argument("--no-show",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating regime-switching BTC price path…")
    prices, regimes = generate_btc_price_regime(
        S0=args.S0,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    trend_pct = regimes.mean() * 100
    print(f"  {len(prices):,} bars  |  trending regime: {trend_pct:.1f}% of session")
    print(f"  price range: ${prices.min():,.0f} – ${prices.max():,.0f}")

    shared = dict(lot_size=0.01, initial_cash=100_000.0, seed=args.seed)

    print("\nRunning Phase 2 (AS, no adverse selection detection)…")
    as_mm = AvellanedaStoikov(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=args.n_steps, **shared,
    )
    as_results = as_mm.run(prices)
    as_mm.summary(as_results)

    print("\nRunning Phase 3 (AS + adverse selection detection)…")
    p3_mm = AdverseSelectionMM(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=args.n_steps,
        toxicity_window=args.toxicity_window,
        toxicity_mult=args.toxicity_mult,
        **shared,
    )
    p3_results = p3_mm.run(prices)
    p3_mm.summary(p3_results)

    print("\n=== Head-to-head ===")
    print(f"  Inventory std dev  —  AS: {as_results['inventory'].std():.4f}  |  Phase 3: {p3_results['inventory'].std():.4f}")
    print(f"  Max |inventory|    —  AS: {as_results['inventory'].abs().max():.4f}  |  Phase 3: {p3_results['inventory'].abs().max():.4f}")
    print(f"  Final MtM P&L      —  AS: ${as_results['mtm_pnl'].iloc[-1]:+,.2f}  |  Phase 3: ${p3_results['mtm_pnl'].iloc[-1]:+,.2f}")
    print(f"  Max drawdown       —  AS: ${as_results['mtm_pnl'].min():+,.2f}  |  Phase 3: ${p3_results['mtm_pnl'].min():+,.2f}")

    subtitle = (
        f"toxicity window={args.toxicity_window}  |  "
        f"toxicity mult={args.toxicity_mult}  |  seed={args.seed}"
    )
    fig = plot_phase3(
        p3_results, regimes, as_results=as_results,
        title=f"Phase 3 — Adverse Selection Detection\n{subtitle}",
    )

    out_path = os.path.join(os.path.dirname(__file__), "phase3_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
