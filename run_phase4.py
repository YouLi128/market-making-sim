"""
Phase 4 — Risk Controls.

Runs Phase 3 and Phase 4 on the same regime-switching price path.

Usage:
    python run_phase4.py
    python run_phase4.py --seed 7
    python run_phase4.py --max-inventory 0.20   # tighter hard limit
    python run_phase4.py --vol-threshold 1.2    # more sensitive vol trigger
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price_regime
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM
from simulator.visualize import plot_phase4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",             type=float, default=50_000.0)
    p.add_argument("--n-steps",        type=int,   default=1440)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--max-inventory",    type=float, default=0.10)
    p.add_argument("--vol-threshold",    type=float, default=1.2)
    p.add_argument("--vol-spread-mult",  type=float, default=1.3)
    p.add_argument("--max-holding-steps",type=int,   default=120, help="Emergency if same-side inventory held > N steps")
    p.add_argument("--max-inv-loss",     type=float, default=150.0, help="Emergency if inventory P&L < -X USD")
    p.add_argument("--no-show",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Generating regime-switching BTC price path…")
    prices, regimes = generate_btc_price_regime(
        S0=args.S0, n_steps=args.n_steps, seed=args.seed,
    )
    print(f"  {len(prices):,} bars  |  trending: {regimes.mean()*100:.1f}%  "
          f"|  range ${prices.min():,.0f}–${prices.max():,.0f}")

    shared = dict(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=args.n_steps, lot_size=0.01,
        toxicity_window=30, toxicity_mult=2.0,
        initial_cash=100_000.0, seed=args.seed,
    )

    print("\nRunning Phase 3 (no risk controls)…")
    p3 = AdverseSelectionMM(**shared)
    p3_results = p3.run(prices)
    p3.summary(p3_results)

    print("\nRunning Phase 4 (risk controls)…")
    p4 = RiskControlMM(
        **shared,
        max_inventory=args.max_inventory,
        vol_threshold=args.vol_threshold,
        vol_spread_mult=args.vol_spread_mult,
        max_holding_steps=args.max_holding_steps,
        max_inventory_loss=args.max_inv_loss,
    )
    p4_results = p4.run(prices)
    p4.summary(p4_results)

    print("\n=== Head-to-head ===")
    for label, r in [("Phase 3", p3_results), ("Phase 4", p4_results)]:
        print(f"  {label}:  inv_std={r['inventory'].std():.4f}  "
              f"max|inv|={r['inventory'].abs().max():.4f}  "
              f"P&L=${r['mtm_pnl'].iloc[-1]:+,.2f}  "
              f"drawdown=${r['mtm_pnl'].min():+,.2f}")

    subtitle = (
        f"max_inv=±{args.max_inventory} BTC  |  "
        f"vol_thr={args.vol_threshold}×  |  seed={args.seed}"
    )
    fig = plot_phase4(
        p4_results, p3_results, regimes,
        max_inventory=args.max_inventory,
        title=f"Phase 4 — Risk Controls\n{subtitle}",
    )

    out_path = os.path.join(os.path.dirname(__file__), "phase4_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
