"""
Phase 5+ — Exact P&L Attribution.

Runs all four models on the same regime-switching price path and compares:
  - Exact spread P&L: per-fill (mid − fill_price) × lot
  - Approximate spread P&L: half_spread × lot × n_fills  (old method)
  - Exact inventory P&L: Σ inventory × Δmid  (direct, not residual)

Key insight: in the AS model the bid/ask are shifted by the reservation price,
so (mid − bid) ≠ δ when inventory ≠ 0. The approximation overestimates spread
P&L when the pushed side fills more, and underestimates when the other side does.

Usage:
    python run_exact_attribution.py
    python run_exact_attribution.py --seed 7
"""

import argparse
import os

import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price_regime
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM
from simulator.visualize import plot_exact_attribution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0",      type=float, default=50_000.0)
    p.add_argument("--n-steps", type=int,   default=1440)
    p.add_argument("--seed",    type=int,   default=7)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def print_comparison_table(results: dict) -> None:
    print()
    print("=" * 110)
    print(f"  {'Model':<28} {'Exact Spread':>13} {'Approx Spread':>14} {'Error':>10} {'Error%':>8} "
          f"{'Exact Inv':>12} {'Total P&L':>12}")
    print("=" * 110)
    for name, df in results.items():
        exact_s  = df["spread_pnl"].iloc[-1]
        approx_s = df["spread_pnl_approx"].iloc[-1] if "spread_pnl_approx" in df.columns else exact_s
        error    = exact_s - approx_s
        pct      = (error / approx_s * 100) if abs(approx_s) > 1e-6 else 0.0
        exact_i  = df["inventory_pnl"].iloc[-1]
        total    = df["mtm_pnl"].iloc[-1]
        identity_ok = abs(exact_s + exact_i - total) < 0.01
        flag = "✓" if identity_ok else "✗"
        print(f"  {name:<28} {exact_s:>+13,.2f} {approx_s:>+14,.2f} {error:>+10,.2f} {pct:>7.1f}% "
              f"{exact_i:>+12,.2f} {total:>+12,.2f}  {flag}")
    print("=" * 110)
    print("  ✓ = spread_pnl + inventory_pnl == mtm_pnl  (exact attribution identity)")
    print()
    print("  Interpretation:")
    print("  • Baseline: exact = approx (bid = mid−δ, ask = mid+δ, so mid−bid = δ always)")
    print("  • AS+:  approx uses δ×lot×fills; exact uses (mid−bid)×lot per buy,")
    print("          (ask−mid)×lot per sell. When holding inventory, the 'pushed' side")
    print("          captures δ ± γ·q·τ — more or less than δ depending on side and sign.")
    print()


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

    b  = BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared)
    b_res = b.run(prices)
    # Baseline has no approx column — add it as alias for comparison code
    b_res["spread_pnl_approx"] = b_res["spread_pnl"]

    a  = AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
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

    print_comparison_table(all_results)

    fig = plot_exact_attribution(
        all_results, regimes,
        title=f"Exact P&L Attribution  (seed={args.seed})",
    )

    out_path = os.path.join(os.path.dirname(__file__), "exact_attribution_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
