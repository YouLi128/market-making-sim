"""
Compare simplified AS (Phase 2) vs exact Avellaneda-Stoikov on real BTC data.

Usage:
    python run_exact_as.py
    python run_exact_as.py --date 2024-06-10
    python run_exact_as.py --no-show
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from simulator.data_loader import fetch_btc_price
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.exact_as import ExactAvellanedaStoikov, calibrate_as_params


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
    S0 = prices.iloc[0]
    print(f"  {len(prices):,} bars  |  ${prices.min():,.0f}–${prices.max():,.0f}  "
          f"|  move {(prices.iloc[-1]/S0-1)*100:+.2f}%")

    # Per-step absolute vol from real data
    import numpy as np
    log_rets  = np.diff(np.log(prices.values))
    sigma_per_step = log_rets.std() * S0                          # $/√step for AS formula
    sigma_annual   = log_rets.std() * np.sqrt(365 * 24 * 60)    # fraction/yr for display
    print(f"\n  Realised σ: ${sigma_per_step:.2f}/√step  ({sigma_annual:.0%}/yr annual)")

    gamma, kappa = calibrate_as_params(
        target_spread=50.0, target_skew=25.0,
        sigma_abs=sigma_per_step, T=args.n_steps,
    )
    print(f"  Calibrated: γ = {gamma:.3e}  κ = {kappa:.4f}")
    print(f"  Fill prob at $50: {np.exp(-kappa*50):.1%}  "
          f"(simplified: ~30%)")

    shared = dict(lot_size=0.01, initial_cash=S0 * 2, seed=0)

    print("\nRunning simplified AS (Phase 2)…")
    simple = AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
                               T=args.n_steps, **shared)
    simple_res = simple.run(prices)

    print("Running exact AS (paper formula + calibrated params)…")
    exact = ExactAvellanedaStoikov(target_spread=50.0, target_skew=25.0,
                                   sigma_abs=sigma_per_step,
                                   T=args.n_steps, **shared)
    exact_res = exact.run(prices)
    exact.summary(exact_res)

    # Summary
    print("\n" + "=" * 62)
    print(f"  {'':32} {'Simplified':>12} {'Exact AS':>12}")
    print("=" * 62)
    rows = [
        ("Total fills",         len(simple.trades),          len(exact.trades)),
        ("Final P&L ($)",       simple_res["mtm_pnl"].iloc[-1],  exact_res["mtm_pnl"].iloc[-1]),
        ("Spread P&L ($)",      simple_res["spread_pnl"].iloc[-1],exact_res["spread_pnl"].iloc[-1]),
        ("Max |inventory| BTC", simple_res["inventory"].abs().max(), exact_res["inventory"].abs().max()),
        ("Inventory std BTC",   simple_res["inventory"].std(),  exact_res["inventory"].std()),
    ]
    for label, sv, ev in rows:
        if isinstance(sv, float):
            print(f"  {label:<32} {sv:>+12.2f} {ev:>+12.2f}")
        else:
            print(f"  {label:<32} {sv:>12,} {ev:>12,}")
    print("=" * 62)

    # Plot
    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(4, 1, hspace=0.42, top=0.93, bottom=0.05)
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)]

    # Panel 1: half-spread comparison
    ax1.plot(simple_res.index, simple_res["half_spread"],
             color="steelblue", lw=1.0, label="Simplified AS  δ = δ₀·τ + δ_min")
    ax1.plot(exact_res.index, exact_res["half_spread"],
             color="firebrick", lw=1.0, label="Exact AS  δ* = γσ²τ/2 + (1/γ)ln(1+γ/κ)")
    ax1.set_ylabel("Half-spread (USD)", fontsize=9)
    ax1.set_title("Spread Formula Comparison  —  exact AS narrows faster toward session end", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # Panel 2: reservation price vs mid
    ax2.plot(prices.index, prices.values, color="grey", lw=0.8, alpha=0.6, label="Mid price")
    ax2.plot(simple_res.index, simple_res["reservation_price"],
             color="steelblue", lw=1.0, label="Simplified r")
    ax2.plot(exact_res.index, exact_res["reservation_price"],
             color="firebrick", lw=1.0, label="Exact r")
    ax2.set_ylabel("Price (USD)", fontsize=9)
    ax2.set_title("Reservation Price  —  both models skew away from mid when inventory builds", fontsize=10)
    ax2.legend(fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelbottom=False)

    # Panel 3: inventory
    ax3.plot(simple_res.index, simple_res["inventory"],
             color="steelblue", lw=1.0, label=f"Simplified  std={simple_res['inventory'].std():.4f}")
    ax3.plot(exact_res.index, exact_res["inventory"],
             color="firebrick", lw=1.0, label=f"Exact AS    std={exact_res['inventory'].std():.4f}")
    ax3.axhline(0, color="black", lw=0.6, linestyle="--")
    ax3.set_ylabel("Inventory (BTC)", fontsize=9)
    ax3.set_title("Inventory", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)
    ax3.tick_params(labelbottom=False)

    # Panel 4: P&L
    ax4.plot(simple_res.index, simple_res["mtm_pnl"],
             color="steelblue", lw=1.2, label="Simplified AS")
    ax4.plot(exact_res.index, exact_res["mtm_pnl"],
             color="firebrick", lw=1.2, label="Exact AS")
    ax4.axhline(0, color="black", lw=0.6, linestyle="--")
    ax4.set_ylabel("MtM P&L (USD)", fontsize=9)
    ax4.set_title("P&L  (note: different fill rates make direct comparison approximate)", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2)

    date_str = args.date or prices.index[0].strftime("%Y-%m-%d")
    fig.suptitle(f"Simplified AS vs Exact Avellaneda-Stoikov — Real BTC  {date_str}",
                 fontsize=13, fontweight="bold")

    out_path = os.path.join(os.path.dirname(__file__), "exact_as_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
