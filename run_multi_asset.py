"""
Multi-Asset Market Making — BTC + ETH with cross-asset inventory hedging.

Compares:
  1. MultiAssetMM — single model posting on both assets with joint reservation
     prices that account for cross-asset portfolio variance.
  2. Two independent AvellanedaStoikov models (one BTC, one ETH) that treat
     each asset in isolation.

Key insight: when BTC and ETH positions happen to offset each other (ρ>0 but
opposite signs), the multi-asset model recognises the hedge and quotes more
patiently. Independent models ignore this and over-trade to reduce each asset's
own inventory independently.

Usage:
    python run_multi_asset.py
    python run_multi_asset.py --seed 7
    python run_multi_asset.py --corr 0.60   # lower correlation
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from simulator.data_gen import generate_correlated_btc_eth
from simulator.multi_asset_mm import MultiAssetMM
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.visualize import plot_multi_asset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0-btc",   type=float, default=50_000.0)
    p.add_argument("--S0-eth",   type=float, default=3_000.0)
    p.add_argument("--corr",     type=float, default=0.85)
    p.add_argument("--n-steps",  type=int,   default=1440)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--no-show",  action="store_true")
    return p.parse_args()


def compute_port_var(inv_btc, inv_eth, corr,
                     sigma_btc_abs=None, sigma_eth_abs=None) -> np.ndarray:
    dt_year = 1.0 / (365.0 * 24.0 * 60.0)
    sb = sigma_btc_abs or 50_000.0 * 0.80 * dt_year ** 0.5
    se = sigma_eth_abs or  3_000.0 * 1.20 * dt_year ** 0.5
    return (
        (inv_btc * sb) ** 2
        + (inv_eth * se) ** 2
        + 2.0 * corr * inv_btc * sb * inv_eth * se
    )


def print_summary_table(multi_res, btc_res, eth_res, corr: float) -> None:
    dt_year = 1.0 / (365.0 * 24.0 * 60.0)
    sb = 50_000.0 * 0.80 * dt_year ** 0.5
    se =  3_000.0 * 1.20 * dt_year ** 0.5

    pv_multi = np.sqrt(compute_port_var(multi_res["inv_btc"],  multi_res["inv_eth"],  corr, sb, se))
    pv_indep = np.sqrt(compute_port_var(btc_res["inventory"],  eth_res["inventory"],  corr, sb, se))

    final_multi  = multi_res["mtm_pnl"].iloc[-1]
    final_indep  = btc_res["mtm_pnl"].iloc[-1] + eth_res["mtm_pnl"].iloc[-1]
    spread_multi = multi_res["spread_pnl"].iloc[-1]
    spread_indep = btc_res["spread_pnl"].iloc[-1] + eth_res["spread_pnl"].iloc[-1]
    inv_corr_m   = multi_res["inv_btc"].corr(multi_res["inv_eth"])
    inv_corr_i   = btc_res["inventory"].corr(eth_res["inventory"])

    print()
    print("=" * 72)
    print(f"  Multi-Asset vs Independent AS  (ρ={corr:.2f})")
    print("=" * 72)
    print(f"  {'Metric':<30} {'Multi-Asset':>14} {'Independent':>14}")
    print("-" * 72)
    print(f"  {'BTC inv std (BTC)':<30} {multi_res['inv_btc'].std():>14.4f} {btc_res['inventory'].std():>14.4f}")
    print(f"  {'ETH inv std (ETH)':<30} {multi_res['inv_eth'].std():>14.4f} {eth_res['inventory'].std():>14.4f}")
    print(f"  {'BTC-ETH inv correlation':<30} {inv_corr_m:>+14.3f} {inv_corr_i:>+14.3f}")
    print(f"  {'Portfolio σ avg ($/√step)':<30} {pv_multi.mean():>14.3f} {pv_indep.mean():>14.3f}")
    print(f"  {'Portfolio σ max ($/√step)':<30} {pv_multi.max():>14.3f} {pv_indep.max():>14.3f}")
    print(f"  {'Spread P&L ($)':<30} {spread_multi:>+14.2f} {spread_indep:>+14.2f}")
    print(f"  {'Total MtM P&L ($)':<30} {final_multi:>+14.2f} {final_indep:>+14.2f}")
    print(f"  {'Max drawdown ($)':<30} {multi_res['mtm_pnl'].min():>+14.2f}"
          f" {(btc_res['mtm_pnl']+eth_res['mtm_pnl']).min():>+14.2f}")
    print("=" * 72)
    pct = (1 - pv_multi.mean() / pv_indep.mean()) * 100
    print(f"\n  Portfolio risk reduction: {pct:.1f}%")
    print(f"  BTC-ETH inventory correlation:")
    print(f"    Multi-asset: {inv_corr_m:+.3f} "
          f"({'negative = hedging ✓' if inv_corr_m < -0.1 else 'not hedged'})")
    print(f"    Independent: {inv_corr_i:+.3f} "
          f"({'both trend together — no hedging' if inv_corr_i > 0.3 else ''})")
    print()


def main() -> None:
    args = parse_args()

    print(f"Generating correlated BTC + ETH price paths  (ρ={args.corr:.2f}, seed={args.seed})…")
    btc_prices, eth_prices = generate_correlated_btc_eth(
        S0_btc=args.S0_btc,
        S0_eth=args.S0_eth,
        corr=args.corr,
        n_steps=args.n_steps,
        seed=args.seed,
    )
    print(f"  BTC: ${btc_prices.min():,.0f}–${btc_prices.max():,.0f}  "
          f"ETH: ${eth_prices.min():,.0f}–${eth_prices.max():,.0f}")

    # -- Multi-asset model ---------------------------------------------------
    multi = MultiAssetMM(
        gamma=0.0165,
        corr=args.corr,
        S0_btc=args.S0_btc,
        S0_eth=args.S0_eth,
        T=args.n_steps,
        lot_size_btc=0.01,
        lot_size_eth=0.10,
        initial_cash=100_000.0,
        seed=args.seed,
    )
    multi_res = multi.run(btc_prices, eth_prices)

    # -- Independent BTC AS --------------------------------------------------
    # gamma_btc_P2 = gamma_MA * sigma_btc_abs^2 ≈ 0.0165 * 55^2 ≈ 50
    indep_btc = AvellanedaStoikov(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=args.n_steps, lot_size=0.01,
        initial_cash=50_000.0, seed=args.seed,
    )
    btc_res = indep_btc.run(btc_prices)

    # -- Independent ETH AS --------------------------------------------------
    # gamma_eth_P2 = gamma_MA * sigma_eth_abs^2 ≈ 0.0165 * 4.97^2 ≈ 0.41
    indep_eth = AvellanedaStoikov(
        gamma=0.41, delta_0=2.7, delta_min=0.3,
        T=args.n_steps, lot_size=0.10,
        fill_k=0.446,
        initial_cash=50_000.0, seed=args.seed,
    )
    eth_res = indep_eth.run(eth_prices)

    print("\nMulti-Asset MM:")
    multi.summary(multi_res)

    print_summary_table(multi_res, btc_res, eth_res, args.corr)

    fig = plot_multi_asset(
        multi_res, btc_res, eth_res,
        corr=args.corr,
        title=f"Multi-Asset MM — BTC + ETH  (ρ={args.corr:.2f}, seed={args.seed})",
    )

    out_path = os.path.join(os.path.dirname(__file__), "multi_asset_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
