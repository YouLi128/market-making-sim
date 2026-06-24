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
    python run_multi_asset.py                        # synthetic correlated GBM
    python run_multi_asset.py --real-data            # real Binance data + rolling ρ
    python run_multi_asset.py --real-data --date 2024-06-01
    python run_multi_asset.py --seed 7
    python run_multi_asset.py --corr 0.60            # lower correlation (synthetic only)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator.data_gen import generate_correlated_btc_eth
from simulator.multi_asset_mm import MultiAssetMM
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.visualize import plot_multi_asset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--S0-btc",     type=float, default=50_000.0)
    p.add_argument("--S0-eth",     type=float, default=3_000.0)
    p.add_argument("--corr",       type=float, default=0.85,
                   help="Fixed correlation for synthetic mode (ignored in --real-data mode)")
    p.add_argument("--n-steps",    type=int,   default=1440)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--real-data",  action="store_true",
                   help="Fetch real BTC+ETH prices from Binance and use rolling 60-min ρ")
    p.add_argument("--date",       type=str,   default=None,
                   help="YYYY-MM-DD to fetch (default: latest). Only used with --real-data")
    p.add_argument("--roll-window", type=int,  default=60,
                   help="Rolling window in minutes for ρ estimation (default: 60)")
    p.add_argument("--no-show",    action="store_true")
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


def compute_rolling_corr(btc_prices: pd.Series, eth_prices: pd.Series,
                          window: int = 60, fallback: float = 0.85) -> pd.Series:
    """Compute rolling Pearson ρ of 1-min log returns, fill NaN with fallback."""
    lr_btc = np.log(btc_prices / btc_prices.shift(1))
    lr_eth = np.log(eth_prices / eth_prices.shift(1))
    rolling_corr = lr_btc.rolling(window).corr(lr_eth)
    rolling_corr = rolling_corr.clip(-0.9999, 0.9999).fillna(fallback)
    return rolling_corr


def main() -> None:
    args = parse_args()

    corr_series = None   # None → fixed ρ; Series → time-varying ρ
    effective_corr = args.corr  # for plot title / summary table

    if args.real_data:
        from simulator.data_loader import fetch_btc_eth_prices
        date_str = args.date or "latest"
        print(f"Fetching real BTC + ETH 1-min prices from Binance  (date={date_str})…")
        btc_prices, eth_prices = fetch_btc_eth_prices(
            date=args.date, n_steps=args.n_steps
        )
        print(f"  Fetched {len(btc_prices)} bars")
        print(f"  BTC: ${btc_prices.min():,.0f}–${btc_prices.max():,.0f}  "
              f"ETH: ${eth_prices.min():,.2f}–${eth_prices.max():,.2f}")

        corr_series = compute_rolling_corr(btc_prices, eth_prices,
                                           window=args.roll_window,
                                           fallback=args.corr)
        effective_corr = float(corr_series.mean())
        print(f"  Rolling ρ ({args.roll_window}-min window):  "
              f"mean={effective_corr:.3f}  min={corr_series.min():.3f}  "
              f"max={corr_series.max():.3f}")
        mode_label = f"real data, rolling ρ (avg={effective_corr:.2f})"
    else:
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
        mode_label = f"ρ={args.corr:.2f}, seed={args.seed}"

    n_steps = len(btc_prices)

    # -- Multi-asset model ---------------------------------------------------
    multi = MultiAssetMM(
        gamma=0.0165,
        corr=args.corr,            # starting ρ (overridden per-step by corr_series)
        S0_btc=float(btc_prices.iloc[0]),
        S0_eth=float(eth_prices.iloc[0]),
        T=n_steps,
        lot_size_btc=0.01,
        lot_size_eth=0.10,
        initial_cash=100_000.0,
        seed=args.seed,
    )
    multi_res = multi.run(btc_prices, eth_prices, corr_series=corr_series)

    # -- Independent BTC AS --------------------------------------------------
    indep_btc = AvellanedaStoikov(
        gamma=50.0, delta_0=45.0, delta_min=5.0,
        T=n_steps, lot_size=0.01,
        initial_cash=50_000.0, seed=args.seed,
    )
    btc_res = indep_btc.run(btc_prices)

    # -- Independent ETH AS --------------------------------------------------
    indep_eth = AvellanedaStoikov(
        gamma=0.41, delta_0=2.7, delta_min=0.3,
        T=n_steps, lot_size=0.10,
        fill_k=0.446,
        initial_cash=50_000.0, seed=args.seed,
    )
    eth_res = indep_eth.run(eth_prices)

    print("\nMulti-Asset MM:")
    multi.summary(multi_res)

    print_summary_table(multi_res, btc_res, eth_res, effective_corr)

    suffix = "real" if args.real_data else f"seed{args.seed}"
    fig = plot_multi_asset(
        multi_res, btc_res, eth_res,
        corr=effective_corr,
        corr_series=corr_series,
        title=f"Multi-Asset MM — BTC + ETH  ({mode_label})",
    )

    out_path = os.path.join(os.path.dirname(__file__), f"multi_asset_results_{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
