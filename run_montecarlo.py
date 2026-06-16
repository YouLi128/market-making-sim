"""
Monte Carlo Backtest — runs all four models across multiple price paths.

Default: real historical BTC/USDT data from Binance (90 days).
         First run downloads and caches data; subsequent runs are instant.
Fallback: synthetic GBM paths via --synthetic flag.

Usage:
    python run_montecarlo.py                  # real data, last 90 days
    python run_montecarlo.py --n-days 180     # real data, last 6 months
    python run_montecarlo.py --synthetic      # GBM simulation (fast)
    python run_montecarlo.py --no-show
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from simulator.data_loader import fetch_historical_days
from simulator.data_gen import generate_btc_price_regime
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM
from simulator.visualize import plot_montecarlo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-days",   type=int,  default=90,    help="Days of real data (default 90)")
    p.add_argument("--n-paths",  type=int,  default=500,   help="Paths for --synthetic mode")
    p.add_argument("--n-steps",  type=int,  default=1440)
    p.add_argument("--synthetic",action="store_true",      help="Use GBM instead of real data")
    p.add_argument("--no-show",  action="store_true")
    return p.parse_args()


def run_one_path(prices, seed=0):
    n = len(prices)
    S0 = float(prices.iloc[0])
    shared = dict(lot_size=0.01, initial_cash=S0 * 2, seed=seed)

    models = {
        "Baseline": BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared),
        "AS":       AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                      T=n, **shared),
        "Phase3":   AdverseSelectionMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                       T=n, toxicity_window=30, toxicity_mult=2.0,
                                       **shared),
        "Phase4":   RiskControlMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                   T=n, toxicity_window=30, toxicity_mult=2.0,
                                   max_inventory=0.10, vol_threshold=1.2,
                                   vol_spread_mult=1.3, max_holding_steps=120,
                                   max_inventory_loss=150.0, **shared),
    }

    results = {}
    for name, mm in models.items():
        df = mm.run(prices)
        results[name] = {
            "final_pnl":    df["mtm_pnl"].iloc[-1],
            "max_drawdown": df["mtm_pnl"].min(),
            "inv_std":      df["inventory"].std(),
        }
    return results


def main():
    args = parse_args()
    stats = {name: {"final_pnl": [], "max_drawdown": [], "inv_std": []}
             for name in ["Baseline", "AS", "Phase3", "Phase4"]}

    if args.synthetic:
        # ── GBM mode ───────────────────────────────────────────────────
        N = args.n_paths
        print(f"Running {N} synthetic GBM paths…")
        for i in range(N):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{N}")
            prices, _ = generate_btc_price_regime(
                S0=50_000.0, n_steps=args.n_steps, seed=i)
            for name, metrics in run_one_path(prices, seed=i).items():
                for key, val in metrics.items():
                    stats[name][key].append(val)
        title = f"Monte Carlo — Synthetic GBM  ({N} paths)"

    else:
        # ── Real data mode ─────────────────────────────────────────────
        print(f"Fetching {args.n_days} days of real BTC/USDT data…")
        print("  (cached days load instantly, new days download from Binance)")
        days = fetch_historical_days(n_days=args.n_days)
        N = len(days)
        print(f"  {N} valid days loaded\n")

        for i, prices in enumerate(days):
            date_str = prices.index[0].strftime("%Y-%m-%d")
            move = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:3d}/{N}] {date_str}  {move:+.2f}%")
            for name, metrics in run_one_path(prices, seed=i).items():
                for key, val in metrics.items():
                    stats[name][key].append(val)

        start = days[0].index[0].strftime("%Y-%m-%d")
        end   = days[-1].index[0].strftime("%Y-%m-%d")
        title = f"Monte Carlo — Real BTC/USDT  ({N} days  {start} → {end})"

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  {'Model':<20} {'P&L mean':>10} {'P&L std':>10} {'Sharpe':>8} "
          f"{'MaxDD mean':>12} {'InvStd mean':>12}")
    print("=" * 80)
    for name in stats:
        pnl    = np.array(stats[name]["final_pnl"])
        mdd    = np.array(stats[name]["max_drawdown"])
        inv    = np.array(stats[name]["inv_std"])
        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        win    = (pnl > 0).mean()
        print(f"  {name:<20} {pnl.mean():>+10.2f} {pnl.std():>10.2f} {sharpe:>8.3f} "
              f"{mdd.mean():>+12.2f} {inv.mean():>12.4f}  win={win:.0%}")
    print("=" * 80)

    fig = plot_montecarlo(stats, n_paths=N, title=title)

    out_path = os.path.join(os.path.dirname(__file__), "montecarlo_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
