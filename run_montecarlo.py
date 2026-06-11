"""
Monte Carlo Backtest — runs all four models across N random price paths.

Usage:
    python run_montecarlo.py              # 500 paths (default)
    python run_montecarlo.py --n-paths 200
    python run_montecarlo.py --no-show
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from simulator.data_gen import generate_btc_price_regime
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM
from simulator.visualize import plot_montecarlo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int,   default=500)
    p.add_argument("--n-steps", type=int,   default=1440)
    p.add_argument("--S0",      type=float, default=50_000.0)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def run_one_path(seed, n_steps, S0):
    prices, _ = generate_btc_price_regime(S0=S0, n_steps=n_steps, seed=seed)
    shared = dict(lot_size=0.01, initial_cash=100_000.0, seed=seed)

    models = {
        "Baseline": BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared),
        "AS":       AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                      T=n_steps, **shared),
        "Phase3":   AdverseSelectionMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                       T=n_steps, toxicity_window=30, toxicity_mult=2.0,
                                       **shared),
        "Phase4":   RiskControlMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                   T=n_steps, toxicity_window=30, toxicity_mult=2.0,
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
    N = args.n_paths

    # Aggregate stats
    stats = {name: {"final_pnl": [], "max_drawdown": [], "inv_std": []}
             for name in ["Baseline", "AS", "Phase3", "Phase4"]}

    print(f"Running {N} paths across 4 models…")
    for i in range(N):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}")
        path_results = run_one_path(seed=i, n_steps=args.n_steps, S0=args.S0)
        for name, metrics in path_results.items():
            for key, val in metrics.items():
                stats[name][key].append(val)

    # Summary table
    print("\n" + "=" * 80)
    print(f"  {'Model':<20} {'P&L mean':>10} {'P&L std':>10} {'Sharpe':>8} "
          f"{'MaxDD mean':>12} {'InvStd mean':>12}")
    print("=" * 80)
    for name in stats:
        pnl = np.array(stats[name]["final_pnl"])
        mdd = np.array(stats[name]["max_drawdown"])
        inv = np.array(stats[name]["inv_std"])
        sharpe = pnl.mean() / (pnl.std() + 1e-9)
        print(f"  {name:<20} {pnl.mean():>+10.2f} {pnl.std():>10.2f} {sharpe:>8.3f} "
              f"{mdd.mean():>+12.2f} {inv.mean():>12.4f}")
    print("=" * 80)

    fig = plot_montecarlo(stats, n_paths=N,
                          title="Monte Carlo Backtest — Regime-Switching BTC")

    out_path = os.path.join(os.path.dirname(__file__), "montecarlo_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
