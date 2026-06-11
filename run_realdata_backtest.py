"""
Multi-day backtest on real BTC/USDT data from Binance.
Fetches the last N calendar days and runs all four models on each.

Usage:
    python run_realdata_backtest.py            # last 7 days
    python run_realdata_backtest.py --days 14
    python run_realdata_backtest.py --no-show
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from simulator.data_loader import fetch_btc_price
from simulator.baseline_mm import BaselineMarketMaker
from simulator.avellaneda_stoikov import AvellanedaStoikov
from simulator.adverse_selection import AdverseSelectionMM
from simulator.risk_controls import RiskControlMM


COLORS = {"Baseline": "grey", "AS": "steelblue", "Phase3": "darkorange", "Phase4": "firebrick"}
MODELS = ["Baseline", "AS", "Phase3", "Phase4"]


def run_day(date_str: str) -> dict | None:
    """Run all models on one calendar day. Returns None if data unavailable."""
    try:
        prices = fetch_btc_price(date=date_str, n_steps=1440)
    except Exception as e:
        print(f"  [{date_str}] fetch failed: {e}")
        return None

    if len(prices) < 100:
        print(f"  [{date_str}] not enough bars ({len(prices)}), skipping")
        return None

    n = len(prices)
    shared = dict(lot_size=0.01, initial_cash=prices.iloc[0] * 2, seed=0)

    models = {
        "Baseline": BaselineMarketMaker(delta=50.0, prob_fill=0.30, **shared),
        "AS":       AvellanedaStoikov(gamma=50.0, delta_0=45.0, delta_min=5.0, T=n, **shared),
        "Phase3":   AdverseSelectionMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                       T=n, toxicity_window=30, toxicity_mult=2.0, **shared),
        "Phase4":   RiskControlMM(gamma=50.0, delta_0=45.0, delta_min=5.0,
                                   T=n, toxicity_window=30, toxicity_mult=2.0,
                                   max_inventory=0.10, vol_threshold=1.2,
                                   vol_spread_mult=1.3, max_holding_steps=120,
                                   max_inventory_loss=150.0, **shared),
    }

    day_results = {"prices": prices}
    for name, mm in models.items():
        df = mm.run(prices)
        day_results[name] = {
            "df":           df,
            "final_pnl":    df["mtm_pnl"].iloc[-1],
            "max_drawdown": df["mtm_pnl"].min(),
            "inv_std":      df["inventory"].std(),
            "spread_pnl":   df["spread_pnl"].iloc[-1],
            "inv_pnl":      df["inventory_pnl"].iloc[-1],
        }
    return day_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--days",    type=int, default=7)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(args.days, 0, -1)]

    print(f"Fetching {args.days} days of real BTC/USDT data…\n")

    all_days = {}
    for date_str in dates:
        print(f"  {date_str}…", end=" ", flush=True)
        result = run_day(date_str)
        if result:
            move = (result["prices"].iloc[-1] / result["prices"].iloc[0] - 1) * 100
            print(f"${result['prices'].iloc[0]:,.0f}→${result['prices'].iloc[-1]:,.0f}  {move:+.2f}%")
            all_days[date_str] = result
        else:
            print("skipped")

    if not all_days:
        print("No data fetched.")
        return

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"  {'Date':<12} {'BTC move':>9} | " +
          "  ".join(f"{'P&L '+m:>12}" for m in MODELS))
    print("=" * 95)

    for date_str, day in all_days.items():
        move = (day["prices"].iloc[-1] / day["prices"].iloc[0] - 1) * 100
        pnls = "  ".join(f"{day[m]['final_pnl']:>+12.2f}" for m in MODELS)
        print(f"  {date_str:<12} {move:>+8.2f}% | {pnls}")

    print("=" * 95)

    # Totals
    totals = {m: sum(all_days[d][m]["final_pnl"] for d in all_days) for m in MODELS}
    t_str  = "  ".join(f"{totals[m]:>+12.2f}" for m in MODELS)
    print(f"  {'TOTAL':<12} {'':>9} | {t_str}")
    print("=" * 95)

    # Win rate (days where each model made money)
    wins = {m: sum(1 for d in all_days if all_days[d][m]["final_pnl"] > 0) for m in MODELS}
    w_str = "  ".join(f"{wins[m]:>10}/{len(all_days)}" for m in MODELS)
    print(f"  {'WIN DAYS':<12} {'':>9} | {w_str}")
    print("=" * 95)

    # ── Plot ───────────────────────────────────────────────────────────────
    n_days = len(all_days)
    fig, axes = plt.subplots(n_days, 1, figsize=(14, 3 * n_days),
                             sharex=False, constrained_layout=True)
    if n_days == 1:
        axes = [axes]

    for ax, (date_str, day) in zip(axes, all_days.items()):
        for m in MODELS:
            ax.plot(day[m]["df"].index, day[m]["df"]["mtm_pnl"],
                    color=COLORS[m], lw=1.0, label=m if date_str == dates[0] else "")
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        move = (day["prices"].iloc[-1] / day["prices"].iloc[0] - 1) * 100
        ax.set_title(f"{date_str}  |  BTC {move:+.2f}%  |  "
                     f"P4: ${day['Phase4']['final_pnl']:+.0f}  "
                     f"Baseline: ${day['Baseline']['final_pnl']:+.0f}",
                     fontsize=9)
        ax.set_ylabel("P&L (USD)", fontsize=8)
        ax.grid(True, alpha=0.2)

    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=1.5, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="upper right", fontsize=9, ncol=4)
    fig.suptitle(f"Real BTC Data Backtest — Last {n_days} Days", fontsize=13, fontweight="bold")

    out_path = os.path.join(os.path.dirname(__file__), "realdata_backtest.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
