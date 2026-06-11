"""
Phase 3 — Adverse Selection Detection.

Builds on the AS model by adding a real-time toxicity detector:

    toxicity = |fraction_of_up_moves_in_window - 0.5| × 2

    = 0  →  price moves randomly (uninformed flow, safe to quote normally)
    = 1  →  price moves all in one direction (informed flow, widen spread)

When toxicity is high, the effective spread widens:

    delta_eff = delta_base × (1 + toxicity_mult × toxicity)

Wider spread → quotes move further from mid → fill probability drops →
we make fewer trades when informed traders are active → less adverse selection loss.

Fill model (absolute decay, consistent across all phases):
    p_fill = exp(-fill_k × |quote - mid|)
    Calibrated: at distance $50, p_fill = 0.30.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str
    price: float
    qty: float


class AdverseSelectionMM:
    def __init__(
        self,
        gamma: float = 50.0,
        delta_0: float = 45.0,
        delta_min: float = 5.0,
        T: int = 1440,
        lot_size: float = 0.01,
        fill_k: float = 0.024,       # fill decay per $: exp(-fill_k * dist), gives 0.30 at $50
        toxicity_window: int = 30,   # rolling window for toxicity score
        toxicity_mult: float = 2.0,  # how much to widen when toxicity=1: delta × (1 + mult)
        initial_cash: float = 100_000.0,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.delta_0 = delta_0
        self.delta_min = delta_min
        self.T = T
        self.lot_size = lot_size
        self.fill_k = fill_k
        self.toxicity_window = toxicity_window
        self.toxicity_mult = toxicity_mult

        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.inventory: float = 0.0
        self._step: int = 0
        self._cumulative_spread_pnl: float = 0.0
        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)
        self._mid_history: List[float] = []

    # ------------------------------------------------------------------
    def _tau(self) -> float:
        return max(self.T - self._step, 1) / self.T

    def _half_spread_base(self, tau: float) -> float:
        return self.delta_0 * tau + self.delta_min

    def _fill_prob(self, quote_dist: float) -> float:
        """Absolute exponential decay — independent of current spread."""
        return min(float(np.exp(-self.fill_k * quote_dist)), 1.0)

    def _toxicity_score(self) -> float:
        n = min(len(self._mid_history), self.toxicity_window)
        if n < 5:
            return 0.0
        recent = self._mid_history[-n:]
        up_frac = np.mean(np.diff(recent) > 0)
        return float(abs(up_frac - 0.5) * 2)   # 0 = random, 1 = all one direction

    # ------------------------------------------------------------------
    def step(self, timestamp: pd.Timestamp, mid: float) -> dict:
        self._mid_history.append(mid)
        tau = self._tau()
        self._step += 1

        toxicity = self._toxicity_score()

        r = mid - self.gamma * self.inventory * tau

        delta_base = self._half_spread_base(tau)
        delta_eff  = delta_base * (1.0 + self.toxicity_mult * toxicity)

        bid = r - delta_eff
        ask = r + delta_eff

        p_bid = self._fill_prob(max(mid - bid, 0.0))
        p_ask = self._fill_prob(max(ask - mid, 0.0))

        bid_fill = self._rng.random() < p_bid
        ask_fill = self._rng.random() < p_ask

        if bid_fill:
            self.cash -= bid * self.lot_size
            self.inventory += self.lot_size
            self.trades.append(Trade(timestamp, "buy", bid, self.lot_size))

        if ask_fill:
            self.cash += ask * self.lot_size
            self.inventory -= self.lot_size
            self.trades.append(Trade(timestamp, "sell", ask, self.lot_size))

        n_fills = int(bid_fill) + int(ask_fill)
        self._cumulative_spread_pnl += delta_eff * self.lot_size * n_fills

        mtm_pnl = self.cash + self.inventory * mid - self._initial_cash
        inventory_pnl = mtm_pnl - self._cumulative_spread_pnl

        return {
            "mid_price": mid,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "delta_base": delta_base,
            "delta_eff": delta_eff,
            "toxicity": toxicity,
            "inventory": self.inventory,
            "cash": self.cash,
            "spread_pnl": self._cumulative_spread_pnl,
            "inventory_pnl": inventory_pnl,
            "mtm_pnl": mtm_pnl,
        }

    # ------------------------------------------------------------------
    def run(self, prices: pd.Series) -> pd.DataFrame:
        records = [self.step(ts, p) for ts, p in prices.items()]
        return pd.DataFrame(records, index=prices.index)

    # ------------------------------------------------------------------
    def summary(self, results: pd.DataFrame) -> None:
        buys  = sum(1 for t in self.trades if t.side == "buy")
        sells = sum(1 for t in self.trades if t.side == "sell")
        final = results.iloc[-1]

        print("=" * 42)
        print("  Adverse Selection MM — Results")
        print("=" * 42)
        print(f"  Timesteps simulated : {len(results):,}")
        print(f"  Fills (buy / sell)  : {buys:,} / {sells:,}  ({buys+sells:,} total)")
        print(f"  Avg toxicity score  : {results['toxicity'].mean():.3f}")
        print(f"  Time in toxic regime: {(results['toxicity'] > 0.4).mean():.1%}")
        print(f"  Final inventory     : {final['inventory']:+.4f} BTC")
        print(f"  Max |inventory|     : {results['inventory'].abs().max():.4f} BTC")
        print(f"  Inventory std dev   : {results['inventory'].std():.4f} BTC")
        print(f"  Spread P&L          : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L       : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total (MtM) P&L     : ${final['mtm_pnl']:+,.2f}")
        print(f"  Peak P&L            : ${results['mtm_pnl'].max():+,.2f}")
        print(f"  Max drawdown        : ${results['mtm_pnl'].min():+,.2f}")
        print("=" * 42)
