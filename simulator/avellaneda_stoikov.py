"""
Phase 2 — Avellaneda-Stoikov (AS) market making model.

Two key improvements over the baseline:

1. Reservation price — quotes shift away from mid proportional to inventory:
       r = mid - gamma * q * tau
   When long (q > 0): r < mid → ask moves closer to mid → easier to fill → inventory sells down.
   When short (q < 0): r > mid → bid moves closer to mid → inventory buys back.

2. Time-decaying spread — spread narrows as the session ends (urgency rises):
       delta(tau) = delta_0 * tau + delta_min
   Near end of session the maker accepts tighter margins to flatten inventory.

Fill model (key difference from baseline):
   Fill probability decays exponentially with distance from mid.
   This is what actually creates inventory mean-reversion:
       p_fill = prob_fill * exp(k * (delta(tau) - |quote_dist_from_mid|))
   When quote is closer to mid than delta(tau): p > prob_fill  (fills more)
   When quote is further from mid:              p < prob_fill  (fills less)
   When inventory=0 (symmetric quotes):         p = prob_fill  (same as baseline)

Parameters:
   gamma     — $/BTC: how far the reservation price shifts per BTC of inventory
               (at tau=1.0). Default 50 → 0.5 BTC inventory causes $25 skew at session midpoint.
   delta_0   — maximum half-spread added at session start (narrows to delta_min at end).
   delta_min — minimum half-spread (floor, prevents quotes from crossing).
   k         — fill sensitivity to quote distance. Calibrated so that at distance=50
               from mid, fill prob = prob_fill (matching the baseline's behaviour).

Reference: Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
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


class AvellanedaStoikov:
    def __init__(
        self,
        gamma: float = 50.0,      # $/BTC — inventory skew sensitivity at tau=1
        delta_0: float = 45.0,    # additional half-spread at session start
        delta_min: float = 5.0,   # minimum half-spread at session end
        T: int = 1440,            # total session steps
        lot_size: float = 0.01,
        prob_fill: float = 0.30,
        fill_k: float = 0.024,
        baseline_sigma: float = 0.80,  # reference vol used during calibration
        initial_cash: float = 100_000.0,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.baseline_sigma = baseline_sigma
        self.delta_0 = delta_0
        self.delta_min = delta_min
        self.T = T
        self.lot_size = lot_size
        self.prob_fill = prob_fill
        self.fill_k = fill_k

        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.inventory: float = 0.0
        self._step: int = 0
        self._cumulative_spread_pnl: float = 0.0
        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _tau(self) -> float:
        return max(self.T - self._step, 1) / self.T

    def _half_spread(self, tau: float) -> float:
        return self.delta_0 * tau + self.delta_min

    def _fill_prob(self, delta: float, quote_dist: float) -> float:
        """Fill probability given current half-spread and distance of quote from mid."""
        return min(self.prob_fill * np.exp(self.fill_k * (delta - quote_dist)), 1.0)

    # ------------------------------------------------------------------
    def step(self, timestamp: pd.Timestamp, mid: float) -> dict:
        tau = self._tau()
        self._step += 1

        # Reservation price: shift away from mid proportional to inventory
        r = mid - self.gamma * self.inventory * tau

        # Time-decaying half-spread around reservation price
        delta = self._half_spread(tau)

        bid = r - delta
        ask = r + delta

        # Fill probabilities respond to quote distance from mid
        ask_dist = max(ask - mid, 0.0)
        bid_dist = max(mid - bid, 0.0)
        p_ask = self._fill_prob(delta, ask_dist)
        p_bid = self._fill_prob(delta, bid_dist)

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
        self._cumulative_spread_pnl += delta * self.lot_size * n_fills

        mtm_pnl = self.cash + self.inventory * mid - self._initial_cash
        inventory_pnl = mtm_pnl - self._cumulative_spread_pnl

        return {
            "mid_price": mid,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "half_spread": delta,
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

    def run_garch(self, prices: pd.Series, sigma_series: pd.Series) -> pd.DataFrame:
        """Run with GARCH-estimated sigma at each step."""
        sigma_aligned = sigma_series.reindex(prices.index).ffill().fillna(self.baseline_sigma)
        records = []
        for ts, p in prices.items():
            sigma_t   = float(sigma_aligned.loc[ts])
            vol_ratio = sigma_t / self.baseline_sigma
            # Scale gamma and delta_0 by vol ratio so model responds to vol clustering
            orig_gamma, orig_delta_0 = self.gamma, self.delta_0
            self.gamma   = orig_gamma   * vol_ratio
            self.delta_0 = orig_delta_0 * vol_ratio
            rec = self.step(ts, p)
            rec["garch_sigma"] = sigma_t
            rec["vol_ratio"]   = vol_ratio
            self.gamma, self.delta_0 = orig_gamma, orig_delta_0
            records.append(rec)
        return pd.DataFrame(records, index=prices.index)

    # ------------------------------------------------------------------
    def summary(self, results: pd.DataFrame) -> None:
        buys  = sum(1 for t in self.trades if t.side == "buy")
        sells = sum(1 for t in self.trades if t.side == "sell")
        final = results.iloc[-1]

        print("=" * 42)
        print("  Avellaneda-Stoikov — Results")
        print("=" * 42)
        print(f"  Timesteps simulated : {len(results):,}")
        print(f"  Fills (buy / sell)  : {buys:,} / {sells:,}  ({buys+sells:,} total)")
        print(f"  Final inventory     : {final['inventory']:+.4f} BTC")
        print(f"  Max |inventory|     : {results['inventory'].abs().max():.4f} BTC")
        print(f"  Inventory std dev   : {results['inventory'].std():.4f} BTC")
        print(f"  Spread P&L          : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L       : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total (MtM) P&L     : ${final['mtm_pnl']:+,.2f}")
        print(f"  Peak P&L            : ${results['mtm_pnl'].max():+,.2f}")
        print(f"  Max drawdown        : ${results['mtm_pnl'].min():+,.2f}")
        print("=" * 42)
