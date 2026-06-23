"""
Baseline market maker: post bid = mid - delta, ask = mid + delta.

Fill model: at each minute-bar, a buy order and a sell order each arrive
independently with probability `prob_fill`. This decouples fill arrival from
price direction — the adverse selection shows up purely through inventory
mark-to-market as prices drift against accumulated positions.

P&L decomposition:
    spread_pnl     = delta * lot_size * total_fills      (monetised edge)
    inventory_pnl  = mtm_pnl - spread_pnl               (directional exposure)
    mtm_pnl        = cash + inventory * mid - initial_cash
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str        # 'buy' | 'sell'
    price: float
    qty: float


class BaselineMarketMaker:
    def __init__(
        self,
        delta: float = 50.0,
        lot_size: float = 0.01,
        prob_fill: float = 0.30,
        initial_cash: float = 100_000.0,
        seed: int = 0,
    ):
        self.delta = delta
        self.lot_size = lot_size
        self.prob_fill = prob_fill

        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.inventory: float = 0.0
        self._cumulative_spread_pnl: float = 0.0
        self._cumulative_inventory_pnl: float = 0.0
        self._prev_mid: float = 0.0
        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def step(self, timestamp: pd.Timestamp, mid: float) -> dict:
        # Exact inventory MTM from price move (before fills)
        self._cumulative_inventory_pnl += self.inventory * (mid - self._prev_mid)

        bid = mid - self.delta
        ask = mid + self.delta

        bid_fill = self._rng.random() < self.prob_fill
        ask_fill = self._rng.random() < self.prob_fill

        if bid_fill:
            self.cash -= bid * self.lot_size
            self.inventory += self.lot_size
            self._cumulative_spread_pnl += (mid - bid) * self.lot_size
            self.trades.append(Trade(timestamp, "buy", bid, self.lot_size))

        if ask_fill:
            self.cash += ask * self.lot_size
            self.inventory -= self.lot_size
            self._cumulative_spread_pnl += (ask - mid) * self.lot_size
            self.trades.append(Trade(timestamp, "sell", ask, self.lot_size))

        self._prev_mid = mid
        mtm_pnl = self.cash + self.inventory * mid - self._initial_cash

        return {
            "mid_price": mid,
            "bid": bid,
            "ask": ask,
            "inventory": self.inventory,
            "cash": self.cash,
            "spread_pnl": self._cumulative_spread_pnl,
            "inventory_pnl": self._cumulative_inventory_pnl,
            "mtm_pnl": mtm_pnl,
        }

    # ------------------------------------------------------------------
    def run(self, prices: pd.Series) -> pd.DataFrame:
        records = [self.step(ts, p) for ts, p in prices.items()]
        df = pd.DataFrame(records, index=prices.index)
        return df

    # ------------------------------------------------------------------
    def summary(self, results: pd.DataFrame) -> None:
        buys  = sum(1 for t in self.trades if t.side == "buy")
        sells = sum(1 for t in self.trades if t.side == "sell")
        final = results.iloc[-1]

        print("=" * 42)
        print("  Baseline Market Maker — Results")
        print("=" * 42)
        print(f"  Timesteps simulated : {len(results):,}")
        print(f"  Fills (buy / sell)  : {buys:,} / {sells:,}  ({buys+sells:,} total)")
        print(f"  Final inventory     : {final['inventory']:+.4f} BTC")
        print(f"  Max |inventory|     : {results['inventory'].abs().max():.4f} BTC")
        print(f"  Spread P&L          : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L       : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total (MtM) P&L     : ${final['mtm_pnl']:+,.2f}")
        print(f"  Peak P&L            : ${results['mtm_pnl'].max():+,.2f}")
        print(f"  Max drawdown        : ${results['mtm_pnl'].min():+,.2f}")
        print("=" * 42)
