"""
Phase 4 — Risk Controls.

Builds on Phase 3 (adverse selection detection) with three additional layers:

1. Hard inventory limit
   If inventory >= +max_inventory : block bid fills  (stop buying)
   If inventory <= -max_inventory : block ask fills  (stop selling)
   Prevents runaway inventory in extreme markets.

2. Volatility regime detection
   Rolling realised vol is compared to the expected baseline vol.
   When realised_vol > vol_threshold × baseline_vol:
       delta_final = delta_eff × vol_spread_mult
   Wider spread in high-vol regimes reduces adverse selection exposure.

3. Emergency liquidation — BTC-appropriate (no session close)
   Triggers when EITHER condition is met:
     a) Same-side inventory held > max_holding_steps AND |inventory| > emergency_inv_threshold
     b) Inventory floating P&L < -max_inventory_loss
   When triggered: gamma_eff = gamma × emergency_gamma_mult
   The reservation price is pushed hard away from mid, making it cheap
   for the market to take the inventory off our hands.

   Note: tau is still used in the reservation price and spread formulas
   (represents fraction of the simulation window remaining), but the
   emergency trigger itself no longer depends on tau.

All three mechanisms stack on top of Phase 2 (reservation price, dynamic
spread) and Phase 3 (toxicity-based spread widening).
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


class RiskControlMM:
    def __init__(
        self,
        # ── Phase 2 params ──────────────────────────────────────────────
        gamma: float = 50.0,
        delta_0: float = 45.0,
        delta_min: float = 5.0,
        T: int = 1440,
        lot_size: float = 0.01,
        fill_k: float = 0.024,
        # ── Phase 3 params ──────────────────────────────────────────────
        toxicity_window: int = 30,
        toxicity_mult: float = 2.0,
        # ── Phase 4 params ──────────────────────────────────────────────
        max_inventory: float = 0.10,        # BTC hard limit each side
        vol_window: int = 30,               # steps for realised vol estimate
        baseline_sigma: float = 0.80,       # expected annual vol (for comparison)
        vol_threshold: float = 1.2,         # trigger: realised_vol > vol_threshold × baseline
        vol_spread_mult: float = 1.3,       # spread multiplier in high-vol regime
        # Emergency liquidation — BTC-appropriate (no session close)
        max_holding_steps: int = 120,       # trigger if same-side inventory held > 120 steps
        emergency_inv_threshold: float = 0.05,  # min |inventory| for holding-time trigger
        max_inventory_loss: float = 150.0,  # trigger if inventory P&L < -$150
        emergency_gamma_mult: float = 5.0,  # gamma multiplier during emergency
        # ── shared ──────────────────────────────────────────────────────
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

        self.max_inventory = max_inventory
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.vol_spread_mult = vol_spread_mult
        self.baseline_sigma = baseline_sigma
        self.max_holding_steps = max_holding_steps
        self.emergency_inv_threshold = emergency_inv_threshold
        self.max_inventory_loss = max_inventory_loss
        self.emergency_gamma_mult = emergency_gamma_mult

        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.inventory: float = 0.0
        self._step: int = 0
        self._cumulative_spread_pnl: float = 0.0
        self._cumulative_inventory_pnl: float = 0.0
        self._approx_spread_pnl: float = 0.0
        self._prev_mid: float = 0.0
        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)
        self._mid_history: List[float] = []
        self._holding_steps: int = 0        # steps since inventory last crossed zero
        self._last_inv_sign: int = 0        # +1 / -1 / 0

    # ------------------------------------------------------------------
    def _tau(self) -> float:
        return max(self.T - self._step, 1) / self.T

    def _update_holding_steps(self) -> None:
        sign = 1 if self.inventory > self.emergency_inv_threshold else (
               -1 if self.inventory < -self.emergency_inv_threshold else 0)
        if sign == 0 or sign != self._last_inv_sign:
            self._holding_steps = 0
        else:
            self._holding_steps += 1
        self._last_inv_sign = sign

    def _fill_prob(self, quote_dist: float) -> float:
        return min(float(np.exp(-self.fill_k * quote_dist)), 1.0)

    def _toxicity_score(self) -> float:
        n = min(len(self._mid_history), self.toxicity_window)
        if n < 5:
            return 0.0
        recent = self._mid_history[-n:]
        up_frac = np.mean(np.diff(recent) > 0)
        return float(abs(up_frac - 0.5) * 2)

    def _realised_vol_annual(self) -> float:
        """Rolling annualised realised vol (fraction, not dollar)."""
        n = min(len(self._mid_history), self.vol_window)
        if n < 5:
            return self.baseline_sigma
        recent = np.array(self._mid_history[-n:])
        log_rets = np.diff(np.log(recent))
        dt_year = 1.0 / (365.0 * 24.0 * 60.0)
        return float(np.std(log_rets) / np.sqrt(dt_year))

    # ------------------------------------------------------------------
    def step(self, timestamp: pd.Timestamp, mid: float) -> dict:
        # Exact inventory MTM from price move (before fills)
        self._cumulative_inventory_pnl += self.inventory * (mid - self._prev_mid)

        self._mid_history.append(mid)
        tau = self._tau()
        self._step += 1

        # ── toxicity (Phase 3) ─────────────────────────────────────────
        toxicity = self._toxicity_score()

        # ── realised vol (Phase 4 #2) ──────────────────────────────────
        realised_vol = self._realised_vol_annual()
        high_vol = realised_vol > self.vol_threshold * self.baseline_sigma

        # ── emergency mode (Phase 4 #3) — BTC-appropriate ─────────────
        # Inventory P&L before this step's trades (for loss check)
        inv_pnl_now = self.cash + self.inventory * mid - self._initial_cash - self._cumulative_spread_pnl
        self._update_holding_steps()

        holding_too_long = (self._holding_steps > self.max_holding_steps
                            and abs(self.inventory) > self.emergency_inv_threshold)
        losing_too_much  = inv_pnl_now < -self.max_inventory_loss
        emergency = holding_too_long or losing_too_much

        gamma_eff = self.gamma * self.emergency_gamma_mult if emergency else self.gamma

        # ── reservation price ──────────────────────────────────────────
        r = mid - gamma_eff * self.inventory * tau

        # ── spread: base → toxicity-adjusted → vol-adjusted ───────────
        delta_base = self.delta_0 * tau + self.delta_min
        delta_tox  = delta_base * (1.0 + self.toxicity_mult * toxicity)
        delta_final = delta_tox * (self.vol_spread_mult if high_vol else 1.0)

        bid = r - delta_final
        ask = r + delta_final

        # ── fill probabilities ─────────────────────────────────────────
        p_bid = self._fill_prob(max(mid - bid, 0.0))
        p_ask = self._fill_prob(max(ask - mid, 0.0))

        # ── hard inventory limits (Phase 4 #1) ────────────────────────
        bid_allowed = self.inventory < self.max_inventory
        ask_allowed = self.inventory > -self.max_inventory

        bid_fill = (self._rng.random() < p_bid) and bid_allowed
        ask_fill = (self._rng.random() < p_ask) and ask_allowed

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

        n_fills = int(bid_fill) + int(ask_fill)
        self._approx_spread_pnl += delta_final * self.lot_size * n_fills

        self._prev_mid = mid
        mtm_pnl = self.cash + self.inventory * mid - self._initial_cash

        return {
            "mid_price": mid,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "delta_base": delta_base,
            "delta_tox": delta_tox,
            "delta_final": delta_final,
            "toxicity": toxicity,
            "realised_vol": realised_vol,
            "high_vol": int(high_vol),
            "emergency": int(emergency),
            "inventory": self.inventory,
            "cash": self.cash,
            "spread_pnl": self._cumulative_spread_pnl,
            "spread_pnl_approx": self._approx_spread_pnl,
            "inventory_pnl": self._cumulative_inventory_pnl,
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
        print("  Risk Control MM — Results")
        print("=" * 42)
        print(f"  Timesteps simulated : {len(results):,}")
        print(f"  Fills (buy / sell)  : {buys:,} / {sells:,}  ({buys+sells:,} total)")
        print(f"  Avg toxicity        : {results['toxicity'].mean():.3f}")
        print(f"  Time high-vol       : {results['high_vol'].mean():.1%}")
        print(f"  Emergency steps     : {results['emergency'].sum():,}  "
              f"(holding>{self.max_holding_steps}min OR inv_loss>-${self.max_inventory_loss:.0f})")
        print(f"  Final inventory     : {final['inventory']:+.4f} BTC")
        print(f"  Max |inventory|     : {results['inventory'].abs().max():.4f} BTC")
        print(f"  Inventory std dev   : {results['inventory'].std():.4f} BTC")
        print(f"  Spread P&L          : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L       : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total (MtM) P&L     : ${final['mtm_pnl']:+,.2f}")
        print(f"  Peak P&L            : ${results['mtm_pnl'].max():+,.2f}")
        print(f"  Max drawdown        : ${results['mtm_pnl'].min():+,.2f}")
        print("=" * 42)
