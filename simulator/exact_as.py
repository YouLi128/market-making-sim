"""
Exact Avellaneda-Stoikov (2008) market making model.

Uses the original paper's formulas without simplification:

    r(t, q) = mid - γ · σ² · q · (T - t)

    δ*(t)   = (γ · σ² · (T - t)) / 2  +  (1/γ) · ln(1 + γ/κ)

    bid = r - δ*
    ask = r + δ*

Fill model (consistent with the paper's Poisson arrival assumption):
    P(fill | dist) = exp(-κ · dist)
    κ here is the order-flow decay per dollar of distance from mid.

Parameter calibration (see calibrate_as_params below):
    Given a target half-spread and a target inventory skew, we can solve
    analytically for (γ, κ) that satisfy both constraints simultaneously.
    This removes the need to hand-tune parameters.

Reference: Avellaneda & Stoikov (2008), Quantitative Finance 8(3), 217-224.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
def calibrate_as_params(
    target_spread: float = 50.0,
    target_skew: float = 25.0,
    sigma_abs: float = 55.0,
    T: int = 1440,
) -> tuple[float, float]:
    """
    Solve for (gamma, kappa) given two intuitive targets.

    Args:
        target_spread : desired half-spread at session start with zero inventory.
        target_skew   : desired reservation price shift when q=1 BTC and half
                        the session remains (tau = T/2).
        sigma_abs     : per-step absolute price vol in $/√step.
        T             : total session steps.

    Returns:
        (gamma, kappa) in consistent units for the AS formulas.

    Derivation:
        From the skew equation at q=1, tau=T/2:
            target_skew = gamma * sigma_abs^2 * 1.0 * (T/2)
            → gamma = target_skew / (sigma_abs^2 * T/2)

        The optimal spread at tau=T has two terms:
            term1 = gamma * sigma_abs^2 * T / 2  = target_skew  (same expression!)
            term2 = (1/gamma) * ln(1 + gamma/kappa)
                  = target_spread - target_skew
            → kappa = gamma / (exp(term2 * gamma) - 1)
    """
    gamma = target_skew / (sigma_abs ** 2 * T / 2)
    term2 = target_spread - target_skew
    kappa = gamma / (np.exp(term2 * gamma) - 1.0)
    return float(gamma), float(kappa)


# ---------------------------------------------------------------------------
@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str
    price: float
    qty: float


class ExactAvellanedaStoikov:
    """
    Exact AS model with analytically calibrated parameters.

    Key differences from the simplified Phase 2 model:
    - spread formula uses the full paper equation (two terms)
    - gamma is in raw units (1/($² · BTC · step)), not absorbed into $/BTC
    - kappa controls both the fill probability decay AND the spread formula
    - fill probability = exp(-kappa * dist), A=1 (certain fill at mid)
    """

    def __init__(
        self,
        target_spread: float = 50.0,
        target_skew: float = 25.0,
        sigma_abs: float = 55.0,
        T: int = 1440,
        lot_size: float = 0.01,
        initial_cash: float = 100_000.0,
        seed: int = 0,
    ):
        self.sigma_abs = sigma_abs
        self.T = T
        self.lot_size = lot_size

        self.gamma, self.kappa = calibrate_as_params(
            target_spread, target_skew, sigma_abs, T
        )

        self.cash = initial_cash
        self._initial_cash = initial_cash
        self.inventory = 0.0
        self._step = 0
        self._cumulative_spread_pnl = 0.0
        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _tau(self) -> int:
        return max(self.T - self._step, 1)

    def _reservation_price(self, mid: float, tau: int) -> float:
        return mid - self.gamma * self.sigma_abs ** 2 * self.inventory * tau

    def _half_spread(self, tau: int) -> float:
        term1 = (self.gamma * self.sigma_abs ** 2 * tau) / 2.0
        term2 = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return max(term1 + term2, 1.0)

    def _fill_prob(self, dist: float) -> float:
        return float(np.exp(-self.kappa * dist))

    # ------------------------------------------------------------------
    def step(self, timestamp: pd.Timestamp, mid: float) -> dict:
        tau = self._tau()
        self._step += 1

        r     = self._reservation_price(mid, tau)
        delta = self._half_spread(tau)
        bid   = r - delta
        ask   = r + delta

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
        self._cumulative_spread_pnl += delta * self.lot_size * n_fills

        mtm_pnl       = self.cash + self.inventory * mid - self._initial_cash
        inventory_pnl = mtm_pnl - self._cumulative_spread_pnl

        return {
            "mid_price":         mid,
            "reservation_price": r,
            "bid":               bid,
            "ask":               ask,
            "half_spread":       delta,
            "inventory":         self.inventory,
            "cash":              self.cash,
            "spread_pnl":        self._cumulative_spread_pnl,
            "inventory_pnl":     inventory_pnl,
            "mtm_pnl":           mtm_pnl,
        }

    def run(self, prices: pd.Series) -> pd.DataFrame:
        records = [self.step(ts, p) for ts, p in prices.items()]
        return pd.DataFrame(records, index=prices.index)

    def summary(self, results: pd.DataFrame) -> None:
        buys  = sum(1 for t in self.trades if t.side == "buy")
        sells = sum(1 for t in self.trades if t.side == "sell")
        final = results.iloc[-1]
        print("=" * 48)
        print(f"  Exact AS  γ={self.gamma:.2e}  κ={self.kappa:.4f}")
        print("=" * 48)
        print(f"  Fills (buy / sell)  : {buys:,} / {sells:,}")
        print(f"  Final inventory     : {final['inventory']:+.4f} BTC")
        print(f"  Max |inventory|     : {results['inventory'].abs().max():.4f} BTC")
        print(f"  Inventory std dev   : {results['inventory'].std():.4f} BTC")
        print(f"  Spread P&L          : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L       : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total MtM P&L       : ${final['mtm_pnl']:+,.2f}")
        print("=" * 48)
