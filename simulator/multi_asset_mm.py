"""
Multi-Asset Market Making — BTC + ETH with cross-asset inventory hedging.

Extension of the Avellaneda-Stoikov model to two correlated assets.

The reservation price for each asset accounts for the portfolio-level inventory
risk, including the cross-asset covariance term:

    r_BTC = mid_BTC - γ · τ · (q_BTC · σ_BTC² + ρ · q_ETH · σ_BTC · σ_ETH)
    r_ETH = mid_ETH - γ · τ · (q_ETH · σ_ETH² + ρ · q_BTC · σ_BTC · σ_ETH)

where σ_i = per-step absolute price vol ($/√step), γ = portfolio risk aversion.

Key intuition:
  • BTC long + ETH long  (ρ>0): positions amplify each other's risk → both
    reservation prices shift MORE aggressively toward mid → faster joint unwind.
  • BTC long + ETH short (ρ>0): positions hedge each other → reservation prices
    shift LESS aggressively → model holds both positions more patiently.

In contrast, two independent single-asset AS models ignore the cross-term and
treat each position in isolation. This leads to over-trading and higher portfolio
variance when the two positions happen to hedge each other.

P&L uses exact per-fill attribution (same convention as Phase 11):
    buy  fill: spread_gain = (mid − bid) × lot
    sell fill: spread_gain = (ask − mid) × lot
    inventory P&L: Σ inventory × Δmid  (per asset, accumulated each step)

Reference: Avellaneda & Stoikov (2008) §5 — multi-asset extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Trade:
    timestamp: pd.Timestamp
    asset: str      # 'BTC' | 'ETH'
    side: str       # 'buy' | 'sell'
    price: float
    qty: float


class MultiAssetMM:
    """
    Simultaneous BTC + ETH market maker with joint reservation prices.

    Parameters
    ----------
    gamma : float
        Portfolio risk aversion γ, units 1/($ · step).
        Calibrated so that at q_BTC = 0.10 BTC and τ = 1, r_BTC shifts by ~$5
        (same relative behavior as Phase 2 with gamma_P2=50 at 0.10 BTC).
        Default 0.0165 ≈ 50 / (55²) where 55 $/√step is σ_BTC_abs.
    corr : float
        BTC-ETH correlation ρ ∈ (-1, 1). Typical BTC-ETH: ~0.80–0.90.
    sigma_btc_ann, sigma_eth_ann : float
        Annualised vols used to compute per-step σ_abs. Fixed at S0 values.
    delta_0_btc, delta_min_btc : float
        Half-spread range for BTC (same as Phase 2: 45, 5 → [$5, $50]).
    delta_0_eth, delta_min_eth : float
        Half-spread range for ETH, scaled to ~0.09% of price:
        default 2.7 and 0.3 → [$0.30, $3.00] for a $3 000 ETH price.
    fill_k_btc, fill_k_eth : float
        Exponential fill-probability decay constants.  Calibrated so that
        fill prob ≈ 0.30 at the initial half-spread distance from mid.
    lot_size_btc, lot_size_eth : float
        Trade size per fill. Default 0.01 BTC (~$500), 0.10 ETH (~$300).
    """

    def __init__(
        self,
        # Portfolio risk
        gamma: float = 0.0165,
        corr: float = 0.85,
        sigma_btc_ann: float = 0.80,
        sigma_eth_ann: float = 1.20,
        S0_btc: float = 50_000.0,
        S0_eth: float = 3_000.0,
        # Spread parameters — BTC
        delta_0_btc: float = 45.0,
        delta_min_btc: float = 5.0,
        fill_k_btc: float = 0.024,
        lot_size_btc: float = 0.01,
        # Spread parameters — ETH
        delta_0_eth: float = 2.7,
        delta_min_eth: float = 0.3,
        fill_k_eth: float = 0.446,
        lot_size_eth: float = 0.10,
        # Session
        T: int = 1440,
        initial_cash: float = 100_000.0,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.corr = corr
        self.T = T

        # Per-step absolute vol ($/√step) — fixed at S0 for simplicity
        dt_year = 1.0 / (365.0 * 24.0 * 60.0)
        self.sigma_btc_abs = S0_btc * sigma_btc_ann * np.sqrt(dt_year)
        self.sigma_eth_abs = S0_eth * sigma_eth_ann * np.sqrt(dt_year)

        # BTC spread params
        self.delta_0_btc  = delta_0_btc
        self.delta_min_btc = delta_min_btc
        self.fill_k_btc   = fill_k_btc
        self.lot_size_btc = lot_size_btc

        # ETH spread params
        self.delta_0_eth  = delta_0_eth
        self.delta_min_eth = delta_min_eth
        self.fill_k_eth   = fill_k_eth
        self.lot_size_eth = lot_size_eth

        # State
        self.cash: float = initial_cash
        self._initial_cash: float = initial_cash
        self.inv_btc: float = 0.0
        self.inv_eth: float = 0.0
        self._step: int = 0

        # Exact P&L tracking
        self._spread_pnl_btc: float = 0.0
        self._spread_pnl_eth: float = 0.0
        self._inv_pnl_btc: float = 0.0
        self._inv_pnl_eth: float = 0.0
        self._prev_mid_btc: float = 0.0
        self._prev_mid_eth: float = 0.0

        self.trades: List[Trade] = []
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _tau(self) -> float:
        return max(self.T - self._step, 1) / self.T

    def _half_spread(self, tau: float, delta_0: float, delta_min: float) -> float:
        return delta_0 * tau + delta_min

    def _fill_prob(self, dist: float, fill_k: float) -> float:
        return float(np.exp(-fill_k * dist))

    # ------------------------------------------------------------------
    def step(
        self,
        timestamp: pd.Timestamp,
        mid_btc: float,
        mid_eth: float,
    ) -> dict:
        # --- exact inventory MTM (before fills) ----------------------
        self._inv_pnl_btc += self.inv_btc * (mid_btc - self._prev_mid_btc)
        self._inv_pnl_eth += self.inv_eth * (mid_eth - self._prev_mid_eth)

        tau = self._tau()
        self._step += 1

        sb = self.sigma_btc_abs
        se = self.sigma_eth_abs

        # --- joint reservation prices --------------------------------
        r_btc = mid_btc - self.gamma * tau * (
            self.inv_btc * sb ** 2 + self.corr * self.inv_eth * sb * se
        )
        r_eth = mid_eth - self.gamma * tau * (
            self.inv_eth * se ** 2 + self.corr * self.inv_btc * sb * se
        )

        # --- BTC quotes & fills --------------------------------------
        delta_btc = self._half_spread(tau, self.delta_0_btc, self.delta_min_btc)
        bid_btc = r_btc - delta_btc
        ask_btc = r_btc + delta_btc

        p_bid_btc = self._fill_prob(max(mid_btc - bid_btc, 0.0), self.fill_k_btc)
        p_ask_btc = self._fill_prob(max(ask_btc - mid_btc, 0.0), self.fill_k_btc)

        bid_btc_fill = self._rng.random() < p_bid_btc
        ask_btc_fill = self._rng.random() < p_ask_btc

        if bid_btc_fill:
            self.cash -= bid_btc * self.lot_size_btc
            self.inv_btc += self.lot_size_btc
            self._spread_pnl_btc += (mid_btc - bid_btc) * self.lot_size_btc
            self.trades.append(Trade(timestamp, "BTC", "buy", bid_btc, self.lot_size_btc))

        if ask_btc_fill:
            self.cash += ask_btc * self.lot_size_btc
            self.inv_btc -= self.lot_size_btc
            self._spread_pnl_btc += (ask_btc - mid_btc) * self.lot_size_btc
            self.trades.append(Trade(timestamp, "BTC", "sell", ask_btc, self.lot_size_btc))

        # --- ETH quotes & fills --------------------------------------
        delta_eth = self._half_spread(tau, self.delta_0_eth, self.delta_min_eth)
        bid_eth = r_eth - delta_eth
        ask_eth = r_eth + delta_eth

        p_bid_eth = self._fill_prob(max(mid_eth - bid_eth, 0.0), self.fill_k_eth)
        p_ask_eth = self._fill_prob(max(ask_eth - mid_eth, 0.0), self.fill_k_eth)

        bid_eth_fill = self._rng.random() < p_bid_eth
        ask_eth_fill = self._rng.random() < p_ask_eth

        if bid_eth_fill:
            self.cash -= bid_eth * self.lot_size_eth
            self.inv_eth += self.lot_size_eth
            self._spread_pnl_eth += (mid_eth - bid_eth) * self.lot_size_eth
            self.trades.append(Trade(timestamp, "ETH", "buy", bid_eth, self.lot_size_eth))

        if ask_eth_fill:
            self.cash += ask_eth * self.lot_size_eth
            self.inv_eth -= self.lot_size_eth
            self._spread_pnl_eth += (ask_eth - mid_eth) * self.lot_size_eth
            self.trades.append(Trade(timestamp, "ETH", "sell", ask_eth, self.lot_size_eth))

        self._prev_mid_btc = mid_btc
        self._prev_mid_eth = mid_eth

        # --- portfolio metrics ---------------------------------------
        mtm_pnl = (
            self.cash
            + self.inv_btc * mid_btc
            + self.inv_eth * mid_eth
            - self._initial_cash
        )
        spread_pnl = self._spread_pnl_btc + self._spread_pnl_eth
        inv_pnl    = self._inv_pnl_btc    + self._inv_pnl_eth

        # Portfolio variance: Var = (q_b·σ_b)² + (q_e·σ_e)² + 2ρ·q_b·σ_b·q_e·σ_e
        port_var = (
            (self.inv_btc * sb) ** 2
            + (self.inv_eth * se) ** 2
            + 2 * self.corr * self.inv_btc * sb * self.inv_eth * se
        )

        return {
            # BTC
            "mid_btc": mid_btc, "r_btc": r_btc,
            "bid_btc": bid_btc, "ask_btc": ask_btc,
            "inv_btc": self.inv_btc,
            "spread_pnl_btc": self._spread_pnl_btc,
            "inv_pnl_btc": self._inv_pnl_btc,
            # ETH
            "mid_eth": mid_eth, "r_eth": r_eth,
            "bid_eth": bid_eth, "ask_eth": ask_eth,
            "inv_eth": self.inv_eth,
            "spread_pnl_eth": self._spread_pnl_eth,
            "inv_pnl_eth": self._inv_pnl_eth,
            # Portfolio
            "spread_pnl": spread_pnl,
            "inventory_pnl": inv_pnl,
            "mtm_pnl": mtm_pnl,
            "port_var": port_var,
        }

    # ------------------------------------------------------------------
    def run(
        self,
        btc_prices: pd.Series,
        eth_prices: pd.Series,
        corr_series: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Run the model. If corr_series is provided, ρ is updated at each step
        from the series (time-varying correlation). Otherwise uses self.corr.
        """
        records = []
        for ts, p_btc, p_eth in zip(btc_prices.index, btc_prices, eth_prices):
            if corr_series is not None and ts in corr_series.index:
                self.corr = float(corr_series.loc[ts])
            records.append(self.step(ts, p_btc, p_eth))
        return pd.DataFrame(records, index=btc_prices.index)

    # ------------------------------------------------------------------
    def summary(self, results: pd.DataFrame) -> None:
        btc_buys  = sum(1 for t in self.trades if t.asset == "BTC" and t.side == "buy")
        btc_sells = sum(1 for t in self.trades if t.asset == "BTC" and t.side == "sell")
        eth_buys  = sum(1 for t in self.trades if t.asset == "ETH" and t.side == "buy")
        eth_sells = sum(1 for t in self.trades if t.asset == "ETH" and t.side == "sell")
        final = results.iloc[-1]

        print("=" * 52)
        print("  Multi-Asset MM (BTC + ETH, ρ={:.2f})".format(self.corr))
        print("=" * 52)
        print(f"  BTC fills (buy/sell)  : {btc_buys:,} / {btc_sells:,}")
        print(f"  ETH fills (buy/sell)  : {eth_buys:,} / {eth_sells:,}")
        print(f"  Final inv BTC         : {final['inv_btc']:+.4f} BTC")
        print(f"  Final inv ETH         : {final['inv_eth']:+.4f} ETH")
        print(f"  BTC inv std           : {results['inv_btc'].std():.4f} BTC")
        print(f"  ETH inv std           : {results['inv_eth'].std():.4f} ETH")
        print(f"  Spread P&L (BTC)      : ${final['spread_pnl_btc']:+,.2f}")
        print(f"  Spread P&L (ETH)      : ${final['spread_pnl_eth']:+,.2f}")
        print(f"  Spread P&L (total)    : ${final['spread_pnl']:+,.2f}")
        print(f"  Inventory P&L (total) : ${final['inventory_pnl']:+,.2f}")
        print(f"  Total MtM P&L         : ${final['mtm_pnl']:+,.2f}")
        print(f"  Peak P&L              : ${results['mtm_pnl'].max():+,.2f}")
        print(f"  Max drawdown          : ${results['mtm_pnl'].min():+,.2f}")
        inv_corr = results["inv_btc"].corr(results["inv_eth"])
        print(f"  BTC-ETH inv corr      : {inv_corr:+.3f}  "
              f"({'hedged ✓' if inv_corr < -0.1 else 'not hedged'})")
        print("=" * 52)
