"""
VPIN — Volume-synchronized Probability of Informed Trading.

Reference: Easley, Lopez de Prado & O'Hara (2012),
           "Flow Toxicity and Liquidity in a High Frequency World",
           Review of Financial Studies 25(5).

How it works:
1. Classify each bar's volume as buy or sell using Bulk Volume Classification (BVC):
       Z = (close - open) / σ         (normalised price change in the bar)
       buy_vol  = volume × Φ(Z)        Φ = standard normal CDF
       sell_vol = volume × (1 - Φ(Z))

2. Accumulate into fixed-size volume buckets (bucket_size = total_vol / n_buckets).
   A bucket closes when its cumulated volume reaches bucket_size.

3. For each completed bucket i:
       OI_i = |buy_vol_i - sell_vol_i| / bucket_size   ∈ [0, 1]

4. VPIN_t = rolling mean of the last n_window OI values.
   Near 0 → balanced two-sided flow (safe to quote normally).
   Near 1 → heavily one-sided flow (informed traders likely active).

Why better than the simple rolling up-fraction:
- Anchored to volume, not just price direction.
- Responds to large trades (high volume), not just tick counts.
- Standard in academic literature and used by practitioners.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_vpin(
    ohlcv: pd.DataFrame,
    n_buckets: int = 50,
    n_window: int = 50,
    sigma_window: int = 30,
) -> pd.Series:
    """
    Compute VPIN from a 1-minute OHLCV DataFrame.

    Args:
        ohlcv        : DataFrame with columns open, high, low, close, volume.
        n_buckets    : total number of buckets to divide the session volume into.
                       bucket_size = total_volume / n_buckets.
        n_window     : number of completed buckets in the rolling VPIN average.
        sigma_window : lookback for rolling σ used in BVC classification.

    Returns:
        pd.Series of VPIN values aligned to ohlcv.index, range [0, 1].
    """
    closes = ohlcv["close"].values
    opens  = ohlcv["open"].values
    vols   = ohlcv["volume"].values

    # Rolling σ of price changes (absolute, not pct) for BVC
    price_changes = closes - opens
    rolling_std   = (
        pd.Series(price_changes)
        .rolling(window=sigma_window, min_periods=3)
        .std()
        .fillna(pd.Series(price_changes).std())
        .values
    )
    rolling_std = np.where(rolling_std < 1e-8, 1e-8, rolling_std)

    # BVC: classify each bar
    z        = price_changes / rolling_std
    buy_frac = norm.cdf(z)
    buy_vols  = vols * buy_frac
    sell_vols = vols * (1.0 - buy_frac)

    # Bucket size
    total_vol   = vols.sum()
    bucket_size = max(total_vol / n_buckets, 1e-8)

    # Accumulate into buckets
    vpin_at = {}
    oi_history: list[float] = []

    b_buy = b_sell = b_vol = 0.0

    for i in range(len(ohlcv)):
        rem_buy  = buy_vols[i]
        rem_sell = sell_vols[i]
        rem_tot  = rem_buy + rem_sell

        while rem_tot > 1e-10:
            space    = bucket_size - b_vol
            fill     = min(rem_tot, space)
            frac     = fill / rem_tot

            b_buy  += rem_buy  * frac
            b_sell += rem_sell * frac
            b_vol  += fill

            rem_buy  *= (1.0 - frac)
            rem_sell *= (1.0 - frac)
            rem_tot   = rem_buy + rem_sell

            if b_vol >= bucket_size - 1e-10:
                oi = abs(b_buy - b_sell) / bucket_size
                oi_history.append(min(oi, 1.0))
                b_buy = b_sell = b_vol = 0.0

        ts = ohlcv.index[i]
        if oi_history:
            vpin_at[ts] = float(np.mean(oi_history[-n_window:]))
        else:
            vpin_at[ts] = 0.0

    return pd.Series(vpin_at, name="vpin")
