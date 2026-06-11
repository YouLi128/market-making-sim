"""
Real BTC/USDT 1-minute data from Binance public API.
No API key required.

Usage:
    prices = fetch_btc_price()              # latest 1440 bars
    prices = fetch_btc_price(date="2024-06-01")  # specific date
"""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_PER_REQUEST = 1000  # Binance hard limit per request


def fetch_btc_price(
    date: str = None,
    n_steps: int = 1440,
    symbol: str = "BTCUSDT",
) -> pd.Series:
    """
    Fetch BTC/USDT 1-minute close prices from Binance.

    Args:
        date    : "YYYY-MM-DD" — fetch that calendar day (UTC).
                  None → fetch the most recent n_steps bars.
        n_steps : number of 1-minute bars to return.
        symbol  : trading pair (default BTCUSDT).

    Returns:
        pd.Series of close prices with DatetimeIndex (UTC).
    """
    if date is not None:
        start_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
    else:
        # work backwards from now
        start_ms = int((datetime.now(timezone.utc) - timedelta(minutes=n_steps)).timestamp() * 1000)

    bars = []
    current_start = start_ms

    while len(bars) < n_steps:
        remaining = n_steps - len(bars)
        limit = min(remaining, MAX_PER_REQUEST)

        resp = requests.get(
            BINANCE_URL,
            params={
                "symbol":    symbol,
                "interval":  "1m",
                "startTime": current_start,
                "limit":     limit,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        bars.extend(data)
        # next batch starts after the last bar's close time
        current_start = data[-1][6] + 1  # close_time + 1ms

        if len(data) < limit:
            break

    closes     = [float(b[4]) for b in bars[:n_steps]]   # index 4 = close price
    timestamps = [pd.Timestamp(b[0], unit="ms", tz="UTC") for b in bars[:n_steps]]

    return pd.Series(closes, index=timestamps, name="mid_price")
