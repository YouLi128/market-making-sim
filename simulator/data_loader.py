"""
Real BTC/USDT 1-minute data from Binance public API.
No API key required.

Usage:
    prices = fetch_btc_price()              # latest 1440 bars
    prices = fetch_btc_price(date="2024-06-01")  # specific date
"""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")

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


def fetch_historical_days(
    n_days: int = 90,
    end_date: str = None,
    symbol: str = "BTCUSDT",
    min_bars: int = 1200,
) -> List[pd.Series]:
    """
    Fetch n_days of 1-minute close prices, one pd.Series per calendar day.
    Results are cached locally in data_cache/ so subsequent runs are instant.

    Args:
        n_days   : number of calendar days to fetch (default 90 = ~3 months)
        end_date : "YYYY-MM-DD" last day to include, default = yesterday
        min_bars : skip days with fewer bars (e.g. exchange downtime)

    Returns:
        List of pd.Series, each being one day of 1-minute prices.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if end_date is None:
        end_dt = datetime.now(timezone.utc).date() - timedelta(days=1)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()

    days: List[pd.Series] = []

    for i in range(n_days, 0, -1):
        date = end_dt - timedelta(days=i - 1)
        date_str = date.strftime("%Y-%m-%d")
        cache_path = os.path.join(CACHE_DIR, f"{symbol}_{date_str}.csv")

        # Load from cache if available
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            prices = df["mid_price"]
        else:
            try:
                prices = fetch_btc_price(date=date_str, n_steps=1440, symbol=symbol)
                if len(prices) >= min_bars:
                    prices.to_frame().to_csv(cache_path)
                time.sleep(0.1)   # gentle rate limiting
            except Exception as e:
                print(f"  [skip] {date_str}: {e}")
                continue

        if len(prices) >= min_bars:
            days.append(prices)

    return days


def fetch_btc_ohlcv(
    date: str = None,
    n_steps: int = 1440,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """
    Same as fetch_btc_price but returns full OHLCV DataFrame.
    Columns: open, high, low, close, volume
    """
    if date is not None:
        start_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(start_dt.timestamp() * 1000)
    else:
        start_ms = int((datetime.now(timezone.utc) - timedelta(minutes=n_steps)).timestamp() * 1000)

    bars = []
    current_start = start_ms

    while len(bars) < n_steps:
        remaining = n_steps - len(bars)
        limit = min(remaining, MAX_PER_REQUEST)
        resp = requests.get(
            BINANCE_URL,
            params={"symbol": symbol, "interval": "1m",
                    "startTime": current_start, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        bars.extend(data)
        current_start = data[-1][6] + 1
        if len(data) < limit:
            break

    timestamps = [pd.Timestamp(b[0], unit="ms", tz="UTC") for b in bars[:n_steps]]
    return pd.DataFrame(
        {
            "open":   [float(b[1]) for b in bars[:n_steps]],
            "high":   [float(b[2]) for b in bars[:n_steps]],
            "low":    [float(b[3]) for b in bars[:n_steps]],
            "close":  [float(b[4]) for b in bars[:n_steps]],
            "volume": [float(b[5]) for b in bars[:n_steps]],
        },
        index=timestamps,
    )
