import numpy as np
import pandas as pd


def generate_btc_price(
    S0: float = 50_000.0,
    mu: float = 0.0,
    sigma: float = 0.80,
    n_steps: int = 1440,
    dt_minutes: float = 1.0,
    seed: int = 42,
) -> pd.Series:
    """
    Synthetic BTC price path via Geometric Brownian Motion.

    sigma=0.80 gives ~80% annualised vol, typical for BTC.
    With dt=1 min, per-step std dev ≈ S0 * sigma * sqrt(dt_year)
                                     ≈ 50_000 * 0.80 * 0.00138 ≈ $55.
    """
    np.random.seed(seed)
    dt_year = dt_minutes / (365.0 * 24.0 * 60.0)

    Z = np.random.randn(n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt_year + sigma * np.sqrt(dt_year) * Z
    prices = S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))[:n_steps]

    idx = pd.date_range("2024-01-01", periods=n_steps, freq=f"{int(dt_minutes)}min")
    return pd.Series(prices, index=idx, name="mid_price")


def generate_btc_price_regime(
    S0: float = 50_000.0,
    sigma_normal: float = 0.80,
    sigma_trend: float = 1.20,
    mu_trend: float = 3.0,
    p_enter: float = 0.02,
    p_exit: float = 0.10,
    n_steps: int = 1440,
    dt_minutes: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Regime-switching BTC price with alternating normal and trending periods.

    Normal regime : random walk (mu=0, sigma=sigma_normal)
    Trend regime  : strong drift ±mu_trend, higher vol sigma_trend
                    direction chosen randomly at each regime entry

    p_enter : prob per step of switching normal → trend  (avg normal run = 1/p_enter = 50 steps)
    p_exit  : prob per step of switching trend  → normal (avg trend  run = 1/p_exit  = 10 steps)

    Returns (prices pd.Series, regimes pd.Series) where regimes = 0 (normal) or 1 (trending).
    """
    rng = np.random.default_rng(seed)
    dt_year = dt_minutes / (365.0 * 24.0 * 60.0)

    prices  = np.empty(n_steps)
    regimes = np.zeros(n_steps, dtype=int)
    prices[0] = S0

    regime = 0
    trend_direction = 1.0

    for t in range(1, n_steps):
        if regime == 0:
            if rng.random() < p_enter:
                regime = 1
                trend_direction = 1.0 if rng.random() < 0.5 else -1.0
        else:
            if rng.random() < p_exit:
                regime = 0

        regimes[t] = regime

        if regime == 0:
            mu    = 0.0
            sigma = sigma_normal
        else:
            mu    = trend_direction * mu_trend
            sigma = sigma_trend

        Z = rng.standard_normal()
        log_ret = (mu - 0.5 * sigma ** 2) * dt_year + sigma * np.sqrt(dt_year) * Z
        prices[t] = prices[t - 1] * np.exp(log_ret)

    idx = pd.date_range("2024-01-01", periods=n_steps, freq=f"{int(dt_minutes)}min")
    return (
        pd.Series(prices,  index=idx, name="mid_price"),
        pd.Series(regimes, index=idx, name="regime"),
    )


def generate_correlated_btc_eth(
    S0_btc: float = 50_000.0,
    S0_eth: float = 3_000.0,
    sigma_btc: float = 0.80,
    sigma_eth: float = 1.20,
    corr: float = 0.85,
    mu_btc: float = 0.0,
    mu_eth: float = 0.0,
    n_steps: int = 1440,
    dt_minutes: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Correlated GBM price paths for BTC and ETH.

    Uses Cholesky decomposition:
        W_BTC = Z1
        W_ETH = corr * Z1 + sqrt(1 - corr²) * Z2
    so Cov(W_BTC, W_ETH) = corr * dt as required.

    Returns (btc_prices, eth_prices) as pd.Series on a shared index.
    """
    rng = np.random.default_rng(seed)
    dt_year = dt_minutes / (365.0 * 24.0 * 60.0)

    Z1 = rng.standard_normal(n_steps)
    Z2 = rng.standard_normal(n_steps)
    W_btc = Z1
    W_eth = corr * Z1 + np.sqrt(1.0 - corr ** 2) * Z2

    lr_btc = (mu_btc - 0.5 * sigma_btc ** 2) * dt_year + sigma_btc * np.sqrt(dt_year) * W_btc
    lr_eth = (mu_eth - 0.5 * sigma_eth ** 2) * dt_year + sigma_eth * np.sqrt(dt_year) * W_eth

    idx = pd.date_range("2024-01-01", periods=n_steps, freq=f"{int(dt_minutes)}min")
    btc = pd.Series(S0_btc * np.exp(np.cumsum(lr_btc)), index=idx, name="btc_mid")
    eth = pd.Series(S0_eth * np.exp(np.cumsum(lr_eth)), index=idx, name="eth_mid")
    return btc, eth
