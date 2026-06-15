"""
GARCH(1,1) conditional volatility estimator for market making.

Fits GARCH(1,1) on a price series and returns the per-step conditional
annualised sigma, which can be plugged into the AS reservation price and
spread formulas to make them respond to volatility clustering.

Why GARCH beats fixed sigma:
  GBM assumes constant volatility. In reality (especially BTC), large moves
  cluster — a big drop is followed by more large moves. GARCH captures this:

      σ²(t) = ω + α · ε²(t-1) + β · σ²(t-1)

  α weights the latest shock; β weights the previous forecast.
  Typical BTC values: α ≈ 0.10, β ≈ 0.85 → slow decay, long memory.

Usage:
    sigma_series = compute_garch_sigma(prices)
    # sigma_series is annualised vol, same units as the 0.80 baseline
"""

import warnings

import numpy as np
import pandas as pd
from arch import arch_model


def compute_garch_sigma(
    prices: pd.Series,
    baseline_sigma: float = 0.80,
) -> pd.Series:
    """
    Fit GARCH(1,1) on the full price series and return the conditional
    annualised volatility at each step.

    Args:
        prices         : pd.Series of prices (1-minute bars).
        baseline_sigma : fallback annualised vol for the warm-up period
                         before GARCH has enough data.

    Returns:
        pd.Series of annualised sigma, aligned to prices.index.
        First bar is set to baseline_sigma (no prior data).
    """
    log_returns = np.diff(np.log(prices.values)) * 100   # in % for numerical stability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am  = arch_model(log_returns, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = am.fit(disp="off", show_warning=False)

    # conditional_volatility is per-bar sigma in %
    cond_vol_pct = res.conditional_volatility          # length = n-1

    # Convert % per bar → annualised fraction
    dt_year       = 1.0 / (365.0 * 24.0 * 60.0)       # 1 minute in years
    sigma_per_bar = cond_vol_pct / 100.0               # fraction per bar
    sigma_annual  = sigma_per_bar / np.sqrt(dt_year)   # annualised

    # Prepend baseline_sigma for the first bar (no return yet)
    sigma_values = np.concatenate([[baseline_sigma], sigma_annual])

    return pd.Series(sigma_values, index=prices.index, name="garch_sigma")
