import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlc() -> pd.DataFrame:
    """Synthetic GBM daily bars with known annualised volatility of 20%."""
    rng = np.random.default_rng(42)
    n = 2000
    sigma_d = 0.20 / np.sqrt(252)
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma_d, n)))
    intraday = np.abs(rng.normal(0, sigma_d / 2, n))
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": close * np.exp(rng.normal(0, sigma_d / 4, n)),
            "close": close,
        },
        index=idx,
    )
    df["high"] = df[["open", "close"]].max(axis=1) * np.exp(intraday)
    df["low"] = df[["open", "close"]].min(axis=1) * np.exp(-intraday)
    return df[["open", "high", "low", "close"]]


@pytest.fixture
def implied(ohlc) -> pd.Series:
    """Implied vol sitting a few points above realised, as it usually does."""
    rng = np.random.default_rng(7)
    base = 20.0 + rng.normal(3.0, 1.5, len(ohlc)).clip(-2, 12)
    return pd.Series(base, index=ohlc.index, name="VIX").clip(lower=5.0)


@pytest.fixture
def wide(implied) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    slope = rng.normal(0.15, 0.08, len(implied))
    return pd.DataFrame(
        {
            "VIX9D": implied * 0.95,
            "VIX": implied,
            "VIX3M": implied * (1 + slope),
            "VIX6M": implied * (1 + slope * 1.4),
        }
    )
