"""Realized-volatility estimators.

All return annualised volatility in *percentage points*, matching the units the
VIX is quoted in, so implied and realized are directly comparable.

Range-based estimators are more efficient than close-to-close (Parkinson is
roughly 5x for a driftless diffusion) but they are biased low in the presence of
overnight gaps, which is exactly when equity-index volatility matters. Yang-Zhang
is included because it is the one of these that handles both drift and the
overnight jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS

ANN = np.sqrt(TRADING_DAYS) * 100.0


def _logs(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, low, c = df["open"], df["high"], df["low"], df["close"]
    return {
        "hl": np.log(h / low),
        "co": np.log(c / o),
        "ho": np.log(h / o),
        "lo": np.log(low / o),
        "hc": np.log(h / c),
        "lc": np.log(low / c),
        "oc_prev": np.log(o / c.shift(1)),
        "cc": np.log(c / c.shift(1)),
    }


def close_to_close(df: pd.DataFrame, window: int) -> pd.Series:
    return _logs(df)["cc"].rolling(window).std(ddof=1) * ANN


def parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    hl2 = _logs(df)["hl"] ** 2
    var = hl2.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var) * ANN


def garman_klass(df: pd.DataFrame, window: int) -> pd.Series:
    lg = _logs(df)
    daily = 0.5 * lg["hl"] ** 2 - (2.0 * np.log(2.0) - 1.0) * lg["co"] ** 2
    var = daily.rolling(window).mean()
    return np.sqrt(var.clip(lower=0)) * ANN


def rogers_satchell(df: pd.DataFrame, window: int) -> pd.Series:
    lg = _logs(df)
    daily = lg["ho"] * (lg["ho"] - lg["co"]) + lg["lo"] * (lg["lo"] - lg["co"])
    var = daily.rolling(window).mean()
    return np.sqrt(var.clip(lower=0)) * ANN


def yang_zhang(df: pd.DataFrame, window: int) -> pd.Series:
    lg = _logs(df)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    v_over = lg["oc_prev"].rolling(n).var(ddof=1)
    v_open = lg["co"].rolling(n).var(ddof=1)
    rs = lg["ho"] * (lg["ho"] - lg["co"]) + lg["lo"] * (lg["lo"] - lg["co"])
    v_rs = rs.rolling(n).mean()
    var = v_over + k * v_open + (1.0 - k) * v_rs
    return np.sqrt(var.clip(lower=0)) * ANN


ESTIMATORS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
    "yang_zhang": yang_zhang,
}


def realized_vol(df: pd.DataFrame, window: int, estimator: str = "close_to_close") -> pd.Series:
    """Trailing realized vol over the past `window` days. Known at time t."""
    if estimator not in ESTIMATORS:
        raise KeyError(f"unknown estimator {estimator!r}; have {sorted(ESTIMATORS)}")
    return ESTIMATORS[estimator](df, window)


def realized_vol_forward(
    df: pd.DataFrame, window: int, estimator: str = "close_to_close"
) -> pd.Series:
    """Realized vol over the *next* `window` days.

    This is the realised leg of a variance swap struck at t. It is not knowable
    at t, so it may appear in the PnL of a position opened at t but must never
    appear in a signal computed at t. `signals/` never calls this function.
    """
    return realized_vol(df, window, estimator).shift(-window)
