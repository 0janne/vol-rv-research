"""Variance risk premium.

Two distinct objects share the name "VRP" and conflating them is the single
easiest way to produce a beautiful, meaningless backtest:

  ex-post VRP   implied variance at t minus variance actually realised over
                (t, t+H].  This is the *payoff* of a short variance position
                opened at t.  It is not observable at t.

  VRP proxy     implied variance at t minus variance realised over (t-H, t].
                Observable at t, and therefore the only one of the two that
                may inform a trading decision.

Everything in this module that feeds a position is built from the proxy.
"""

from __future__ import annotations

import pandas as pd

from ..config import HORIZON_D, SIGNAL_LAG_D
from ..vol.realized import realized_vol, realized_vol_forward


def vrp_ex_post(implied: pd.Series, ohlc: pd.DataFrame, horizon: int = HORIZON_D,
                estimator: str = "close_to_close") -> pd.Series:
    """Implied variance minus subsequently realised variance. ANALYSIS ONLY."""
    rv_fwd = realized_vol_forward(ohlc, horizon, estimator).reindex(implied.index)
    return (implied**2 - rv_fwd**2).rename("vrp_ex_post")


def vrp_proxy(implied: pd.Series, ohlc: pd.DataFrame, horizon: int = HORIZON_D,
              estimator: str = "yang_zhang") -> pd.Series:
    """Implied minus trailing realised, in vol points. Known at t."""
    rv_trail = realized_vol(ohlc, horizon, estimator).reindex(implied.index)
    return (implied - rv_trail).rename("vrp_proxy")


def _zscore(x: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    mu = x.rolling(lookback, min_periods=min_periods).mean()
    sd = x.rolling(lookback, min_periods=min_periods).std(ddof=1)
    return (x - mu) / sd.replace(0.0, pd.NA)


def vrp_signal(
    implied: pd.Series,
    ohlc: pd.DataFrame,
    horizon: int = HORIZON_D,
    lookback: int = 252,
    entry_z: float = 0.0,
    cap: float = 2.0,
    estimator: str = "yang_zhang",
) -> pd.Series:
    """Target position in the variance leg. Negative = short variance.

    Rich premium (proxy well above its own recent history) implies a larger
    short. Position is capped and scaled inversely to the implied level, so the
    trade is roughly vega-targeted rather than notional-targeted -- without
    that, the strategy silently takes its largest risk precisely when vol is
    highest.
    """
    proxy = vrp_proxy(implied, ohlc, horizon, estimator)
    z = _zscore(proxy, lookback, min_periods=lookback // 2)
    raw = -(z - entry_z).clip(lower=0.0)          # short only when premium is rich
    raw = raw.clip(lower=-cap)

    # Expanding, not full-sample: `implied.median()` over the whole series would
    # let the sizer know the average volatility of the next thirty years.
    # tests/test_no_lookahead.py caught exactly that.
    ref = implied.expanding(min_periods=lookback // 2).median()
    vega_scale = (ref / implied).clip(upper=2.0)
    return (raw * vega_scale).shift(SIGNAL_LAG_D).rename("vrp_position")
