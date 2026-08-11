"""The test that matters.

A signal computed at time t must depend only on data up to t. The way to check
that is not to read the code but to corrupt the future and confirm nothing in
the past moves.
"""

import numpy as np
import pandas as pd

from volrv.signals import calendar_signal, vrp_proxy, vrp_signal


def _corrupt_tail(df: pd.DataFrame | pd.Series, cut: int):
    out = df.copy()
    rng = np.random.default_rng(0)
    if isinstance(out, pd.Series):
        out.iloc[cut:] = out.iloc[cut:] * rng.uniform(3.0, 6.0, len(out) - cut)
    else:
        out.iloc[cut:] = out.iloc[cut:] * rng.uniform(3.0, 6.0, (len(out) - cut, out.shape[1]))
    return out


def test_vrp_proxy_ignores_the_future(implied, ohlc):
    cut = 1200
    base = vrp_proxy(implied, ohlc)
    shocked = vrp_proxy(_corrupt_tail(implied, cut), _corrupt_tail(ohlc, cut))
    pd.testing.assert_series_equal(base.iloc[:cut], shocked.iloc[:cut])


def test_vrp_signal_ignores_the_future(implied, ohlc):
    cut = 1200
    base = vrp_signal(implied, ohlc)
    shocked = vrp_signal(_corrupt_tail(implied, cut), _corrupt_tail(ohlc, cut))
    pd.testing.assert_series_equal(base.iloc[:cut], shocked.iloc[:cut])


def test_calendar_signal_ignores_the_future(wide):
    cut = 1200
    base = calendar_signal(wide)
    shocked = calendar_signal(_corrupt_tail(wide, cut))
    pd.testing.assert_series_equal(base.iloc[:cut], shocked.iloc[:cut])


def test_signals_are_lagged(implied, ohlc):
    """Position on day t must not use day t's own close."""
    sig = vrp_signal(implied, ohlc)
    raw = vrp_proxy(implied, ohlc)
    first_sig = sig.first_valid_index()
    first_raw = raw.first_valid_index()
    assert first_sig > first_raw
