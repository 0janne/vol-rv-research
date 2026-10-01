"""Calendar spreads in VIX futures: the tradable version of a VIX curve trade.

The index-level backtest in `engine.backtest_calendar` marks a spread between
two VIX *indices*. Neither can be bought. The futures curve already prices the
expected mean reversion of the index, so a signal that predicts index moves
need not predict futures returns at all. This module answers whether it does.

Mechanics
---------
On each date we hold a front contract and a back contract `back_offset`
positions further out. The front leg is rolled `roll_days` business days
before it expires, so the book never sits through a final settlement. PnL on
day t is the change in the spread of the contracts that were held at t-1, so a
roll never shows up as a price jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..db import read

ROLL_DAYS = 5
COST_PER_LEG = 0.05  # half the typical front-month bid-ask, in vol points


def futures_panel(engine=None) -> pd.DataFrame:
    """Settlement panel: index = trading date, columns = contract expiry."""
    df = read("SELECT expiry, dt, settle FROM vix_future_daily", engine=engine)
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df.pivot_table(index="dt", columns="expiry", values="settle").sort_index()


def held_contracts(
    panel: pd.DataFrame, roll_days: int = ROLL_DAYS, back_offset: int = 1
) -> pd.DataFrame:
    """Front and back contract held at each date's close.

    Days-to-expiry are counted in business days, not in rows of the panel, so
    the choice does not depend on how much data happens to follow a date.
    Dates without a full set of listed contracts are dropped.
    """
    expiries = np.array(sorted(panel.columns), dtype="datetime64[D]")
    dates = panel.index.values.astype("datetime64[D]")
    rows = []
    for d in dates:
        live = expiries[expiries > d]
        if len(live) == 0:
            rows.append((pd.NaT, pd.NaT))
            continue
        k = 1 if np.busday_count(d, live[0]) <= roll_days else 0
        if len(live) <= k + back_offset:
            rows.append((pd.NaT, pd.NaT))
            continue
        rows.append((live[k], live[k + back_offset]))
    held = pd.DataFrame(rows, index=panel.index, columns=["front", "back"])
    return held.dropna().apply(pd.to_datetime)


def _quote(panel: pd.DataFrame, held: pd.DataFrame, leg: str, at: pd.DatetimeIndex) -> np.ndarray:
    cols = panel.columns.get_indexer(held[leg])
    rows = panel.index.get_indexer(at)
    out = np.full(len(at), np.nan)
    ok = (cols >= 0) & (rows >= 0)
    out[ok] = panel.to_numpy()[rows[ok], cols[ok]]
    return out


def spread_changes(panel: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Daily change in (back - front), always on the contracts held the day before.

    Returns the change and a `rolled` flag marking days on which the contracts
    held at the close differ from those held the previous close.
    """
    held = held.loc[held.index.isin(panel.index)]
    prev, cur = held.iloc[:-1], held.index[1:]
    f0 = _quote(panel, prev, "front", prev.index)
    b0 = _quote(panel, prev, "back", prev.index)
    f1 = _quote(panel, prev, "front", cur)
    b1 = _quote(panel, prev, "back", cur)
    out = pd.DataFrame({"d_spread": (b1 - b0) - (f1 - f0)}, index=cur)
    out["rolled"] = (held["front"].iloc[1:].values != held["front"].iloc[:-1].values)
    return out


def curve_slope(panel: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Front and back settlements laid out like the index panel, so the same
    signal code can run on the tradable curve."""
    held = held.loc[held.index.isin(panel.index)]
    return pd.DataFrame(
        {
            "VIX": _quote(panel, held, "front", held.index),
            "VIX3M": _quote(panel, held, "back", held.index),
        },
        index=held.index,
    )
