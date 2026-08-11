"""Term-structure relative value: front variance against back variance.

The trade is vega-neutral at inception and expresses the *shape* of the curve
rather than its level, which is the distinction between an outright short-vol
carry position and a relative-value one.
"""

from __future__ import annotations

import pandas as pd

from ..config import SIGNAL_LAG_D
from ..vol.term import term_structure_slope


def calendar_signal(
    wide: pd.DataFrame,
    front: str = "VIX",
    back: str = "VIX3M",
    lookback: int = 252,
    entry_z: float = 1.0,
    cap: float = 2.0,
) -> pd.Series:
    """Position in the *slope*. Positive = long back / short front.

    When the curve is unusually steep the back end is expensive relative to the
    front, so the mean-reversion trade is to sell the back and buy the front,
    i.e. a negative position. Sign convention is checked in the tests.
    """
    slope = term_structure_slope(wide, front, back)
    mu = slope.rolling(lookback, min_periods=lookback // 2).mean()
    sd = slope.rolling(lookback, min_periods=lookback // 2).std(ddof=1)
    z = (slope - mu) / sd.replace(0.0, pd.NA)

    pos = pd.Series(0.0, index=slope.index)
    pos[z > entry_z] = -1.0     # curve too steep -> fade the steepness
    pos[z < -entry_z] = 1.0     # inverted -> fade the inversion
    strength = (z.abs() - entry_z).clip(lower=0.0, upper=cap) / cap
    return (pos * strength).shift(SIGNAL_LAG_D).rename("calendar_position")
