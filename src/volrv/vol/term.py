"""Constant-maturity volatility curve from the CBOE index complex.

The listed tenors (VIX9D 9d, VIX 30d, VIX3M 93d, VIX6M 186d) are already
constant-maturity, so a curve at an arbitrary tenor is an interpolation in
*total variance* against calendar days -- interpolating volatility directly
would not be arbitrage-consistent across tenors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TENOR_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 186}


def constant_maturity_vol(wide: pd.DataFrame, target_days: float) -> pd.Series:
    """Interpolate the vol curve to `target_days` in total-variance space."""
    available = [c for c in TENOR_DAYS if c in wide.columns]
    if not available:
        raise ValueError("no VIX-family columns present")
    order = sorted(available, key=lambda c: TENOR_DAYS[c])
    taus = np.array([TENOR_DAYS[c] for c in order], dtype=float)
    vols = wide[order].to_numpy(dtype=float)

    total_var = (vols**2) * taus  # variance x time
    out = np.full(len(wide), np.nan)
    for i in range(len(wide)):
        row = total_var[i]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        interp = np.interp(target_days, taus[mask], row[mask])
        out[i] = np.sqrt(max(interp, 0.0) / target_days)
    return pd.Series(out, index=wide.index, name=f"cmv_{int(target_days)}d")


def term_structure_slope(wide: pd.DataFrame, front: str = "VIX", back: str = "VIX3M") -> pd.Series:
    """Relative slope (back - front) / front.

    Positive is contango, the usual state; negative is backwardation, which
    equity vol enters in stress. Normalising by the front level keeps the series
    comparable between a 12-vol and a 60-vol regime.
    """
    return ((wide[back] - wide[front]) / wide[front]).rename("ts_slope")
