"""Performance statistics.

Reported on the net PnL series. Where a number can be inflated by the way the
strategy is constructed (overlapping holding periods), the honest version is
reported alongside it rather than instead of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS


def max_drawdown(equity: pd.Series) -> float:
    return float((equity - equity.cummax()).min())


def summarise(pnl: pd.Series, position: pd.Series | None = None) -> dict[str, float]:
    pnl = pnl.dropna()
    if pnl.empty:
        return {}
    equity = pnl.cumsum()
    mu, sd = pnl.mean(), pnl.std(ddof=1)
    ann_ret = mu * TRADING_DAYS
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = pnl[pnl < 0].std(ddof=1)
    sortino = ann_ret / (downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan

    # Overlapping holding periods smooth the daily series and flatter Sharpe.
    # Monthly buckets are non-overlapping, so this is the number to trust.
    monthly = pnl.resample("ME").sum()
    m_sharpe = (
        monthly.mean() * 12 / (monthly.std(ddof=1) * np.sqrt(12))
        if monthly.std(ddof=1) > 0
        else np.nan
    )

    mdd = max_drawdown(equity)
    out = {
        "start": pnl.index[0].date().isoformat(),
        "end": pnl.index[-1].date().isoformat(),
        "n_days": int(len(pnl)),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe_daily": float(sharpe),
        "sharpe_monthly": float(m_sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else np.nan,
        "hit_rate": float((pnl > 0).mean()),
        "skew": float(pnl.skew()),
        "excess_kurtosis": float(pnl.kurtosis()),
        "worst_day": float(pnl.min()),
        "worst_month": float(monthly.min()),
        "total_pnl": float(pnl.sum()),
    }
    if position is not None:
        turn = position.diff().abs().dropna()
        out["ann_turnover"] = float(turn.mean() * TRADING_DAYS)
        # A signal that sits flat most of the time has a misleadingly low raw
        # hit rate, because flat days are counted as losses. Report both.
        active = position.reindex(pnl.index).abs() > 1e-12
        out["pct_in_market"] = float(active.mean())
        if active.any():
            out["hit_rate_active"] = float((pnl[active] > 0).mean())
    return out
