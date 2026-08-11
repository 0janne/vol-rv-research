"""Backtest engines.

PnL is modelled at index level -- a one-month variance position is priced off
the VIX and settled against realised variance, and the calendar trade is marked
off the constant-maturity curve. That is a *proxy* for the tradable instrument,
not the instrument itself: no bid-ask on individual strikes, no roll mechanics,
no margin. See README "What this does not model" before quoting any number
here as a tradable Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COST_VOL_POINTS, HORIZON_D, TRADING_DAYS
from ..vol.realized import realized_vol_forward


def backtest_variance_carry(
    position: pd.Series,
    implied: pd.Series,
    ohlc: pd.DataFrame,
    horizon: int = HORIZON_D,
    cost_vol_points: float = COST_VOL_POINTS,
    estimator: str = "close_to_close",
) -> pd.DataFrame:
    """Short-variance carry via a ladder of overlapping one-month tranches.

    A tranche opened at t settles at t+H against realised variance. One tranche
    is opened every trading day and each carries 1/H of the capital, so the book
    is fully invested and exactly one tranche settles per day -- which keeps the
    daily PnL series free of double counting.

    Payoff is stated per unit of *long* variance exposure, so that a negative
    position is a short and the sign convention matches the signal module:
        (RV^2 - K^2) / (2K)     with K = implied strike at inception
    Dividing by 2K is the standard vega-notional convention: PnL comes out in
    volatility points rather than variance points, which keeps it comparable
    across a 12-vol and a 60-vol regime.
    """
    idx = implied.index
    rv_fwd = realized_vol_forward(ohlc, horizon, estimator).reindex(idx)
    pos = position.reindex(idx).fillna(0.0)

    payoff = (rv_fwd**2 - implied**2) / (2.0 * implied)
    tranche_pnl = pos * payoff / horizon
    tranche_cost = pos.abs() * cost_vol_points / horizon

    # Attribute each tranche's PnL to its settlement date, t + H.
    gross = tranche_pnl.shift(horizon)
    cost = tranche_cost.shift(horizon)
    net = gross - cost

    out = pd.DataFrame(
        {
            "position": pos,
            "settled_position": pos.shift(horizon),
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": net,
        }
    )
    return out.dropna(subset=["net_pnl"])


def backtest_calendar(
    position: pd.Series,
    wide: pd.DataFrame,
    front: str = "VIX",
    back: str = "VIX3M",
    holding_d: int = 5,
    cost_vol_points: float = 0.20,
) -> pd.DataFrame:
    """Vega-neutral front-vs-back variance spread, marked daily.

    Both legs are held in equal vega notional, so a parallel 1-vol shift across
    the curve nets to zero and the PnL comes from the change in *slope* alone.
    Without that the trade is a disguised outright short.
    """
    idx = wide.index
    pos = position.reindex(idx).fillna(0.0)

    f, b = wide[front], wide[back]

    # Equal and opposite vega notional on each leg -> pure slope exposure.
    d_front = f.diff(holding_d).shift(-holding_d)
    d_back = b.diff(holding_d).shift(-holding_d)
    spread_move = d_back - d_front

    gross = pos * spread_move / holding_d
    cost = pos.diff().abs().fillna(0.0) * cost_vol_points / holding_d
    net = gross - cost

    out = pd.DataFrame(
        {"position": pos, "gross_pnl": gross, "cost": cost, "net_pnl": net}
    )
    return out.dropna(subset=["net_pnl"])


def walk_forward(
    build_position,
    run_backtest,
    grid: list[dict],
    index: pd.DatetimeIndex,
    train_years: int = 5,
    step_months: int = 12,
    min_train_days: int = 756,
) -> pd.DataFrame:
    """Expanding-window walk-forward.

    At each rebalance the parameter set with the best in-sample monthly Sharpe
    over the data available *so far* is selected, then applied to the following
    `step_months` of unseen data. The concatenated out-of-sample PnL is the only
    series that should be quoted; the in-sample choice is free to look good.
    """
    index = pd.DatetimeIndex(index).sort_values()
    start = index[0] + pd.DateOffset(years=train_years)
    edges = pd.date_range(start, index[-1], freq=pd.DateOffset(months=step_months))

    cached = {i: run_backtest(build_position(**p)) for i, p in enumerate(grid)}
    chunks, chosen = [], []

    for k, edge in enumerate(edges):
        nxt = edges[k + 1] if k + 1 < len(edges) else index[-1] + pd.Timedelta(days=1)
        best_i, best_s = None, -np.inf
        for i, _ in enumerate(grid):
            tr = cached[i].loc[cached[i].index < edge, "net_pnl"].dropna()
            if len(tr) < min_train_days:
                continue
            m = tr.resample("ME").sum()
            s = m.mean() * 12 / (m.std(ddof=1) * np.sqrt(12)) if m.std(ddof=1) > 0 else -np.inf
            if s > best_s:
                best_i, best_s = i, s
        if best_i is None:
            continue
        oos = cached[best_i].loc[(cached[best_i].index >= edge) & (cached[best_i].index < nxt)]
        if not oos.empty:
            chunks.append(oos)
            chosen.append({"from": edge.date().isoformat(), **grid[best_i],
                           "is_sharpe": round(float(best_s), 3)})

    if not chunks:
        return pd.DataFrame()
    result = pd.concat(chunks).sort_index()
    result.attrs["selections"] = chosen
    return result


def annualise_note() -> str:
    return f"PnL is in volatility points; {TRADING_DAYS} trading days assumed per year."
