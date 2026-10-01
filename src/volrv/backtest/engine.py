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


def backtest_spread(
    position: pd.Series,
    d_spread: pd.Series,
    cost_per_leg: float,
    rolled: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily PnL of a two-leg spread position.

    `position` is the target held over (t, t+1], already lagged by the signal
    code; `d_spread` on day t is the change in (back - front) from t-1 to t.
    Costs: each unit of position change trades two legs; a roll closes and
    reopens the spread, which is four contract-units per unit held.
    """
    idx = d_spread.index
    pos = position.reindex(idx).fillna(0.0)
    held_pos = pos.shift(1).fillna(0.0)
    gross = held_pos * d_spread
    cost = pos.diff().fillna(pos).abs() * 2.0 * cost_per_leg  # entry is charged too
    if rolled is not None:
        roll = rolled.reindex(idx).fillna(False).astype(float)
        cost = cost + roll * held_pos.abs() * 4.0 * cost_per_leg
    out = pd.DataFrame({"position": pos, "gross_pnl": gross, "cost": cost})
    out["net_pnl"] = out["gross_pnl"] - out["cost"]
    return out.dropna(subset=["net_pnl"])


def backtest_calendar(
    position: pd.Series,
    wide: pd.DataFrame,
    front: str = "VIX",
    back: str = "VIX3M",
    cost_per_leg: float = 0.05,
) -> pd.DataFrame:
    """Front-vs-back spread marked on the VIX *indices*. NOT TRADABLE.

    Kept as the counterfactual: it shows what a curve signal appears to earn
    when the curve itself could be traded. `backtest.futures` runs the same
    position on futures that can be. Equal and opposite vega notional on each
    leg, so a parallel shift of the curve nets to zero.
    """
    d_spread = (wide[back] - wide[front]).diff()
    return backtest_spread(position, d_spread, cost_per_leg)


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
