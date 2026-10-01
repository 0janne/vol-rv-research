import datetime as dt

import numpy as np
import pandas as pd
import pytest

from volrv.backtest import backtest_spread, held_contracts, spread_changes
from volrv.ingest.cboe_futures import _parse, standard_expiry


@pytest.mark.parametrize(
    "year,month,expected",
    [
        (2013, 1, "2013-01-16"),
        (2018, 2, "2018-02-14"),
        (2025, 1, "2025-01-22"),
        (2026, 10, "2026-10-21"),
    ],
)
def test_expiry_rule_matches_published_contracts(year, month, expected):
    assert standard_expiry(year, month).isoformat() == expected


def test_expiry_is_a_wednesday_thirty_days_before_a_third_friday():
    for y in range(2013, 2027):
        for m in range(1, 13):
            e = standard_expiry(y, m)
            assert e.weekday() == 2
            friday = e + dt.timedelta(days=30)
            assert friday.weekday() == 4 and 15 <= friday.day <= 21


def test_parser_drops_zero_settles_and_post_expiry_rows():
    csv = (
        "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest\n"
        "2026-01-26,V (Oct 2026),0,21,22,0,0,0,0,0,0\n"
        "2026-09-24,V (Oct 2026),17.65,18.2,17.6,17.91,17.8081,0.07,81394,0,219122\n"
        "2026-10-22,V (Oct 2026),17.65,18.2,17.6,17.91,17.80,0.07,1,0,1\n"
    )
    out = _parse(csv, dt.date(2026, 10, 21))
    assert len(out) == 1
    assert out["settle"].iloc[0] == pytest.approx(17.8081)


@pytest.fixture
def toy_curve():
    """Three contracts with linear prices so every change is known exactly."""
    dates = pd.bdate_range("2024-01-01", "2024-03-29")
    exps = pd.to_datetime(["2024-01-17", "2024-02-14", "2024-03-20", "2024-04-17", "2024-05-22"])
    panel = pd.DataFrame(index=dates, columns=exps, dtype=float)
    for j, e in enumerate(exps):
        live = dates[dates <= e]
        panel.loc[live, e] = 15.0 + j + 0.01 * np.arange(len(live))
    return panel


def test_front_is_rolled_before_expiry(toy_curve):
    held = held_contracts(toy_curve, roll_days=5)
    pairs = zip(held.index, held["front"], strict=True)
    days_left = [np.busday_count(d.date(), f.date()) for d, f in pairs]
    assert min(days_left) > 5
    assert (held["back"] > held["front"]).all()


def test_spread_change_uses_contracts_held_the_day_before(toy_curve):
    held = held_contracts(toy_curve, roll_days=5)
    ch = spread_changes(toy_curve, held)
    # Every contract rises 0.01/day, so the spread change is zero every day,
    # including roll days: a roll must never appear as a price jump.
    assert np.allclose(ch["d_spread"].dropna(), 0.0)
    assert ch["rolled"].sum() >= 2


def test_roll_and_trade_costs(toy_curve):
    held = held_contracts(toy_curve, roll_days=5)
    ch = spread_changes(toy_curve, held)
    pos = pd.Series(1.0, index=ch.index)
    bt = backtest_spread(pos, ch["d_spread"], cost_per_leg=0.05, rolled=ch["rolled"])
    n_rolls = int(ch["rolled"].iloc[1:].sum())
    # one entry (two legs) plus four contract-units per roll
    assert bt["cost"].sum() == pytest.approx(2 * 0.05 + n_rolls * 4 * 0.05)
