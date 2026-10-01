import numpy as np
import pandas as pd

from volrv.backtest import backtest_calendar, backtest_variance_carry, summarise


def test_flat_position_earns_nothing(implied, ohlc):
    flat = pd.Series(0.0, index=implied.index)
    bt = backtest_variance_carry(flat, implied, ohlc)
    assert np.allclose(bt["net_pnl"], 0.0)


def test_short_variance_profits_when_implied_exceeds_realised(implied, ohlc):
    """Implied is generated above realised, so a short must make money overall."""
    short = pd.Series(-1.0, index=implied.index)
    bt = backtest_variance_carry(short, implied, ohlc, cost_vol_points=0.0)
    assert bt["net_pnl"].sum() > 0
    long_ = backtest_variance_carry(-short, implied, ohlc, cost_vol_points=0.0)
    np.testing.assert_allclose(
        bt["net_pnl"].to_numpy(), -long_["net_pnl"].to_numpy(), rtol=1e-10
    )


def test_costs_only_ever_reduce_pnl(implied, ohlc):
    pos = pd.Series(-1.0, index=implied.index)
    free = backtest_variance_carry(pos, implied, ohlc, cost_vol_points=0.0)
    charged = backtest_variance_carry(pos, implied, ohlc, cost_vol_points=1.0)
    assert charged["net_pnl"].sum() < free["net_pnl"].sum()
    assert (charged["cost"] >= 0).all()


def test_calendar_is_neutral_to_parallel_shifts(wide):
    """A parallel shift in the whole curve must produce no calendar PnL."""
    shifted = wide.copy()
    bump = pd.Series(np.linspace(0, 5, len(wide)), index=wide.index)
    for c in shifted.columns:
        shifted[c] = shifted[c] + bump

    pos = pd.Series(1.0, index=wide.index)
    base = backtest_calendar(pos, wide, cost_per_leg=0.0)
    bumped = backtest_calendar(pos, shifted, cost_per_leg=0.0)
    np.testing.assert_allclose(
        base["gross_pnl"].to_numpy(), bumped["gross_pnl"].to_numpy(), atol=1e-9
    )


def test_summarise_reports_expected_keys(implied, ohlc):
    pos = pd.Series(-1.0, index=implied.index)
    bt = backtest_variance_carry(pos, implied, ohlc)
    stats = summarise(bt["net_pnl"], bt["position"])
    for key in ("sharpe_daily", "sharpe_monthly", "max_drawdown", "skew", "pct_in_market"):
        assert key in stats
    assert stats["max_drawdown"] <= 0


def test_summarise_handles_empty():
    assert summarise(pd.Series(dtype=float)) == {}
