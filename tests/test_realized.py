import numpy as np
import pytest

from volrv.vol.realized import ESTIMATORS, realized_vol, realized_vol_forward


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_estimators_recover_known_volatility(ohlc, name):
    """Every estimator should land near the 20% used to generate the series."""
    v = realized_vol(ohlc, 252, name).dropna()
    assert 14.0 < v.mean() < 27.0, f"{name} mean {v.mean():.2f} implausible"


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_estimators_are_non_negative(ohlc, name):
    v = realized_vol(ohlc, 63, name).dropna()
    assert (v >= 0).all()


def test_window_length_controls_smoothness(ohlc):
    short = realized_vol(ohlc, 21).dropna()
    long = realized_vol(ohlc, 252).dropna()
    common = short.index.intersection(long.index)
    assert short.loc[common].std() > long.loc[common].std()


def test_forward_is_trailing_shifted(ohlc):
    w = 21
    trail = realized_vol(ohlc, w)
    fwd = realized_vol_forward(ohlc, w)
    np.testing.assert_allclose(
        fwd.dropna().to_numpy()[:200],
        trail.shift(-w).dropna().to_numpy()[:200],
        rtol=1e-12,
    )


def test_unknown_estimator_raises(ohlc):
    with pytest.raises(KeyError):
        realized_vol(ohlc, 21, "not_an_estimator")
