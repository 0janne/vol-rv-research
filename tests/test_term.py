import numpy as np
import pytest

from volrv.vol.term import TENOR_DAYS, constant_maturity_vol, term_structure_slope


@pytest.mark.parametrize("col,days", sorted(TENOR_DAYS.items(), key=lambda kv: kv[1]))
def test_interpolation_reproduces_listed_tenors(wide, col, days):
    """At a listed tenor the interpolated curve must return that index exactly."""
    cmv = constant_maturity_vol(wide, days).dropna()
    np.testing.assert_allclose(cmv.to_numpy(), wide[col].reindex(cmv.index).to_numpy(), rtol=1e-9)


def test_interpolation_is_between_neighbours(wide):
    cmv = constant_maturity_vol(wide, 60).dropna()
    lo = wide["VIX"].reindex(cmv.index)
    hi = wide["VIX3M"].reindex(cmv.index)
    assert ((cmv >= np.minimum(lo, hi) - 1e-9) & (cmv <= np.maximum(lo, hi) + 1e-9)).all()


def test_slope_sign_convention(wide):
    slope = term_structure_slope(wide)
    contango = wide["VIX3M"] > wide["VIX"]
    assert (slope[contango] > 0).all()
    assert (slope[~contango] <= 0).all()


def test_requires_vix_family():
    import pandas as pd

    with pytest.raises(ValueError):
        constant_maturity_vol(pd.DataFrame({"SPX": [1.0, 2.0]}), 30)
