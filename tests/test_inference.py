import math

import numpy as np
import pytest

from volrv.backtest import block_bootstrap_sharpe_ci, deflated_sharpe, newey_west_t
from volrv.backtest.inference import probabilistic_sharpe, sharpe


def test_newey_west_with_zero_lags_is_the_plain_t_stat():
    rng = np.random.default_rng(0)
    x = rng.normal(0.2, 1.0, 400)
    _, t, _ = newey_west_t(x, lags=0)
    plain = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert t == pytest.approx(plain, rel=1e-12)


def test_positive_autocorrelation_shrinks_the_t_stat():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 2000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.6 * x[i - 1] + e[i]
    x += 0.1
    _, t0, _ = newey_west_t(x, lags=0)
    _, t8, _ = newey_west_t(x, lags=8)
    assert abs(t8) < abs(t0)


def test_bootstrap_ci_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    x = rng.normal(0.5, 1.0, 240)
    lo, hi = block_bootstrap_sharpe_ci(x, seed=3)
    assert lo < sharpe(x) < hi
    assert (lo, hi) == block_bootstrap_sharpe_ci(x, seed=3)


def test_more_trials_means_a_higher_bar():
    p1, sr0_1 = deflated_sharpe(0.2, 200, 0.0, 3.0, n_trials=1, trial_sr_var=0.01)
    p50, sr0_50 = deflated_sharpe(0.2, 200, 0.0, 3.0, n_trials=50, trial_sr_var=0.01)
    assert sr0_1 == 0.0 and sr0_50 > 0
    assert p50 < p1
    assert p1 == pytest.approx(probabilistic_sharpe(0.2, 0.0, 200, 0.0, 3.0))


def test_negative_skew_lowers_confidence():
    normal = probabilistic_sharpe(0.15, 0.0, 200, 0.0, 3.0)
    crashy = probabilistic_sharpe(0.15, 0.0, 200, -3.0, 20.0)
    assert crashy < normal
