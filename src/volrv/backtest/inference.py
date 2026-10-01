"""Statistical inference for strategy returns.

A Sharpe ratio is an estimate, and for volatility strategies a noisy one:
returns are autocorrelated, fat-tailed and heavily skewed. Three tools, each
answering a different objection:

  newey_west_t              is the mean return distinguishable from zero once
                            autocorrelation is accounted for?
  block_bootstrap_sharpe_ci how wide is the Sharpe estimate, without assuming
                            normal returns?
  deflated_sharpe           does the Sharpe survive the fact that it was the
                            best of several configurations tried?
                            (Bailey & Lopez de Prado, 2014)

Run all three on non-overlapping (monthly) returns.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def newey_west_t(x, lags: int | None = None) -> tuple[float, float, float]:
    """Mean, HAC t-statistic and two-sided p-value (Bartlett kernel).

    Default bandwidth is the Newey-West (1994) rule floor(4 (T/100)^(2/9)).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = e @ e / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        lrv += 2.0 * w * (e[lag:] @ e[:-lag]) / n
    se = math.sqrt(max(lrv, 0.0) / n)
    t = mu / se if se > 0 else float("nan")
    p = 2.0 * (1.0 - stats.norm.cdf(abs(t))) if se > 0 else float("nan")
    return float(mu), float(t), float(p)


def sharpe(x, periods: int = 12) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def block_bootstrap_sharpe_ci(
    x, block: int = 6, n_boot: int = 5000, periods: int = 12, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Circular block bootstrap confidence interval for the annualised Sharpe.

    Blocks of `block` consecutive periods preserve short-range dependence, which
    an i.i.d. bootstrap would destroy.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samples = x[idx]
    sd = samples.std(axis=1, ddof=1)
    srs = np.where(sd > 0, samples.mean(axis=1) / sd * math.sqrt(periods), np.nan)
    lo, hi = np.nanpercentile(srs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """P(true Sharpe > benchmark), per-period Sharpe, non-excess kurtosis."""
    denom = math.sqrt(max(1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / denom))


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurtosis: float, n_trials: int, trial_sr_var: float
) -> tuple[float, float]:
    """Deflated Sharpe ratio: probability the selected strategy's true Sharpe is
    positive, after accounting for the best of `n_trials` being chosen.

    All Sharpe inputs are per-period (not annualised). Returns the probability
    and the per-period benchmark Sharpe that pure selection luck would produce.
    """
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sr0 = math.sqrt(trial_sr_var) * (
            (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )
    return probabilistic_sharpe(sr, sr0, n_obs, skew, kurtosis), float(sr0)
