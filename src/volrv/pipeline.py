"""Feature build, backtests and figures. The narrative layer above the modules."""

from __future__ import annotations

import json
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from .backtest import (  # noqa: E402
    backtest_calendar,
    backtest_spread,
    backtest_variance_carry,
    block_bootstrap_sharpe_ci,
    curve_slope,
    deflated_sharpe,
    futures_panel,
    held_contracts,
    newey_west_t,
    spread_changes,
    summarise,
    walk_forward,
)
from .backtest.futures import COST_PER_LEG  # noqa: E402
from .config import FIG_DIR, HORIZON_D  # noqa: E402
from .db import get_engine, now, read, upsert  # noqa: E402
from .signals import calendar_signal, vrp_ex_post, vrp_proxy, vrp_signal  # noqa: E402
from .vol import constant_maturity_vol, realized_vol, term_structure_slope  # noqa: E402
from .vol.realized import ESTIMATORS  # noqa: E402

PALETTE = {"bg": "#FFFFFF", "line": "#1F3864", "accent": "#C00000", "grey": "#8A8A8A"}


def load_panel(engine=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide index panel and SPX OHLC, aligned on the common trading calendar."""
    idx = read("SELECT symbol, dt, close FROM index_daily", engine=engine)
    wide = idx.pivot(index="dt", columns="symbol", values="close").sort_index()
    ohlc = read(
        "SELECT dt, open, high, low, close FROM underlying_daily WHERE symbol = 'SPX'",
        engine=engine,
    ).set_index("dt").sort_index()
    common = wide.index.intersection(ohlc.index)
    return wide.loc[common], ohlc.loc[common]


def build_features(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    wide, ohlc = load_panel(engine)
    ts = now()

    rows = []
    for est in ESTIMATORS:
        rv = realized_vol(ohlc, HORIZON_D, est)
        rows.append(
            pd.DataFrame({"symbol": "SPX", "dt": rv.index, "window_d": HORIZON_D,
                          "estimator": est, "value": rv.to_numpy(), "ingested_at": ts})
        )
    rv_df = pd.concat(rows).dropna(subset=["value"])
    upsert(rv_df, "realized_vol", ["symbol", "dt", "window_d", "estimator"], engine=engine)

    feats = {
        "vrp_proxy": vrp_proxy(wide["VIX"], ohlc),
        "vrp_ex_post": vrp_ex_post(wide["VIX"], ohlc),
        "ts_slope": term_structure_slope(wide),
        "cmv_30d": constant_maturity_vol(wide, 30),
        "cmv_60d": constant_maturity_vol(wide, 60),
    }
    fr = pd.concat(
        [pd.DataFrame({"name": k, "dt": v.index, "value": v.to_numpy()}) for k, v in feats.items()]
    ).dropna(subset=["value"])
    fr["ingested_at"] = ts
    upsert(fr, "feature", ["name", "dt"], engine=engine)
    return fr


def _persist(strategy: str, params: dict, bt: pd.DataFrame, engine) -> str:
    run_id = f"{strategy}-{uuid.uuid4().hex[:8]}"
    upsert(pd.DataFrame([{"run_id": run_id, "strategy": strategy,
                          "params_json": json.dumps(params, default=str),
                          "created_at": now()}]),
           "backtest_run", ["run_id"], engine=engine)
    out = bt.reset_index().rename(columns={"index": "dt"})
    out["run_id"] = run_id
    upsert(out[["run_id", "dt", "position", "gross_pnl", "cost", "net_pnl"]],
           "backtest_pnl", ["run_id", "dt"], engine=engine)
    return run_id


def run_backtests(engine=None) -> dict:
    engine = engine or get_engine()
    wide, ohlc = load_panel(engine)
    vix = wide["VIX"]
    results: dict[str, dict] = {}

    # --- 1. Naive always-on short variance: the null hypothesis. -------------
    naive_pos = pd.Series(-1.0, index=vix.index)
    naive = backtest_variance_carry(naive_pos, vix, ohlc)
    results["naive_short_variance"] = summarise(naive["net_pnl"], naive["position"])
    _persist("naive_short_variance", {"position": -1.0}, naive, engine)

    # --- 2. VRP-conditioned short variance, fixed parameters. ---------------
    cond_pos = vrp_signal(vix, ohlc, lookback=252, entry_z=0.0)
    cond = backtest_variance_carry(cond_pos, vix, ohlc)
    results["vrp_conditioned"] = summarise(cond["net_pnl"], cond["position"])
    _persist("vrp_conditioned", {"lookback": 252, "entry_z": 0.0}, cond, engine)

    # --- 3. Same, but every parameter chosen out of sample. -----------------
    grid = [{"lookback": lb, "entry_z": z}
            for lb in (126, 252, 504) for z in (-0.5, 0.0, 0.5, 1.0)]
    wf = walk_forward(
        build_position=lambda lookback, entry_z: vrp_signal(
            vix, ohlc, lookback=lookback, entry_z=entry_z),
        run_backtest=lambda p: backtest_variance_carry(p, vix, ohlc),
        grid=grid,
        index=vix.index,
    )
    if not wf.empty:
        results["vrp_walk_forward"] = summarise(wf["net_pnl"], wf["position"])
        results["vrp_walk_forward"]["selections"] = wf.attrs.get("selections", [])[:6]
        _persist("vrp_walk_forward", {"grid_size": len(grid)}, wf, engine)

    # --- 4. Term-structure relative value: index (not tradable) vs futures. --
    # Same signal, same timing, same costs. The only difference is whether the
    # spread is marked on the VIX indices or on futures that can be traded.
    series = {"naive": naive, "conditioned": cond, "walk_forward": wf}
    panel = futures_panel(engine)
    if not panel.empty:
        idx = wide[["VIX", "VIX3M"]].dropna()
        cal_pos = calendar_signal(idx, entry_z=1.0)
        for offset, tag in ((1, ""), (2, "_m3")):
            held = held_contracts(panel, back_offset=offset)
            ch = spread_changes(panel, held)
            common = ch.index.intersection(idx.index)
            ch = ch.loc[common]
            fut = backtest_spread(cal_pos, ch["d_spread"], COST_PER_LEG, ch["rolled"])
            results[f"calendar_futures{tag}"] = summarise(fut["net_pnl"], fut["position"])
            _persist(f"calendar_futures{tag}", {"entry_z": 1.0, "back_offset": offset}, fut, engine)
            series[f"calendar_futures{tag}"] = fut
            if offset == 1:
                ix = backtest_calendar(cal_pos.loc[common], idx.loc[common])
                results["calendar_index"] = summarise(ix["net_pnl"], ix["position"])
                _persist("calendar_index", {"entry_z": 1.0, "tradable": False}, ix, engine)
                series["calendar_index"] = ix
                free = backtest_spread(cal_pos, ch["d_spread"], 0.0, ch["rolled"])
                results["calendar_futures_nocost"] = summarise(free["net_pnl"], free["position"])
                series["calendar_futures_nocost"] = free
                fsig = calendar_signal(curve_slope(panel, held).loc[common], entry_z=1.0)
                fs = backtest_spread(fsig, ch["d_spread"], COST_PER_LEG, ch["rolled"])
                results["calendar_futures_curve_signal"] = summarise(
                    fs["net_pnl"], fs["position"]
                )
                series["calendar_futures_curve_signal"] = fs

    results["_inference"] = inference_table(series, vix, ohlc, grid)
    results["_series"] = series
    return results


def inference_table(series: dict, vix, ohlc, grid) -> pd.DataFrame:
    """Monthly, non-overlapping inference for every strategy."""
    rows = []
    for name, bt in series.items():
        if bt is None or bt.empty:
            continue
        m = bt["net_pnl"].resample("ME").sum()
        mu, t, p = newey_west_t(m.values)
        lo, hi = block_bootstrap_sharpe_ci(m.values)
        rows.append({
            "strategy": name, "start": m.index[0].strftime("%Y-%m"),
            "months": len(m), "sharpe": m.mean() / m.std(ddof=1) * np.sqrt(12),
            "nw_t": t, "p": p, "ci_lo": lo, "ci_hi": hi,
            "skew": float(sps.skew(m)), "worst_month": float(m.min()),
        })
    table = pd.DataFrame(rows).set_index("strategy")

    # Deflated Sharpe for the walk-forward: it selected among len(grid) configs.
    wf = series.get("walk_forward")
    if wf is not None and not wf.empty:
        trial = []
        for g in grid:
            m = backtest_variance_carry(vrp_signal(vix, ohlc, **g), vix, ohlc)["net_pnl"]
            m = m.resample("ME").sum()
            trial.append(m.mean() / m.std(ddof=1))
        m = wf["net_pnl"].resample("ME").sum()
        prob, sr0 = deflated_sharpe(
            m.mean() / m.std(ddof=1), len(m), float(sps.skew(m)),
            float(sps.kurtosis(m, fisher=False)), len(grid), float(np.var(trial, ddof=1)),
        )
        table.attrs["walk_forward_deflated"] = {
            "prob": prob, "benchmark_sharpe_ann": sr0 * np.sqrt(12), "n_trials": len(grid),
        }
    return table


# ---------------------------------------------------------------- figures ---
def _save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=11, color=PALETTE["line"], loc="left", weight="bold")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def make_figures(results: dict, engine=None) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    wide, ohlc = load_panel(engine)
    vix = wide["VIX"]
    paths = []

    # Fig 1: implied vs realised, and the ex-post premium.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    rv = realized_vol(ohlc, HORIZON_D, "yang_zhang")
    axes[0].plot(vix.index, vix, lw=0.7, color=PALETTE["line"], label="VIX (implied, 30d)")
    axes[0].plot(rv.index, rv, lw=0.7, color=PALETTE["accent"], alpha=0.8,
                 label="Realised (Yang-Zhang, 21d)")
    axes[0].legend(fontsize=8, frameon=False)
    _style(axes[0], "Implied vs realised volatility, S&P 500", "vol points")

    exp = vrp_ex_post(vix, ohlc)
    axes[1].fill_between(exp.index, 0, exp, where=exp >= 0, color=PALETTE["line"], alpha=0.6, lw=0)
    axes[1].fill_between(exp.index, 0, exp, where=exp < 0, color=PALETTE["accent"], alpha=0.8, lw=0)
    axes[1].axhline(0, color="k", lw=0.6)
    _style(
        axes[1],
        "Ex-post variance risk premium (implied\u00b2 \u2212 subsequently realised\u00b2)",
        "variance points",
    )
    fig.tight_layout()
    paths.append(_save(fig, "01_implied_vs_realised.png"))

    # Fig 2: the headline. Cumulative PnL, and where it goes wrong.
    s = results["_series"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, label, color, lw in (
        ("naive", "Always-on short variance", PALETTE["accent"], 1.2),
        ("conditioned", "VRP-conditioned (in-sample params)", PALETTE["grey"], 1.0),
        ("walk_forward", "VRP walk-forward (out-of-sample)", PALETTE["line"], 1.4),
    ):
        d = s.get(key)
        if d is not None and not d.empty:
            eq = d["net_pnl"].cumsum()
            ax.plot(eq.index, eq, lw=lw, color=color, label=label)
    for date, tag in (("2018-02-05", "Feb 2018"), ("2020-03-16", "Mar 2020")):
        ax.axvline(pd.Timestamp(date), color="k", ls=":", lw=0.8)
        ax.text(pd.Timestamp(date), ax.get_ylim()[0], f" {tag}",
                fontsize=7, rotation=90, va="bottom")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax, "Cumulative net PnL (volatility points, after costs)", "cumulative vol points")
    fig.tight_layout()
    paths.append(_save(fig, "02_cumulative_pnl.png"))

    # Fig 3: the same calendar signal, marked on the index and on futures.
    if "calendar_index" in s and "calendar_futures" in s:
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]}
        )
        slope = term_structure_slope(wide).loc[s["calendar_index"].index[0]:].dropna()
        axes[0].plot(slope.index, slope, lw=0.6, color=PALETTE["line"])
        axes[0].axhline(0, color=PALETTE["accent"], lw=0.8)
        _style(axes[0], "VIX curve slope (VIX3M \u2212 VIX) / VIX \u2014 below 0 = inverted",
               "ratio")
        for key, label, color, lw in (
            ("calendar_index", "Marked on the VIX indices (not tradable)", PALETTE["grey"], 1.2),
            ("calendar_futures", "Same trades in VIX futures, after costs", PALETTE["accent"], 1.6),
        ):
            eq = s[key]["net_pnl"].cumsum()
            axes[1].plot(eq.index, eq, lw=lw, color=color, label=label)
        axes[1].axhline(0, color="k", lw=0.6)
        axes[1].legend(fontsize=8, frameon=False, loc="upper left")
        _style(axes[1], "Same signal, two instruments: cumulative net PnL", "vol points")
        fig.tight_layout()
        paths.append(_save(fig, "03_index_vs_futures.png"))

    # Fig 4: the tail. Distribution of daily PnL, naive vs conditioned.
    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label, color in (("naive", "Always-on", PALETTE["accent"]),
                              ("walk_forward", "Walk-forward", PALETTE["line"])):
        d = s.get(key)
        if d is not None and not d.empty:
            ax.hist(d["net_pnl"].dropna(), bins=200, alpha=0.55, color=color, label=label, log=True)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Daily net PnL distribution (log count) \u2014 the left tail is the whole story",
           "log count")
    fig.tight_layout()
    paths.append(_save(fig, "04_pnl_distribution.png"))
    return paths
