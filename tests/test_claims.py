import pandas as pd

from volrv.claims import CLAIMS, check_claims


def _table(overrides=None):
    base = {
        ("naive", "nw_t"): 4.4, ("naive", "skew"): -10.0,
        ("calendar_index", "sharpe"): 0.95, ("calendar_futures", "sharpe"): -0.77,
        ("calendar_futures_nocost", "nw_t"): 0.7,
    }
    base.update(overrides or {})
    t = pd.DataFrame(columns=["sharpe", "nw_t", "skew"], dtype=float)
    for (row, col), v in base.items():
        t.loc[row, col] = v
    return t


def test_current_results_satisfy_every_claim():
    assert all(ok for _, ok, _ in check_claims(_table()))


def test_a_futures_edge_would_break_the_claim():
    results = check_claims(_table({("calendar_futures", "sharpe"): 0.4}))
    failed = [c.strategy for c, ok, _ in results if not ok]
    assert failed == ["calendar_futures"]


def test_every_claim_names_a_column_the_report_writes():
    written = {"sharpe", "nw_t", "p", "ci_lo", "ci_hi", "skew", "worst_month", "months", "start"}
    assert all(c.column in written for c in CLAIMS)
