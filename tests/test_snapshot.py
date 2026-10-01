import datetime as dt

from volrv.ingest.snapshot import is_fresh, market_date

# 2026-09-04 16:00 New York (Friday close) and 2026-09-08 16:00 (Tuesday close)
FRI_CLOSE = 1788552000
TUE_CLOSE = 1788897600


def test_market_date_is_new_york_date():
    assert market_date({"regularMarketTime": FRI_CLOSE}) == dt.date(2026, 9, 4)
    assert market_date({}) is None


def test_holiday_run_is_rejected():
    # Labor Day, Monday 7 Sept 2026: the endpoint still serves Friday's chain.
    assert not is_fresh({"regularMarketTime": FRI_CLOSE}, today=dt.date(2026, 9, 7))


def test_normal_session_is_accepted():
    assert is_fresh({"regularMarketTime": TUE_CLOSE}, today=dt.date(2026, 9, 8))
