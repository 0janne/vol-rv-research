"""Loader for CBOE's free per-contract VIX futures settlement history.

CBOE publishes one CSV per monthly contract, named by its final settlement
date. The public endpoint covers contracts expiring from January 2013 onward;
earlier contracts sit in an archive that is not openly served.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io

import pandas as pd
import requests

from ..db import now, upsert

URL = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{date}.csv"
TIMEOUT = 30
FIRST_EXPIRY_YEAR = 2013


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def standard_expiry(year: int, month: int) -> dt.date:
    """Contract-month expiry rule: the Wednesday 30 days before the third Friday
    of the following calendar month. When that Wednesday is an exchange holiday
    the contract settles on the Tuesday before; `fetch_contract` tries both."""
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    return third_friday(ny, nm) - dt.timedelta(days=30)


def _parse(text: str, expiry: dt.date) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text))
    out = pd.DataFrame(
        {
            "expiry": pd.Timestamp(expiry),
            "dt": pd.to_datetime(raw["Trade Date"]),
            "settle": pd.to_numeric(raw["Settle"], errors="coerce"),
            "volume": pd.to_numeric(raw.get("Total Volume"), errors="coerce"),
            "open_interest": pd.to_numeric(raw.get("Open Interest"), errors="coerce"),
        }
    )
    out = out[(out["settle"] > 0) & (out["dt"] <= out["expiry"])]
    out["source"] = "cboe_cfe"
    out["ingested_at"] = now()
    return out


def fetch_contract(year: int, month: int, session=None) -> pd.DataFrame:
    """Settlement history for one contract month; empty frame if not published."""
    session = session or requests
    base = standard_expiry(year, month)
    for candidate in (base, base - dt.timedelta(days=1)):
        r = session.get(URL.format(date=candidate.isoformat()), timeout=TIMEOUT)
        if r.status_code == 200 and r.text.startswith("Trade Date"):
            return _parse(r.text, candidate)
    return pd.DataFrame()


def contract_months(first_year: int = FIRST_EXPIRY_YEAR, months_ahead: int = 9):
    today = dt.date.today()
    last = (today.year * 12 + today.month - 1) + months_ahead
    m = first_year * 12
    while m <= last:
        yield m // 12, m % 12 + 1
        m += 1


def load_vix_futures(engine=None, first_year: int = FIRST_EXPIRY_YEAR, workers: int = 8) -> int:
    with requests.Session() as s, cf.ThreadPoolExecutor(workers) as ex:
        months = list(contract_months(first_year))
        frames = list(ex.map(lambda ym: fetch_contract(*ym, session=s), months))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return 0
    data = pd.concat(frames, ignore_index=True)
    return upsert(data, "vix_future_daily", ["expiry", "dt"], engine=engine)
