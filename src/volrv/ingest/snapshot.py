"""Daily option-chain recorder.

Historical option chains are expensive. Rather than backfill from a vendor,
this records a chain every trading day from the moment it is scheduled, so the
resulting table is point-in-time *by construction*: no restatements, no
survivorship, and the quote you see is the quote that was there.

Schedule it (cron or a GitHub Action) shortly after the US close:
    volrv snapshot --underlying SPY
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from ..db import now, upsert

TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CHAIN_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
COOKIE_URL = "https://fc.yahoo.com"
NEW_YORK = ZoneInfo("America/New_York")


def market_date(quote: dict) -> dt.date | None:
    """Trading date of the quote's last regular-session print, in New York time."""
    ts = quote.get("regularMarketTime")
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(int(ts), NEW_YORK).date()


def is_fresh(quote: dict, today: dt.date | None = None) -> bool:
    """True only if the quote comes from today's regular session.

    On an exchange holiday the endpoint still answers, with the previous
    session's quotes. Recording those under today's date would put a stale
    chain into a table that claims to be point-in-time.
    """
    today = today or dt.datetime.now(NEW_YORK).date()
    return market_date(quote) == today


def _authenticated_session() -> tuple[requests.Session, str]:
    """Yahoo's v7 options endpoint returns 401 without a session cookie and a
    matching crumb token. Establish both once and reuse them for the whole
    chain walk, rather than per-expiry."""
    session = requests.Session()
    session.headers.update(HEADERS)
    # This request 404s by design; the value is in the Set-Cookie header.
    session.get(COOKIE_URL, timeout=TIMEOUT)
    crumb = session.get(CRUMB_URL, timeout=TIMEOUT).text.strip()
    if not crumb or "<" in crumb:
        raise RuntimeError("could not obtain a Yahoo crumb token")
    return session, crumb


def _leg(rows: list[dict], cp: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = {
        "strike": "strike",
        "bid": "bid",
        "ask": "ask",
        "lastPrice": "last",
        "impliedVolatility": "implied_vol",
        "openInterest": "open_interest",
        "volume": "volume",
        "expiration": "expiry",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df["cp"] = cp
    return df


def fetch_chain(underlying: str = "SPY") -> pd.DataFrame:
    """Fetch every listed expiry for `underlying` as one tidy frame."""
    session, crumb = _authenticated_session()
    base = session.get(
        CHAIN_URL.format(symbol=underlying), params={"crumb": crumb}, timeout=TIMEOUT
    )
    base.raise_for_status()
    payload = base.json()["optionChain"]["result"][0]
    quote = payload.get("quote", {})
    if not is_fresh(quote):
        return pd.DataFrame()
    expiries = payload.get("expirationDates", [])
    spot = quote.get("regularMarketPrice")

    frames = []
    for ts in expiries:
        r = session.get(
            CHAIN_URL.format(symbol=underlying),
            params={"date": ts, "crumb": crumb},
            timeout=TIMEOUT,
        )
        if r.status_code != 200:
            continue
        blocks = r.json()["optionChain"]["result"][0].get("options", [])
        if not blocks:
            continue
        for cp, key in (("C", "calls"), ("P", "puts")):
            leg = _leg(blocks[0].get(key, []), cp)
            if not leg.empty:
                frames.append(leg)

    if not frames:
        return pd.DataFrame()

    chain = pd.concat(frames, ignore_index=True)
    chain["expiry"] = pd.to_datetime(chain["expiry"], unit="s").dt.normalize()
    chain["underlying"] = underlying
    chain["snapshot_dt"] = pd.Timestamp(market_date(quote))
    chain["spot"] = spot
    chain["ingested_at"] = now()
    return chain[
        [
            "underlying", "snapshot_dt", "expiry", "strike", "cp", "bid", "ask", "last",
            "implied_vol", "open_interest", "volume", "spot", "ingested_at",
        ]
    ]


def record_chain_snapshot(underlying: str = "SPY", engine=None) -> int:
    chain = fetch_chain(underlying)
    return upsert(
        chain,
        "option_quote",
        ["underlying", "snapshot_dt", "expiry", "strike", "cp"],
        engine=engine,
    )


def audit_snapshots(engine=None, underlying: str = "SPY", delete: bool = False) -> list[dt.date]:
    """Find snapshot days whose spot equals the previous day's to the cent.

    Two consecutive sessions closing at an identical price is possible but rare;
    a repeated chain recorded on an exchange holiday is the usual cause. With
    `delete=True` the flagged days are removed.
    """
    from sqlalchemy import text

    from ..db import get_engine, read

    engine = engine or get_engine()
    days = read(
        "SELECT snapshot_dt, MAX(spot) AS spot, COUNT(*) AS n FROM option_quote "
        "WHERE underlying = :u GROUP BY snapshot_dt ORDER BY snapshot_dt",
        engine=engine,
        u=underlying,
    )
    if days.empty:
        return []
    days["snapshot_dt"] = pd.to_datetime(days["snapshot_dt"])
    repeat = days["spot"].round(2).eq(days["spot"].shift(1).round(2))
    flagged = [d.date() for d in days.loc[repeat, "snapshot_dt"]]
    if delete and flagged:
        with engine.begin() as conn:
            for d in flagged:
                conn.execute(
                    text("DELETE FROM option_quote WHERE underlying = :u AND snapshot_dt = :d"),
                    {"u": underlying, "d": d},
                )
    return flagged
