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

import pandas as pd
import requests

from ..db import now, upsert

TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
CHAIN_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
COOKIE_URL = "https://fc.yahoo.com"


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
    expiries = payload.get("expirationDates", [])
    spot = payload.get("quote", {}).get("regularMarketPrice")

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
    chain["snapshot_dt"] = pd.Timestamp(dt.date.today())
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
