"""Underlying OHLC loader.

The CBOE SPX file is close-only, but the Parkinson / Garman-Klass /
Yang-Zhang realized-volatility estimators need the full daily range.
"""

from __future__ import annotations

import pandas as pd
import requests

from ..config import UNDERLYING, YAHOO_URL
from ..db import now, upsert

TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (research; volrv/0.1)"}


def fetch_underlying(symbol: str = "SPX") -> pd.DataFrame:
    ticker = UNDERLYING[symbol]
    resp = requests.get(
        YAHOO_URL.format(symbol=ticker),
        params={"period1": 0, "period2": 4102444800, "interval": "1d"},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {
            "dt": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote.get("volume"),
        }
    ).dropna(subset=["open", "high", "low", "close"])

    # Enforce the CHECK constraints before they reach the database.
    ok = (
        (df["high"] >= df["low"])
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
    )
    df = df[ok]

    df["symbol"] = symbol
    df["source"] = "yahoo"
    df["ingested_at"] = now()
    return df[
        ["symbol", "dt", "open", "high", "low", "close", "volume", "source", "ingested_at"]
    ]


def load_underlying(symbol: str = "SPX", engine=None) -> int:
    return upsert(fetch_underlying(symbol), "underlying_daily", ["symbol", "dt"], engine=engine)
