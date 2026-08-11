"""Loader for CBOE's free daily index history (VIX complex, SKEW, SPX)."""

from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import CBOE_INDICES, CBOE_URL
from ..db import now, upsert

TIMEOUT = 30


def fetch_cboe_index(symbol: str) -> pd.DataFrame:
    """Return tidy OHLC for one CBOE index.

    The CSVs are not uniform: the VIX family ships DATE,OPEN,HIGH,LOW,CLOSE
    while VVIX/SKEW/SPX ship DATE,<SYMBOL> only. Normalise both to one shape.
    """
    resp = requests.get(CBOE_URL.format(symbol=symbol), timeout=TIMEOUT)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    raw.columns = [c.strip().upper() for c in raw.columns]

    out = pd.DataFrame()
    out["dt"] = pd.to_datetime(raw["DATE"], format="mixed")
    if {"OPEN", "HIGH", "LOW", "CLOSE"}.issubset(raw.columns):
        for c in ("open", "high", "low", "close"):
            out[c] = pd.to_numeric(raw[c.upper()], errors="coerce")
    else:
        value_col = [c for c in raw.columns if c != "DATE"][0]
        close = pd.to_numeric(raw[value_col], errors="coerce")
        out["open"] = out["high"] = out["low"] = None
        out["close"] = close

    out = out.dropna(subset=["close"])
    # A handful of pre-1992 rows carry zero placeholders.
    out = out[out["close"] > 0]
    out["symbol"] = symbol
    out["source"] = "cboe"
    out["ingested_at"] = now()
    return out[["symbol", "dt", "open", "high", "low", "close", "source", "ingested_at"]]


def load_cboe_indices(symbols=CBOE_INDICES, engine=None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sym in symbols:
        df = fetch_cboe_index(sym)
        counts[sym] = upsert(df, "index_daily", ["symbol", "dt"], engine=engine)
    return counts
