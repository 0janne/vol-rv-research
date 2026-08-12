"""Database access. One engine, portable DDL, idempotent upserts."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATABASE_URL, ROOT


def get_engine(url: str | None = None):
    url = url or DATABASE_URL
    if url.startswith("sqlite"):
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, future=True)


def split_statements(ddl: str) -> list[str]:
    """Split a .sql file into statements.

    Comments are stripped first: a ';' inside a '--' comment is not a statement
    terminator, and splitting naively on ';' silently mangles the file.
    """
    lines = [ln.split("--", 1)[0] for ln in ddl.splitlines()]
    body = "\n".join(lines)
    return [s.strip() for s in body.split(";") if s.strip()]


def init_db(engine=None) -> None:
    """Apply every migration in sql/ in filename order. Safe to re-run."""
    engine = engine or get_engine()
    is_sqlite = engine.url.get_backend_name() == "sqlite"
    for path in sorted((ROOT / "sql").glob("*.sql")):
        ddl = path.read_text()
        if is_sqlite:
            # SQLite accepts the VARCHAR(n)/TIMESTAMP tokens as-is; only the
            # DOUBLE PRECISION spelling has no SQLite affinity rule.
            ddl = ddl.replace("DOUBLE PRECISION", "REAL")
        with engine.begin() as conn:
            for stmt in split_statements(ddl):
                conn.execute(text(stmt))


def upsert(df: pd.DataFrame, table: str, pk: list[str], engine=None, chunk: int = 500) -> int:
    """Insert-or-replace `df` into `table` keyed on `pk`.

    Deliberately not `df.to_sql(if_exists='append')`: re-running a loader must
    not duplicate rows, and must not silently drop corrections either.

    Rows are sent as chunked multi-row VALUES rather than one statement per row.
    Passing a list of dicts to execute() looks batched but psycopg2 issues one
    round-trip per row underneath -- fine against a local socket, and about ten
    minutes for a 10k-row option chain against a hosted database.
    """
    if df.empty:
        return 0
    engine = engine or get_engine()
    df = df.drop_duplicates(subset=pk).copy()
    cols = list(df.columns)
    collist = ", ".join(cols)
    is_pg = engine.url.get_backend_name() == "postgresql"

    records = df.to_dict("records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                r[k] = v.to_pydatetime()
            elif pd.isna(v):
                r[k] = None

    if is_pg:
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk)
        head = f"INSERT INTO {table} ({collist}) VALUES "
        tail = f" ON CONFLICT ({', '.join(pk)}) DO UPDATE SET {updates}"
    else:
        head = f"INSERT OR REPLACE INTO {table} ({collist}) VALUES "
        tail = ""

    with engine.begin() as conn:
        for start in range(0, len(records), chunk):
            batch = records[start : start + chunk]
            tuples, params = [], {}
            for i, row in enumerate(batch):
                keys = [f"{c}_{i}" for c in cols]
                tuples.append("(" + ", ".join(f":{k}" for k in keys) + ")")
                for c, k in zip(cols, keys, strict=True):
                    params[k] = row[c]
            conn.execute(text(head + ", ".join(tuples) + tail), params)
    return len(records)


def read(sql: str, engine=None, **params) -> pd.DataFrame:
    engine = engine or get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])
    return df


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)
