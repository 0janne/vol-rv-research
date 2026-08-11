import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from volrv.db import get_engine, init_db, read, split_statements, upsert


@pytest.fixture
def engine(tmp_path):
    eng = get_engine(f"sqlite:///{tmp_path / 't.db'}")
    init_db(eng)
    return eng


def _row(dt_, close):
    return {
        "symbol": "VIX", "dt": dt_, "open": None, "high": None, "low": None,
        "close": close, "source": "test", "ingested_at": dt.datetime(2026, 1, 1),
    }


def test_split_statements_ignores_semicolons_in_comments():
    sql = "-- a; b; c\nCREATE TABLE x (i INT);\n-- trailing; comment\nCREATE TABLE y (j INT);"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert all(s.strip().startswith("CREATE") for s in stmts)


def test_upsert_is_idempotent(engine):
    df = pd.DataFrame([_row(dt.date(2020, 1, 2), 15.0)])
    upsert(df, "index_daily", ["symbol", "dt"], engine=engine)
    upsert(df, "index_daily", ["symbol", "dt"], engine=engine)
    assert len(read("SELECT * FROM index_daily", engine=engine)) == 1


def test_upsert_applies_corrections(engine):
    upsert(pd.DataFrame([_row(dt.date(2020, 1, 2), 15.0)]), "index_daily",
           ["symbol", "dt"], engine=engine)
    upsert(pd.DataFrame([_row(dt.date(2020, 1, 2), 16.5)]), "index_daily",
           ["symbol", "dt"], engine=engine)
    out = read("SELECT close FROM index_daily", engine=engine)
    assert len(out) == 1 and out["close"].iloc[0] == 16.5


def test_upsert_deduplicates_within_a_batch(engine):
    df = pd.DataFrame([_row(dt.date(2020, 1, 2), 15.0), _row(dt.date(2020, 1, 2), 15.0)])
    assert upsert(df, "index_daily", ["symbol", "dt"], engine=engine) == 1


def test_empty_frame_is_a_noop(engine):
    assert upsert(pd.DataFrame(), "index_daily", ["symbol", "dt"], engine=engine) == 0


def test_option_quote_rejects_expiry_before_snapshot(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO option_quote (underlying, snapshot_dt, expiry, strike, cp,"
                " ingested_at) VALUES ('SPY', '2026-06-01', '2026-05-01', 500, 'C',"
                " '2026-06-01')"
            ))


def test_underlying_rejects_impossible_bar(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO underlying_daily (symbol, dt, open, high, low, close,"
                " source, ingested_at) VALUES ('SPX','2020-01-02',100,90,95,98,"
                "'test','2020-01-03')"
            ))
