-- Point-in-time research store.
--
-- Design rules enforced here:
--   1. Every fact table carries ingested_at. A row is only usable by a backtest
--      if ingested_at <= the simulated decision time. This is what makes the
--      store point-in-time rather than merely historical.
--   2. Natural primary keys, so re-running a loader is idempotent (upsert).
--   3. Derived tables (realized_vol, feature, backtest_*) are rebuildable from
--      the raw tables alone; nothing downstream is a source of truth.

CREATE TABLE IF NOT EXISTS index_daily (
    symbol       VARCHAR(16)  NOT NULL,
    dt           DATE         NOT NULL,
    open         DOUBLE PRECISION,
    high         DOUBLE PRECISION,
    low          DOUBLE PRECISION,
    close        DOUBLE PRECISION NOT NULL,
    source       VARCHAR(32)  NOT NULL,
    ingested_at  TIMESTAMP    NOT NULL,
    PRIMARY KEY (symbol, dt)
);

CREATE TABLE IF NOT EXISTS underlying_daily (
    symbol       VARCHAR(16)  NOT NULL,
    dt           DATE         NOT NULL,
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION,
    source       VARCHAR(32)  NOT NULL,
    ingested_at  TIMESTAMP    NOT NULL,
    PRIMARY KEY (symbol, dt),
    CHECK (high >= low),
    CHECK (high >= open AND high >= close),
    CHECK (low  <= open AND low  <= close)
);

-- Option chain snapshots. The recorder (volrv snapshot) appends one row per
-- quote per day. No vendor history is redistributed with this repo; the table
-- fills forward from the day the recorder is first scheduled.
CREATE TABLE IF NOT EXISTS option_quote (
    underlying    VARCHAR(16) NOT NULL,
    snapshot_dt   DATE        NOT NULL,
    expiry        DATE        NOT NULL,
    strike        DOUBLE PRECISION NOT NULL,
    cp            CHAR(1)     NOT NULL,
    bid           DOUBLE PRECISION,
    ask           DOUBLE PRECISION,
    last          DOUBLE PRECISION,
    implied_vol   DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    volume        DOUBLE PRECISION,
    spot          DOUBLE PRECISION,
    ingested_at   TIMESTAMP   NOT NULL,
    PRIMARY KEY (underlying, snapshot_dt, expiry, strike, cp),
    CHECK (cp IN ('C', 'P')),
    CHECK (expiry >= snapshot_dt)
);

CREATE TABLE IF NOT EXISTS realized_vol (
    symbol       VARCHAR(16) NOT NULL,
    dt           DATE        NOT NULL,
    window_d     INTEGER     NOT NULL,
    estimator    VARCHAR(24) NOT NULL,
    value        DOUBLE PRECISION,
    ingested_at  TIMESTAMP   NOT NULL,
    PRIMARY KEY (symbol, dt, window_d, estimator)
);

CREATE TABLE IF NOT EXISTS feature (
    name         VARCHAR(48) NOT NULL,
    dt           DATE        NOT NULL,
    value        DOUBLE PRECISION,
    ingested_at  TIMESTAMP   NOT NULL,
    PRIMARY KEY (name, dt)
);

CREATE TABLE IF NOT EXISTS backtest_run (
    run_id       VARCHAR(40)  NOT NULL PRIMARY KEY,
    strategy     VARCHAR(48)  NOT NULL,
    params_json  TEXT         NOT NULL,
    created_at   TIMESTAMP    NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_pnl (
    run_id       VARCHAR(40) NOT NULL,
    dt           DATE        NOT NULL,
    position     DOUBLE PRECISION,
    gross_pnl    DOUBLE PRECISION,
    cost         DOUBLE PRECISION,
    net_pnl      DOUBLE PRECISION,
    PRIMARY KEY (run_id, dt)
);

CREATE INDEX IF NOT EXISTS ix_index_daily_dt      ON index_daily (dt);
CREATE INDEX IF NOT EXISTS ix_realized_vol_dt     ON realized_vol (dt);
CREATE INDEX IF NOT EXISTS ix_feature_dt          ON feature (dt);
CREATE INDEX IF NOT EXISTS ix_option_quote_expiry ON option_quote (underlying, snapshot_dt, expiry);
