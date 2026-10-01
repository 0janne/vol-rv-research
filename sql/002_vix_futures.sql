-- VIX futures daily settlements, one row per contract per trading day.
--
-- The VIX index itself cannot be bought. Any strategy on the shape of the VIX
-- curve has to be expressed in these contracts, whose prices already embed the
-- market's expectation of where the index will go. See README, "Index versus
-- tradable".

CREATE TABLE IF NOT EXISTS vix_future_daily (
    expiry        DATE        NOT NULL,
    dt            DATE        NOT NULL,
    settle        DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION,
    open_interest DOUBLE PRECISION,
    source        VARCHAR(32) NOT NULL,
    ingested_at   TIMESTAMP   NOT NULL,
    PRIMARY KEY (expiry, dt),
    CHECK (dt <= expiry),
    CHECK (settle > 0)
);

CREATE INDEX IF NOT EXISTS ix_vix_future_daily_dt ON vix_future_daily (dt);
