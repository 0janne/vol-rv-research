"""Central configuration. Everything tunable lives here, not scattered in modules."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"

# Postgres is the intended store; SQLite keeps the test suite and CI serverless.
DEFAULT_DB = f"sqlite:///{DATA_DIR / 'volrv.db'}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB)

TRADING_DAYS = 252

# CBOE publishes these as free daily CSVs.
CBOE_INDICES = ("VIX", "VIX9D", "VIX3M", "VIX6M", "VVIX", "SKEW", "SPX")
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
UNDERLYING = {"SPX": "%5EGSPC"}

# Horizon of the variance contract we model, in trading days (~1 calendar month).
HORIZON_D = 21

# Round-trip transaction cost charged in *volatility points* of the traded leg.
# 0.50 vol points is a deliberately conservative retail-ish estimate for a
# one-month S&P variance position; see README "Costs".
COST_VOL_POINTS = 0.50

# Signals may only use data available strictly before the decision date.
SIGNAL_LAG_D = 1
