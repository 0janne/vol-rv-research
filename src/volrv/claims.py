"""The README's headline claims, as executable checks.

`volrv report` writes reports/inference.csv; `volrv check-claims` reads it and
fails if any claim no longer holds on the latest data. The end-to-end workflow
runs both weekly, so the README cannot quietly drift away from the code.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ROOT

INFERENCE_CSV = ROOT / "reports" / "inference.csv"


@dataclass(frozen=True)
class Claim:
    text: str
    strategy: str
    column: str
    op: str
    threshold: float

    def holds(self, table: pd.DataFrame) -> tuple[bool, float]:
        value = float(table.loc[self.strategy, self.column])
        ok = {
            ">": value > self.threshold,
            "<": value < self.threshold,
            "abs<": abs(value) < self.threshold,
        }[self.op]
        return ok, value


CLAIMS = (
    Claim("Variance risk premium is significant", "naive", "nw_t", ">", 2.0),
    Claim("Short variance carries a heavy left tail", "naive", "skew", "<", -3.0),
    Claim("Curve signal looks strong on the VIX indices", "calendar_index", "sharpe", ">", 0.5),
    Claim("Same trades lose money in futures after costs", "calendar_futures", "sharpe", "<", 0.0),
    Claim("Before costs, futures PnL is not significant",
          "calendar_futures_nocost", "nw_t", "abs<", 2.0),
)


def check_claims(table: pd.DataFrame | None = None) -> list[tuple[Claim, bool, float]]:
    if table is None:
        table = pd.read_csv(INFERENCE_CSV, index_col=0)
    return [(c, *c.holds(table)) for c in CLAIMS]
