"""Deterministic SQL layers (DuckDB). Every number the agent reports comes from here."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).parent / "sql"
REASONS = ["buyer_remorse", "damage", "defect", "carrier_delay", "listing_error", "lower_price", "unknown"]


def _reason_columns() -> str:
    cols = []
    for r in REASONS:
        cols.append(f"SUM(CASE WHEN refund_reason = '{r}' THEN 1 ELSE 0 END) AS reason_{r},")
        cols.append(f"SUM(CASE WHEN refund_reason = '{r}' THEN refund ELSE 0 END) AS refund_{r},")
    return "\n    ".join(cols)


def run_pipeline(skus: pd.DataFrame, orders: pd.DataFrame, separate_return_freight: bool = True) -> pd.DataFrame:
    """Run the SQL layers in order and return the SKU feature table (layer 6)."""
    con = duckdb.connect()
    con.register("skus", skus)
    con.register("orders", orders)
    params = {"SEPARATE_RETURN_FREIGHT": "TRUE" if separate_return_freight else "FALSE",
              "REASON_COLUMNS": _reason_columns()}
    for f in sorted(SQL_DIR.glob("*.sql")):
        sql = f.read_text()
        for k, v in params.items():
            sql = sql.replace("{" + k + "}", v)
        con.execute(sql)
    out = con.execute("SELECT * FROM l6_features").df()
    con.close()
    return out
