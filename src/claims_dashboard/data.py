from __future__ import annotations
import pandas as pd

# Loads data from data/processed and ensures date columns are parsed as datetime types
# Returns policies and claims dataframes

def load_processed(processed_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = pd.read_parquet(f"{processed_dir}/policies.parquet")
    claims = pd.read_parquet(f"{processed_dir}/claims.parquet")

    for col in ["inception_date", "expiry_date"]:
        policies[col] = pd.to_datetime(policies[col], errors="coerce")

    for col in ["loss_date", "report_date", "close_date"]:
        claims[col] = pd.to_datetime(claims[col], errors="coerce")

    return policies, claims