from __future__ import annotations
import numpy as np
import pandas as pd

# Function that adds derived fields to copy of claims df

def add_derived_fields(claims: pd.DataFrame) -> pd.DataFrame:
    df = claims.copy()
    df["report_delay_days"] = (df["report_date"] - df["loss_date"]).dt.days
    df["close_delay_days"] = (df["close_date"] - df["report_date"]).dt.days
    df["loss_month"] = df["loss_date"].dt.to_period("M").dt.to_timestamp()
    return df

# Function that calculates KPIs from policies and claims dataframes
# Takes two dataframes as input and returns a dictionary of KPIs

def kpis(policies: pd.DataFrame, claims: pd.DataFrame) -> dict:
    exposure = float(policies["exposure_years"].sum()) if not policies.empty else 0.0
    earned_premium = float(policies["earned_premium"].sum()) if not policies.empty else 0.0
    n_claims = int(len(claims))

    incurred = float(claims["incurred"].sum()) if n_claims else 0.0
    paid = float(claims["paid"].sum()) if n_claims else 0.0

    freq = n_claims / exposure if exposure > 0 else np.nan
    sev = incurred / n_claims if n_claims > 0 else np.nan
    loss_cost = incurred / exposure if exposure > 0 else np.nan
    loss_ratio = incurred / earned_premium if earned_premium > 0 else np.nan

    return {
        "exposure_years": exposure,
        "earned_premium": earned_premium,
        "n_claims": n_claims,
        "incurred": incurred,
        "paid": paid,
        "frequency_per_py": freq,
        "avg_severity": sev,
        "loss_cost": loss_cost,
        "loss_ratio": loss_ratio,
    }

# Returns a dataframe with  monthly claim counts and incured grouped by loss month
# Returns an empty dataframe with column names if no claims are provided for stability

def monthly_trend(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=["loss_month", "claims", "incurred"])
    out = claims.groupby("loss_month", as_index=False).agg(
        claims=("claim_id", "count"),
        incurred=("incurred", "sum"),
    )
    return out.sort_values("loss_month")