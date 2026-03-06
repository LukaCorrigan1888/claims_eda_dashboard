from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from claims_dashboard.data import load_processed
from claims_dashboard.metrics import add_derived_fields, kpis, monthly_trend

# Streamlit dashboard

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="Motor Claims EDA", layout="wide")
st.title("Motor Claims EDA Dashboard (Synthetic Portfolio)")


# -------------------------------
# Load data (cached)
# -------------------------------
@st.cache_data
def load_data():
    policies, claims = load_processed()
    claims = add_derived_fields(claims)
    return policies, claims


policies, claims = load_data()


# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")

# Set defualt range to choose from
 
min_date = claims["loss_date"].min()
max_date = claims["loss_date"].max()

# Add sidebar to select date range

date_range = st.sidebar.date_input(
    "Loss Date Range",
    value=(min_date.date(), max_date.date()),
)

# Get unique values for categorical filters from policies and claims dataframes

states = sorted(policies["state"].unique())
covers = sorted(policies["cover_type"].unique())
bands = sorted(policies["driver_age_band"].unique())
types = sorted(claims["claim_type"].unique())
statuses = sorted(claims["status"].unique())

# Add filters for categories in dataframes with default to select all

state_sel = st.sidebar.multiselect("State", states, default=states)
cover_sel = st.sidebar.multiselect("Cover Type", covers, default=covers)
band_sel = st.sidebar.multiselect("Driver Age Band", bands, default=bands)
type_sel = st.sidebar.multiselect("Claim Type", types, default=types)
status_sel = st.sidebar.multiselect("Status", statuses, default=statuses)


# -------------------------------
# Merge policy dimensions onto claims
# -------------------------------
pol_dim = policies[
    ["policy_id", "state", "cover_type", "driver_age_band",
     "exposure_years", "earned_premium"]
]

df = claims.merge(pol_dim, on="policy_id", how="left")


# -------------------------------
# Apply filters
# -------------------------------

# Applys filters to be used in KPIs and charts. 
# Applys the input to start and end and filters for state, cover type, driver age band, claim type and status.

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

df = df[
    (df["loss_date"] >= start)
    & (df["loss_date"] <= end)
    & (df["state"].isin(state_sel))
    & (df["cover_type"].isin(cover_sel))
    & (df["driver_age_band"].isin(band_sel))
    & (df["claim_type"].isin(type_sel))
    & (df["status"].isin(status_sel))
]

pol_filt = policies[
    (policies["state"].isin(state_sel))
    & (policies["cover_type"].isin(cover_sel))
    & (policies["driver_age_band"].isin(band_sel))
]


# -------------------------------
# KPIs
# -------------------------------

# Calculates kpis using the filtered claims and policies dataframes and displays them 

k = kpis(pol_filt, df)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Claims", f"{k['n_claims']:,}")
c2.metric("Exposure (PY)", f"{k['exposure_years']:.1f}")
c3.metric("Earned Premium", f"${k['earned_premium']:,.0f}")
c4.metric("Incurred", f"${k['incurred']:,.0f}")
c5.metric("Frequency (/PY)", f"{k['frequency_per_py']:.3f}")
c6.metric("Avg Severity", f"${k['avg_severity']:,.0f}")

st.caption(f"Loss Ratio: {k['loss_ratio']:.3f}")


st.divider()


# -------------------------------
# Monthly Trends
# -------------------------------
trend = monthly_trend(df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Claims Count by Loss Month")
    fig = px.line(trend, x="loss_month", y="claims", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Incurred by Loss Month")
    fig = px.line(trend, x="loss_month", y="incurred", markers=True)
    st.plotly_chart(fig, use_container_width=True)


st.divider()


# -------------------------------
# Claim Type Breakdown
# -------------------------------
st.subheader("Incurred by Claim Type")

by_type = df.groupby("claim_type", as_index=False)["incurred"].sum()
fig = px.bar(by_type, x="claim_type", y="incurred")
st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Severity Distribution
# -------------------------------
st.subheader("Severity Distribution")

fig = px.histogram(df, x="incurred", nbins=60)
st.plotly_chart(fig, use_container_width=True)