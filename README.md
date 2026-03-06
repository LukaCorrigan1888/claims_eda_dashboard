# Motor Claims EDA Dashboard

A portfolio project that simulates a synthetic motor insurance portfolio and explores claims experience through an interactive Streamlit dashboard.

This project demonstrates practical actuarial and analytics skills, including:

- synthetic insurance data generation
- exploratory data analysis
- claims KPI development
- portfolio segmentation
- interactive dashboard development using Python and Streamlit

---

# Project Overview

This project contains two main components:

1. **Synthetic motor insurance data simulation**
   - policy-level data such as exposure, earned premium, vehicle value, cover type, state, and driver age band
   - claim-level data such as loss date, report date, close date, claim type, status, paid, case reserve, and incurred

2. **Interactive claims EDA dashboard**
   - filter claims by state, cover type, driver age band, claim type, status, and date range
   - view key portfolio KPIs
   - inspect monthly claim frequency and incurred trends
   - analyse claim type cost breakdowns
   - explore severity distributions
   - run basic data quality checks

The goal is to replicate the type of exploratory portfolio analysis often performed in insurance analytics or actuarial work.

---

# Synthetic Data Generation

The dataset used in this project is **fully synthetic** and was created using a custom Python simulation.

The simulation generates both **policy-level** and **claim-level** data designed to resemble a simplified motor insurance portfolio. It includes features such as:

- policy exposure and earned premium
- vehicle value and depreciation
- claim frequency influenced by rating factors
- claim types (collision, windscreen, theft, etc.)
- reporting delays and settlement delays
- heavy-tailed claim severities
- open and closed claims with case reserves

The purpose of the simulation is to produce data that behaves similarly to a real insurance dataset so that exploratory analysis and dashboarding techniques can be demonstrated.

**Disclosure**

The synthetic portfolio simulation and initial implementation of the data generation logic in `simulate.py` were created with assistance from **ChatGPT**.

---

# Project Structure


```text
claims-eda-dashboard/
  pyproject.toml
  README.md
  data/
    processed/
  notebooks/
    01_generate_and_eda.ipynb
  src/
    claims_dashboard/
      __init__.py
      simulate.py
      data.py
      metrics.py
      app.py
```
---

# Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/LukaCorrigan1888/claims-eda-dashboard.git
cd claims-eda-dashboard
uv sync
```

## Generate Data

Running this notebook first will generate the data

```text
notebooks/01_generate_and_eda.ipynb
```

## Run The Dashboard

```bash
uv run streamlit run src/claims_dashboard/app.py
```