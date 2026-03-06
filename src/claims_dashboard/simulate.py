from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# Simulation of synthetic data created using chat gpt


@dataclass(frozen=True)
class MotorSimConfig:
    n_policies: int = 60_000
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    seed: int = 42

    base_lambda: float = 0.10          # expected claims per policy-year (before multipliers)
    annual_inflation: float = 0.06
    open_claim_prob: float = 0.07


def _days_between(a: np.datetime64, b: np.datetime64) -> int:
    return int((b - a).astype("timedelta64[D]") / np.timedelta64(1, "D"))


def simulate_motor(cfg: MotorSimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    start = np.datetime64(cfg.start_date)
    end = np.datetime64(cfg.end_date)
    horizon_days = _days_between(start, end)

    # --- Policy attributes (Motor-only) ---
    states = np.array(["VIC", "NSW", "QLD", "WA", "SA", "TAS", "ACT", "NT"])
    channels = np.array(["Online", "Broker", "Agency"])
    cover = np.array(["Comprehensive", "TPFT", "TPP"])  # third party property only

    state = rng.choice(states, size=cfg.n_policies, p=[0.27, 0.33, 0.20, 0.10, 0.06, 0.02, 0.015, 0.005])
    channel = rng.choice(channels, size=cfg.n_policies, p=[0.50, 0.40, 0.10])
    cover_type = rng.choice(cover, size=cfg.n_policies, p=[0.68, 0.18, 0.14])

    driver_band = rng.choice(
        np.array(["<25", "25-34", "35-49", "50-64", "65+"]),
        size=cfg.n_policies,
        p=[0.12, 0.22, 0.30, 0.26, 0.10],
    )

    vehicle_age = rng.integers(0, 21, size=cfg.n_policies)  # 0..20
    base_value = 45_000 * np.exp(-vehicle_age / 10.0)       # depreciation curve
    vehicle_value = base_value * rng.lognormal(mean=0.0, sigma=0.35, size=cfg.n_policies)
    vehicle_value = np.clip(vehicle_value, 2_000, 120_000)

    # inception/expiry and exposure
    inception_offset = rng.integers(0, horizon_days, size=cfg.n_policies)
    inception_date = start + inception_offset.astype("timedelta64[D]")

    term_days = rng.integers(200, 366, size=cfg.n_policies)
    expiry_date = np.minimum(inception_date + term_days.astype("timedelta64[D]"), end)

    exposure_days = (expiry_date - inception_date).astype("timedelta64[D]") / np.timedelta64(1, "D")
    exposure_years = np.clip(exposure_days / 365.25, 0.01, None)

    # premium (rough rating)
    cover_load = np.where(cover_type == "Comprehensive", 1.20, np.where(cover_type == "TPFT", 1.05, 0.90))
    band_load = np.where(driver_band == "<25", 1.60,
                 np.where(driver_band == "25-34", 1.20,
                 np.where(driver_band == "35-49", 1.00,
                 np.where(driver_band == "50-64", 0.90, 1.05))))
    channel_load = np.where(channel == "Broker", 1.10, np.where(channel == "Agency", 1.04, 1.00))
    state_load = np.where(state == "NSW", 1.05, np.where(state == "VIC", 1.00, 0.98))

    annual_premium = (
        350
        + 0.018 * vehicle_value * cover_load * band_load * channel_load * state_load
        * rng.lognormal(0.0, 0.12, cfg.n_policies)
    )
    earned_premium = annual_premium * exposure_years

    policies = pd.DataFrame({
        "policy_id": np.arange(cfg.n_policies, dtype=int),
        "product": "Motor",
        "state": state,
        "channel": channel,
        "cover_type": cover_type,
        "driver_age_band": driver_band,
        "vehicle_age": vehicle_age,
        "vehicle_value": vehicle_value,
        "inception_date": inception_date.astype("datetime64[D]"),
        "expiry_date": expiry_date.astype("datetime64[D]"),
        "exposure_years": exposure_years,
        "earned_premium": earned_premium,
    })

    # --- Frequency model (Poisson) ---
    band_mult = pd.Series({"<25": 1.45, "25-34": 1.15, "35-49": 1.00, "50-64": 0.90, "65+": 1.05})
    cover_mult = pd.Series({"Comprehensive": 1.10, "TPFT": 0.95, "TPP": 0.85})
    channel_mult = pd.Series({"Online": 0.95, "Broker": 1.05, "Agency": 1.00})
    state_mult = pd.Series({"VIC": 1.00, "NSW": 1.05, "QLD": 1.02, "WA": 0.95, "SA": 0.92, "TAS": 0.90, "ACT": 0.88, "NT": 0.85})

    lam = (
        cfg.base_lambda
        * policies["exposure_years"].to_numpy()
        * policies["driver_age_band"].map(band_mult).to_numpy()
        * policies["cover_type"].map(cover_mult).to_numpy()
        * policies["channel"].map(channel_mult).to_numpy()
        * policies["state"].map(state_mult).to_numpy()
    )

    n_claims = rng.poisson(lam)
    total_claims = int(n_claims.sum())
    if total_claims == 0:
        return policies, pd.DataFrame()

    policy_ids_rep = np.repeat(policies["policy_id"].to_numpy(), n_claims)

    # coverage windows for each claim
    pol_idx = policies.set_index("policy_id")[["inception_date", "expiry_date", "vehicle_value", "cover_type"]]
    inc_rep = pol_idx.loc[policy_ids_rep, "inception_date"].to_numpy(dtype="datetime64[D]")
    exp_rep = pol_idx.loc[policy_ids_rep, "expiry_date"].to_numpy(dtype="datetime64[D]")
    val_rep = pol_idx.loc[policy_ids_rep, "vehicle_value"].to_numpy(dtype=float)
    cov_rep = pol_idx.loc[policy_ids_rep, "cover_type"].to_numpy(dtype=object)

    cover_days = ((exp_rep - inc_rep).astype("timedelta64[D]") / np.timedelta64(1, "D")).astype(int)
    cover_days = np.clip(cover_days, 1, None)
    loss_date = inc_rep + rng.integers(0, cover_days, size=total_claims).astype("timedelta64[D]")

    # motor claim types
    claim_types = np.array(["Collision", "Theft", "Windscreen", "TPPD", "Hail"])
    claim_type = rng.choice(claim_types, size=total_claims, p=[0.58, 0.10, 0.22, 0.08, 0.02])

    # reporting delay
    rep_delay = rng.lognormal(mean=1.4, sigma=0.85, size=total_claims).astype(int)
    rep_delay = np.clip(rep_delay, 0, 365)
    report_date = loss_date + rep_delay.astype("timedelta64[D]")

    # settlement delay
    close_delay = rng.gamma(shape=2.2, scale=18.0, size=total_claims).astype(int)
    close_delay = np.clip(close_delay, 1, 540)
    close_date = report_date + close_delay.astype("timedelta64[D]")

    # open claims
    open_flag = rng.random(total_claims) < cfg.open_claim_prob
    close_date = np.where(open_flag, np.datetime64("NaT"), close_date)
    status = np.where(open_flag, "Open", "Closed")

    # --- Severity model ---
    years_from_start = ((loss_date - start).astype("timedelta64[D]") / np.timedelta64(1, "D")) / 365.25
    infl = (1.0 + cfg.annual_inflation) ** years_from_start

    base = np.full(total_claims, 2500.0)
    base[claim_type == "Windscreen"] = 650.0
    base[claim_type == "Collision"] = 4200.0
    base[claim_type == "Theft"] = 9000.0
    base[claim_type == "TPPD"] = 6000.0
    base[claim_type == "Hail"] = 5500.0

    incurred = base * rng.lognormal(mean=0.0, sigma=0.95, size=total_claims) * infl

    cap = np.where(claim_type == "Theft", val_rep * 1.05, val_rep * 0.7)
    cap = np.where(claim_type == "TPPD", np.maximum(cap, val_rep * 1.2), cap)
    incurred = np.minimum(incurred, cap)

    # cover-type effect: if TPP, only allow TPPD/Collision
    is_tpp = cov_rep == "TPP"
    if is_tpp.any():
        idx = np.where(is_tpp)[0]
        claim_type[idx] = rng.choice(np.array(["TPPD", "Collision"]), size=len(idx), p=[0.65, 0.35])
        base2 = np.where(claim_type[idx] == "TPPD", 6000.0, 3200.0)
        incurred[idx] = base2 * rng.lognormal(mean=0.0, sigma=0.9, size=len(idx)) * infl[idx]
        incurred[idx] = np.minimum(incurred[idx], np.maximum(val_rep[idx] * 1.2, 15_000))

    # paid vs reserve
    paid_prop = rng.beta(a=3.2, b=2.0, size=total_claims)
    paid = incurred * paid_prop
    case_reserve = incurred - paid

    paid[open_flag] *= 0.30
    case_reserve[open_flag] = incurred[open_flag] - paid[open_flag]

    claims = pd.DataFrame({
        "claim_id": np.arange(total_claims, dtype=int),
        "policy_id": policy_ids_rep.astype(int),
        "loss_date": loss_date.astype("datetime64[D]"),
        "report_date": report_date.astype("datetime64[D]"),
        "close_date": pd.to_datetime(close_date),
        "claim_type": claim_type,
        "status": status,
        "paid": paid,
        "case_reserve": case_reserve,
        "incurred": incurred,
    })

    return policies, claims


def write_processed(policies: pd.DataFrame, claims: pd.DataFrame, out_dir: str = "data/processed") -> None:
    import pathlib
    path = pathlib.Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    policies.to_parquet(path / "policies.parquet", index=False)
    claims.to_parquet(path / "claims.parquet", index=False)