"""
data_preprocessing.py
---------------------
Step 1 of the pipeline: clean the raw housing data, engineer features and
create the two target variables.

    python src/data_preprocessing.py

Outputs
    data/processed/cleaned_housing_data.csv   -> cleaned + engineered dataset
    data/processed/city_reference_stats.csv   -> city medians used by the app

The feature-engineering functions in this file are also imported by the
Streamlit app, so a single property typed into the app goes through exactly
the same transformations as the training data.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("preprocessing")

TRANSPORT_POINTS = {"Low": 0.0, "Medium": 1.5, "High": 3.0}


# ----------------------------------------------------------------------------
# 1. Loading & cleaning
# ----------------------------------------------------------------------------
def load_raw_data(path: Path = cfg.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and fail with a clear message if it is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Place india_housing_prices.csv inside data/raw/."
        )
    df = pd.read_csv(path)
    log.info("Loaded raw data: %s rows x %s columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates, missing values, data-entry inconsistencies and outliers."""
    df = df.copy()

    # --- Duplicates ----------------------------------------------------------
    n_dup = int(df.duplicated().sum())
    n_dup_id = int(df.duplicated(subset="ID").sum())
    df = df.drop_duplicates().drop_duplicates(subset="ID")
    log.info("Duplicates removed: %s full-row, %s repeated IDs", n_dup, n_dup_id)

    # --- Missing values (median for numeric, mode for categorical) -----------
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    n_missing = int(df.isna().sum().sum())
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    log.info("Missing values imputed: %s", n_missing)

    # --- Text hygiene ---------------------------------------------------------
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # --- Logical inconsistency: floor number above total floors --------------
    # In ~46 % of rows Floor_No > Total_Floors, which is impossible. We treat
    # it as the two values being entered in swapped columns and swap them back.
    bad_floor = df["Floor_No"] > df["Total_Floors"]
    log.info("Rows with Floor_No > Total_Floors fixed by swapping: %s", int(bad_floor.sum()))
    df.loc[bad_floor, ["Floor_No", "Total_Floors"]] = (
        df.loc[bad_floor, ["Total_Floors", "Floor_No"]].to_numpy()
    )

    # --- Recompute derived columns precisely ---------------------------------
    # The raw Price_per_SqFt is in lakhs and rounded to 2 decimals (0.1 = ₹10k),
    # which loses most of its precision. We rebuild it in rupees.
    df["Price_per_SqFt_Rs"] = (df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]).round(2)
    df["Age_of_Property"] = cfg.REFERENCE_YEAR - df["Year_Built"]

    # --- Outliers (IQR rule) -> cap (winsorise) instead of dropping ----------
    for col in ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_Rs"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(lower=max(low, df[col].min()), upper=high)
        log.info("Outliers capped in %-18s: %s", col, n_out)

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Feature engineering (shared with the Streamlit app)
# ----------------------------------------------------------------------------
def city_tier(city: str) -> str:
    """Map a city to Tier 1 / 2 / 3."""
    if city in cfg.TIER_1_CITIES:
        return "Tier 1"
    if city in cfg.TIER_2_CITIES:
        return "Tier 2"
    return "Tier 3"


def build_city_reference_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median price & price-per-sqft per city (the 'market benchmark')."""
    stats = (
        df.groupby("City")
        .agg(
            State=("State", "first"),
            City_Median_PPSF=("Price_per_SqFt_Rs", "median"),
            City_Median_Price=("Price_in_Lakhs", "median"),
            Listings=("ID", "count"),
        )
        .reset_index()
    )
    return stats


def engineer_features(df: pd.DataFrame, city_stats: pd.DataFrame) -> pd.DataFrame:
    """Create all model features. Works for 250k rows or a single app input."""
    df = df.copy()

    # Amenities: count + one flag per amenity
    amen = df["Amenities"].fillna("").astype(str)
    for a in cfg.AMENITY_LIST:
        df[f"Has_{a}"] = amen.str.contains(a, regex=False).astype(int)
    df["Amenities_Count"] = df[[f"Has_{a}" for a in cfg.AMENITY_LIST]].sum(axis=1)

    # Yes/No and status columns -> binary flags
    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    df["Ready_To_Move_Flag"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)

    # Location tier
    df["City_Tier"] = df["City"].map(city_tier)

    # School Density Score: schools nearby rescaled to 0-10
    df["School_Density_Score"] = ((df["Nearby_Schools"] - 1) / 9 * 10).clip(0, 10).round(2)

    # Infrastructure Score (0-10): schools 35 %, hospitals 35 %, transport 30 %
    df["Infrastructure_Score"] = (
        df["Nearby_Schools"] / 10 * 3.5
        + df["Nearby_Hospitals"] / 10 * 3.5
        + df["Public_Transport_Accessibility"].map(TRANSPORT_POINTS).fillna(0)
    ).round(2)

    # Relative-to-market features (how cheap/expensive vs the city benchmark)
    ref = city_stats.set_index("City")
    city_ppsf = df["City"].map(ref["City_Median_PPSF"]).fillna(city_stats["City_Median_PPSF"].median())
    city_price = df["City"].map(ref["City_Median_Price"]).fillna(city_stats["City_Median_Price"].median())
    df["PPSF_vs_City_Median"] = (df["Price_per_SqFt_Rs"] / city_ppsf).round(4)
    df["Price_vs_City_Median"] = (df["Price_in_Lakhs"] / city_price).round(4)

    return df


# ----------------------------------------------------------------------------
# 3. Target creation
# ----------------------------------------------------------------------------
def expected_growth_rate(df: pd.DataFrame) -> pd.Series:
    """
    Annual appreciation rate per property (domain rule, before market noise).

    Base 8 % p.a. (project brief) adjusted for location tier, connectivity,
    infrastructure, construction stage, property type, age and amenities.
    """
    r = pd.Series(cfg.BASE_GROWTH_RATE, index=df.index, dtype=float)
    r += df["City_Tier"].map({"Tier 1": 0.010, "Tier 2": 0.0, "Tier 3": -0.010})
    r += df["Public_Transport_Accessibility"].map({"High": 0.005, "Medium": 0.0, "Low": -0.005}).fillna(0)
    r += (df["Infrastructure_Score"] - 5) * 0.001          # ±0.5 %
    r += np.where(df["Ready_To_Move_Flag"] == 0, 0.010, 0)  # under-construction premium
    r += df["Property_Type"].map({"Villa": 0.005, "Independent House": 0.0025, "Apartment": 0.0}).fillna(0)
    r -= df["Age_of_Property"] * 0.0003                     # older -> slower (max ~-1 %)
    r += df["Amenities_Count"] * 0.001                      # up to +0.5 %
    return r


def create_targets(df: pd.DataFrame, seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Add Expected_Growth_Rate, Future_Price_5Y_Lakhs and Good_Investment."""
    df = df.copy()
    rng = np.random.default_rng(seed)

    # --- Regression target: price after 5 years ------------------------------
    # Future = Current * (1 + r)^t, where r = rule-based rate + market noise
    base_rate = expected_growth_rate(df)
    market_noise = rng.normal(0, 0.005, len(df))            # ±0.5 % market uncertainty
    df["Expected_Growth_Rate"] = (base_rate + market_noise).clip(0.03, 0.15).round(4)
    df[cfg.REGRESSION_TARGET] = (
        df["Price_in_Lakhs"] * (1 + df["Expected_Growth_Rate"]) ** cfg.FORECAST_YEARS
    ).round(2)

    # --- Classification target: Good Investment (multi-factor score) --------
    score = (
        2.0 * (df["PPSF_vs_City_Median"] <= 1.0)            # cheaper per sqft than city median
        + 1.0 * (df["Price_vs_City_Median"] <= 1.0)         # affordable vs city
        + 1.0 * (df["BHK"] >= 3)                            # family-size demand
        + 1.0 * (df["Ready_To_Move_Flag"] == 1)             # no construction risk
        + df["Public_Transport_Accessibility"].map({"High": 1.0, "Medium": 0.5, "Low": 0.0}).fillna(0)
        + 1.0 * (df["Infrastructure_Score"] >= 5)           # good social infra
        + 1.0 * (df["Amenities_Count"] >= 3)                # lifestyle amenities
        + 0.5 * (df["Security_Flag"] == 1)
        + 0.5 * (df["Parking_Flag"] == 1)
        + 1.0 * (base_rate >= cfg.BASE_GROWTH_RATE)         # appreciation rate > threshold
    )
    df["Investment_Score"] = score.round(2)                  # 0 - 10
    df[cfg.CLASSIFICATION_TARGET] = (df["Investment_Score"] >= 5.5).astype(int)
    return df


# ----------------------------------------------------------------------------
# 4. Orchestration
# ----------------------------------------------------------------------------
def run_preprocessing() -> pd.DataFrame:
    raw = load_raw_data()
    clean = clean_data(raw)
    city_stats = build_city_reference_stats(clean)
    feat = engineer_features(clean, city_stats)
    final = create_targets(feat)

    cfg.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(cfg.PROCESSED_DATA_PATH, index=False)
    city_stats.to_csv(cfg.CITY_STATS_PATH, index=False)

    log.info("Saved cleaned data -> %s  (%s rows x %s cols)",
             cfg.PROCESSED_DATA_PATH.relative_to(cfg.ROOT_DIR), *final.shape)
    log.info("Good_Investment share: %.1f %%", final[cfg.CLASSIFICATION_TARGET].mean() * 100)
    log.info("Mean expected growth rate: %.2f %%", final["Expected_Growth_Rate"].mean() * 100)
    return final


if __name__ == "__main__":
    try:
        run_preprocessing()
    except Exception as exc:  # clear message instead of a long traceback
        log.error("Preprocessing failed: %s", exc)
        raise
