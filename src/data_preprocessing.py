"""Data ingestion and feature engineering utilities for Rossmann forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NUMERIC_ZERO_COLS = [
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2SinceWeek",
    "Promo2SinceYear",
]

CATEGORICAL_UNKNOWN_COLS = ["PromoInterval", "StateHoliday", "StoreType", "Assortment"]


MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


LAG_COLS = [
    "Sales_lag_1",
    "Sales_lag_7",
    "Sales_roll7_mean",
    "Sales_roll30_mean",
    "Customers_lag_1",
    "Customers_roll7_mean",
]


def load_raw_data(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and store files from raw directory."""
    raw_path = Path(raw_dir)
    train = pd.read_csv(raw_path / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(raw_path / "test.csv", parse_dates=["Date"])
    store = pd.read_csv(raw_path / "store.csv")
    return train, test, store


def merge_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge event-level rows with store metadata."""
    return df.merge(store_df, on="Store", how="left")


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar features from Date column."""
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfYear"] = out["Date"].dt.dayofyear
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["Quarter"] = out["Date"].dt.quarter
    out["IsMonthStart"] = out["Date"].dt.is_month_start.astype(int)
    out["IsMonthEnd"] = out["Date"].dt.is_month_end.astype(int)

    # Cyclical encoding for periodic calendar variables.
    out["Month_sin"] = np.sin(2 * np.pi * out["Month"] / 12)
    out["Month_cos"] = np.cos(2 * np.pi * out["Month"] / 12)
    out["Week_sin"] = np.sin(2 * np.pi * out["WeekOfYear"] / 52)
    out["Week_cos"] = np.cos(2 * np.pi * out["WeekOfYear"] / 52)
    return out


def fill_missing_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline imputation for sparse metadata fields."""
    out = df.copy()

    if "CompetitionDistance" in out.columns:
        out["CompetitionDistance"] = out["CompetitionDistance"].fillna(out["CompetitionDistance"].median())

    for col in NUMERIC_ZERO_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    for col in CATEGORICAL_UNKNOWN_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown")

    if "Open" in out.columns:
        out["Open"] = out["Open"].fillna(1)

    return out


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create competition age feature in months."""
    out = df.copy()

    if "CompetitionOpenSinceYear" not in out.columns or "CompetitionOpenSinceMonth" not in out.columns:
        return out

    comp_start = pd.to_datetime(
        {
            "year": out["CompetitionOpenSinceYear"].replace(0, out["Date"].dt.year.min()).astype(int),
            "month": out["CompetitionOpenSinceMonth"].replace(0, 1).astype(int),
            "day": 1,
        },
        errors="coerce",
    )

    out["CompetitionOpenMonths"] = (
        (out["Date"].dt.year - comp_start.dt.year) * 12 + (out["Date"].dt.month - comp_start.dt.month)
    )
    out["CompetitionOpenMonths"] = out["CompetitionOpenMonths"].clip(lower=0).fillna(0)
    return out


def add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create indicator whether current month is in store Promo2 interval."""
    out = df.copy()

    if "Promo2" not in out.columns:
        return out

    out["Promo2"] = out["Promo2"].fillna(0)
    out["PromoInterval"] = out["PromoInterval"].fillna("")
    month_name = out["Date"].dt.month.map(MONTH_MAP)

    out["IsPromo2Month"] = out.apply(
        lambda r: int(r["Promo2"] == 1 and isinstance(r["PromoInterval"], str) and month_name.loc[r.name] in r["PromoInterval"]),
        axis=1,
    )
    return out


def add_lag_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe lag and rolling features on train by Store."""
    out = train_df.sort_values(["Store", "Date"]).copy()
    g = out.groupby("Store", group_keys=False)

    out["Sales_lag_1"] = g["Sales"].shift(1)
    out["Sales_lag_7"] = g["Sales"].shift(7)
    out["Sales_roll7_mean"] = g["Sales"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    out["Sales_roll30_mean"] = g["Sales"].shift(1).rolling(30).mean().reset_index(level=0, drop=True)

    if "Customers" in out.columns:
        out["Customers_lag_1"] = g["Customers"].shift(1)
        out["Customers_roll7_mean"] = g["Customers"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)

    for col in LAG_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    return out


def clean_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where stores are closed or sales are zero."""
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()


def apply_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run full baseline feature-engineering stack on a merged dataframe."""
    out = df.copy()
    out = add_date_parts(out)
    out = fill_missing_baseline(out)
    out = add_competition_features(out)
    out = add_promo2_features(out)
    return out


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Ensure dataframe contains given columns, adding missing with NaN."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_feature_tables(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    store_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train/test modeling tables with aligned engineered features."""
    train_m = merge_store_data(train_raw, store_raw)
    test_m = merge_store_data(test_raw, store_raw)

    train_feat = apply_feature_pipeline(train_m)
    test_feat = apply_feature_pipeline(test_m)

    train_feat = add_lag_features(train_feat)
    train_feat = clean_train_rows(train_feat)
    train_feat["log_sales"] = np.log1p(train_feat["Sales"])

    # Align test schema for downstream modeling.
    test_feat = ensure_columns(test_feat, [c for c in train_feat.columns if c not in {"Sales", "Customers", "log_sales"}])

    return train_feat, test_feat


def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save processed dataframe to CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
