"""Data loading, cleaning, and feature engineering helpers."""

from pathlib import Path
import pandas as pd


def load_raw_data(raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and store files from raw directory."""
    raw_path = Path(raw_dir)
    train = pd.read_csv(raw_path / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(raw_path / "test.csv", parse_dates=["Date"])
    store = pd.read_csv(raw_path / "store.csv")
    return train, test, store


def merge_store_data(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Merge event-level data with store metadata."""
    return df.merge(store_df, on="Store", how="left")


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Extract common date components for modeling."""
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["WeekOfYear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["IsMonthStart"] = out["Date"].dt.is_month_start.astype(int)
    out["IsMonthEnd"] = out["Date"].dt.is_month_end.astype(int)
    return out


def fill_missing_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple missing value strategy for first iteration."""
    out = df.copy()
    numeric_fill_zero = [
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    categorical_fill_unknown = ["PromoInterval", "StateHoliday", "StoreType", "Assortment"]

    for col in numeric_fill_zero:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    for col in categorical_fill_unknown:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown")

    return out


def clean_train_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where stores are closed or sales are zero."""
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()


def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save processed dataframe to CSV."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
