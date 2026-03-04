"""Time-series modeling helpers for aggregate Rossmann forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error with zero-safe masking."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask]))))


def chrono_split(df: pd.DataFrame, frac: float = 0.85) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Version-safe chronological split by row index."""
    ordered = df.sort_index().copy()
    n = len(ordered)
    split_idx = int(n * frac)
    split_idx = max(1, min(split_idx, n - 1))
    train_part = ordered.iloc[:split_idx].copy()
    valid_part = ordered.iloc[split_idx:].copy()
    split_dt = pd.Timestamp(train_part.index.max())
    return split_dt, train_part, valid_part


def build_daily_series(train_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level rows into a daily chain-level series."""
    base = train_df[(train_df["Open"] == 1) & (train_df["Sales"] > 0)].copy()
    daily = (
        base.groupby("Date", as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            PromoRate=("Promo", "mean"),
            SchoolHolidayRate=("SchoolHoliday", "mean"),
            AvgCustomers=("Customers", "mean"),
        )
        .sort_values("Date")
        .set_index("Date")
    )
    return daily


def fit_sarima_model(
    daily_df: pd.DataFrame,
    exog_cols: list[str] | None = None,
    split_frac: float = 0.85,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit SARIMAX baseline and return validation predictions with metrics."""
    exog_cols = exog_cols or ["PromoRate", "SchoolHolidayRate", "AvgCustomers"]

    split_date, train_ts, valid_ts = chrono_split(daily_df, frac=split_frac)

    for c in exog_cols:
        train_ts[c] = train_ts[c].fillna(0)
        valid_ts[c] = valid_ts[c].fillna(0)

    seasonal_order = (1, 1, 1, 7) if len(train_ts) >= 120 else (0, 0, 0, 0)

    model = SARIMAX(
        train_ts["Sales"],
        exog=train_ts[exog_cols],
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=len(valid_ts), exog=valid_ts[exog_cols])
    valid_ts = valid_ts.copy()
    valid_ts["Forecast"] = pred.predicted_mean

    rmse = float(np.sqrt(np.mean((valid_ts["Sales"] - valid_ts["Forecast"]) ** 2)))
    r = rmspe(valid_ts["Sales"].values, valid_ts["Forecast"].values)

    metrics = {
        "split_date": str(split_date.date()),
        "rmse": rmse,
        "rmspe": r,
        "seasonal_order": str(seasonal_order),
    }

    return valid_ts.reset_index(), metrics
