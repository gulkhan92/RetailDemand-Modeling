"""Machine-learning training and prediction utilities for Rossmann forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelRunResult:
    """Container for model training results."""

    leaderboard: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    errors: pd.DataFrame


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Percentage Error with zero-safe masking."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask]))))


def chrono_split(df: pd.DataFrame, date_col: str = "Date", frac: float = 0.85) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Chronological split without depending on datetime quantile support."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    n = len(ordered)
    split_idx = int(n * frac)
    split_idx = max(1, min(split_idx, n - 1))
    tr = ordered.iloc[:split_idx].copy()
    va = ordered.iloc[split_idx:].copy()
    cutoff = pd.Timestamp(tr[date_col].max())
    return cutoff, tr, va


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Timestamp]:
    """Build aligned X/y matrices for train, validation, and test."""
    exclude_cols = {"Sales", "Customers", "log_sales", "Id"}
    features = [c for c in train_df.columns if c not in exclude_cols]

    cutoff, tr, va = chrono_split(train_df, date_col="Date", frac=0.85)

    x_train = tr[features].copy()
    y_train = tr["Sales"].copy()
    x_valid = va[features].copy()
    y_valid = va["Sales"].copy()

    for df_ in (x_train, x_valid):
        df_["DateOrdinal"] = pd.to_datetime(df_["Date"], errors="coerce").map(pd.Timestamp.toordinal)
        df_.drop(columns=["Date"], inplace=True)

    x_test = test_df.copy()
    for col in features:
        if col not in x_test.columns:
            x_test[col] = np.nan
    x_test = x_test[features].copy()
    x_test["DateOrdinal"] = pd.to_datetime(x_test["Date"], errors="coerce").map(pd.Timestamp.toordinal)
    x_test.drop(columns=["Date"], inplace=True)

    # Convert categorical columns to string to handle mixed types (e.g., StateHoliday has 0 and 'a','b','c')
    cat_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        x_train[col] = x_train[col].astype(str)
        x_valid[col] = x_valid[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    return x_train, y_train, x_valid, y_valid, x_test, cutoff


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    cat_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in x_train.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    preprocessor: ColumnTransformer,
    enable_xgboost: bool = False,
) -> ModelRunResult:
    """Train baseline models and return leaderboard with failures."""
    models: dict[str, object] = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
    }

    if enable_xgboost:
        try:
            from xgboost import XGBRegressor

            models["XGBoost"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        except Exception as err:  # pragma: no cover - environment-dependent
            pass

    results: list[dict[str, float | str]] = []
    model_errors: list[dict[str, str]] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(x_train, y_train)
            pred = np.clip(pipe.predict(x_valid), a_min=0, a_max=None)

            results.append(
                {
                    "model": name,
                    "RMSE": float(np.sqrt(mean_squared_error(y_valid, pred))),
                    "MAE": float(mean_absolute_error(y_valid, pred)),
                    "RMSPE": float(rmspe(y_valid.values, pred)),
                }
            )
            fitted[name] = pipe
        except Exception as err:
            model_errors.append({"model": name, "error": str(err)})

    leaderboard = pd.DataFrame(results, columns=["model", "RMSE", "MAE", "RMSPE"])
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("RMSPE").reset_index(drop=True)

    errors_df = pd.DataFrame(model_errors)
    return ModelRunResult(leaderboard=leaderboard, fitted_models=fitted, errors=errors_df)


def build_submission(
    best_model: Pipeline,
    x_test: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate Kaggle-compatible submission dataframe."""
    preds = np.clip(best_model.predict(x_test), a_min=0, a_max=None)

    if "Id" in test_df.columns:
        return pd.DataFrame({"Id": test_df["Id"], "Sales": preds})

    return pd.DataFrame({"Sales": preds})
