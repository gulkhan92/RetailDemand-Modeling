"""End-to-end pipeline entrypoint for Rossmann Retail Demand Modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_preprocessing import build_feature_tables, load_raw_data, save_processed
from src.ml_models import build_feature_matrices, build_preprocessor, build_submission, train_models
from src.statistical_models import run_baseline_regression
from src.timeseries_models import build_daily_series, fit_sarima_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Rossmann end-to-end forecasting pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw dataset folder")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to output artifacts folder")
    parser.add_argument(
        "--disable-xgboost",
        action="store_true",
        help="Disable XGBoost training (enabled by default if dependency is available)",
    )
    return parser.parse_args()


def main() -> None:
    """Run full data-to-prediction workflow."""
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading raw data...")
    train_raw, test_raw, store_raw = load_raw_data(raw_dir)

    print("[2/6] Building feature tables...")
    train_feat, test_feat = build_feature_tables(train_raw, test_raw, store_raw)
    save_processed(train_feat, processed_dir / "train_features.csv")
    save_processed(test_feat, processed_dir / "test_features.csv")

    print("[3/6] Running statistical baseline...")
    coef_df, promo_stats = run_baseline_regression(train_feat)
    save_processed(coef_df, processed_dir / "statistical_coefficients.csv")
    with open(processed_dir / "statistical_promo_uplift.json", "w", encoding="utf-8") as f:
        json.dump(promo_stats, f, indent=2)

    print("[4/6] Running time-series baseline...")
    daily = build_daily_series(train_raw)
    ts_forecast_df, ts_metrics = fit_sarima_model(daily)
    save_processed(ts_forecast_df, processed_dir / "sarimax_daily_validation_forecast.csv")
    with open(processed_dir / "sarimax_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ts_metrics, f, indent=2)

    print("[5/6] Training ML models...")
    x_train, y_train, x_valid, y_valid, x_test, cutoff = build_feature_matrices(train_feat, test_feat)
    preprocessor = build_preprocessor(x_train)
    run = train_models(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        preprocessor=preprocessor,
        enable_xgboost=not args.disable_xgboost,
    )

    if run.leaderboard.empty:
        raise RuntimeError(
            "No ML models trained successfully. Inspect data/processed/model_errors.csv for details."
        )

    save_processed(run.leaderboard, processed_dir / "model_leaderboard.csv")
    if not run.errors.empty:
        save_processed(run.errors, processed_dir / "model_errors.csv")

    best_name = run.leaderboard.iloc[0]["model"]
    best_model = run.fitted_models[best_name]

    print(f"Best model: {best_name}")
    print("[6/6] Generating test predictions...")
    sub = build_submission(best_model, x_test, test_feat)
    save_processed(sub, processed_dir / "submission_baseline.csv")

    summary = {
        "validation_cutoff": str(cutoff.date()),
        "best_model": best_name,
        "best_rmspe": float(run.leaderboard.iloc[0]["RMSPE"]),
        "output_dir": str(processed_dir),
    }
    with open(processed_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Pipeline completed successfully.")
    print(f"Artifacts saved in: {processed_dir}")


if __name__ == "__main__":
    main()
