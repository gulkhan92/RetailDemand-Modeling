# RetailDemand-Modeling

End-to-end retail demand forecasting project built on the Rossmann Store Sales Kaggle dataset.

## Overview
This repository provides a full forecasting workflow from raw data ingestion to model training and test-set prediction generation. It includes:
- Structured EDA and statistical analysis notebooks.
- Reusable Python modules for preprocessing, statistical baselines, time-series baselines, and ML training.
- A production-style `main.py` pipeline runner for reproducible execution.

Dataset source: [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

## Dataset Summary
- **Training data:** ~844K sales records from 1,115 Rossmann stores (Germany)
- **Test data:** ~41K records to predict
- **Features:** Store ID, Day of Week, Promotions, State Holidays, School Holidays, Store Type, Assortment, Competition Distance, and temporal features
- **Target:** Sales (number of transactions)

### Key Statistical Findings
- **Promo Uplift:** Sales with promo = 8,229 vs without promo = 5,930 (uplift of ~2,299, p < 0.001)
- **Store Type Impact:** Store Type B has highest positive coefficient (+0.55), Type D has lowest (+0.007)
- **Seasonality:** December shows highest positive seasonality effect (+0.26), while Sundays show significant negative impact (-0.25)
- **Competition:** Higher competition distance has slight negative effect on sales

## Project Structure
```
RetailDemand-Modeling/
├── README.md
├── main.py
├── data/
│   ├── raw/                 # Kaggle raw files (local only, git-ignored)
│   └── processed/           # Pipeline outputs
├── notebooks/
│   ├── 01_EDA_Rossmann.ipynb
│   ├── 02_Statistical_Analysis.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_TimeSeries_Modeling.ipynb
│   └── 05_ML_Models.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── statistical_models.py
│   ├── timeseries_models.py
│   └── ml_models.py
├── reports/
│   ├── figures/
│   └── analysis_summary.pdf
├── requirements.txt
└── environment.yml
```

## Setup
### 1. Create environment
Using pip:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or using conda:
```bash
conda env create -f environment.yml
conda activate retail-demand-modeling
```

### 2. Add raw dataset files
Place these files in `data/raw/`:
- `train.csv`
- `test.csv`
- `store.csv`
- `sample_submission.csv`

## Run End-to-End Pipeline
Run baseline pipeline:
```bash
python3 main.py
```

Optional (enable XGBoost if installed):
```bash
python3 main.py --enable-xgboost
```

Custom data locations:
```bash
python3 main.py --raw-dir data/raw --processed-dir data/processed
```

### main.py Pipeline Overview
The `main.py` script orchestrates the complete end-to-end machine learning pipeline with 6 sequential stages:

1. **Loading Raw Data** (`load_raw_data`)
   - Loads train.csv, test.csv, and store.csv from the raw data directory
   - Handles date parsing and data type conversions

2. **Building Feature Tables** (`build_feature_tables`)
   - Creates temporal features (Year, Month, Day, Week, Quarter, cyclical encodings)
   - Adds lag features (Sales_lag_1, Sales_lag_7)
   - Computes rolling statistics (Sales_roll7_mean, Sales_roll30_mean)
   - Handles competition and promo2 features

3. **Statistical Baseline** (`run_baseline_regression`)
   - Robust OLS regression with HC3 heteroskedasticity-consistent standard errors
   - Analyzes promo uplift effect using Welch t-test and Mann-Whitney U test
   - Outputs coefficient table and statistical findings

4. **Time-Series Baseline** (`fit_sarima_model`)
   - Aggregates data to daily level
   - Fits SARIMAX(1,1,1)(1,1,1,7) model with exogenous regressors
   - Generates validation forecasts and calculates metrics

5. **ML Model Training** (`train_models`)
   - Splits data chronologically (85% train, 15% validation)
   - Preprocesses features with median/mode imputation and one-hot encoding
   - Trains Ridge and RandomForest regressors
   - Selects best model based on RMSPE metric

6. **Generating Predictions**
   - Uses best model to generate test set predictions
   - Creates Kaggle-compatible submission file

## Pipeline Outputs (`data/processed/`)
- `train_features.csv`: engineered training table.
- `test_features.csv`: engineered test table aligned to training schema.
- `statistical_coefficients.csv`: robust OLS coefficient table.
- `statistical_promo_uplift.json`: promo uplift hypothesis-test summary.
- `sarimax_daily_validation_forecast.csv`: time-series validation forecast.
- `sarimax_metrics.json`: SARIMAX validation metrics.
- `model_leaderboard.csv`: ML model comparison (RMSE, MAE, RMSPE).
- `model_errors.csv`: model failures if any model could not train.
- `submission_baseline.csv`: final test-set predictions.
- `pipeline_summary.json`: concise run summary.

## Modeling Approach
### Statistical baseline
- Robust OLS (`HC3`) on `log1p(Sales)` with operational and calendar controls.
- Promo effect testing via Welch t-test and Mann-Whitney U.
- **Key findings:** Promotions have significant positive effect (+33% coefficient), with mean uplift of 2,299 in sales.

### Time-series baseline
- Daily aggregate SARIMAX with exogenous regressors (`PromoRate`, `SchoolHolidayRate`, `AvgCustomers`).
- Chronological validation split (training until 2015-03-11).
- **Metrics:** RMSE = 2,724,711, RMSPE = 8.25

### ML baseline
- Feature-engineered tabular modeling with 36 features including lag features and rolling statistics.
- Preprocessing with median/mode imputation and one-hot encoding.
- Baseline models: Ridge, RandomForest (optional XGBoost).
- Selection metric: RMSPE.

### Model Results (Validation)
| Model | RMSE | MAE | RMSPE |
|-------|------|-----|-------|
| **RandomForest** | 1,120.68 | 759.91 | **0.156** |
| Ridge | 2,115.48 | 1,435.94 | 0.272 |

**Best Model:** RandomForest with RMSPE of 0.156

## Notebook Workflow
1. `01_EDA_Rossmann.ipynb`: data quality, sales behavior, baseline figures.
2. `02_Statistical_Analysis.ipynb`: inference and diagnostics.
3. `03_Feature_Engineering.ipynb`: reusable feature generation.
4. `04_TimeSeries_Modeling.ipynb`: SARIMAX benchmark.
5. `05_ML_Models.ipynb`: model training and prediction comparison.

## Notes
- Raw Kaggle files are intentionally excluded from version control.
- The repository is designed to support iterative improvement: feature tuning, walk-forward CV, and model optimization can be layered on top of the current baseline pipeline.
