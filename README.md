# RetailDemand-Modeling

End-to-end retail demand forecasting project built on the Rossmann Store Sales Kaggle dataset.

## Overview
This repository provides a full forecasting workflow from raw data ingestion to model training and test-set prediction generation. It includes:
- Structured EDA and statistical analysis notebooks.
- Reusable Python modules for preprocessing, statistical baselines, time-series baselines, and ML training.
- A production-style `main.py` pipeline runner for reproducible execution.

Dataset source: [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/competitions/rossmann-store-sales/overview)

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

### Time-series baseline
- Daily aggregate SARIMAX with exogenous regressors (`PromoRate`, `SchoolHolidayRate`, `AvgCustomers`).
- Chronological validation split.

### ML baseline
- Feature-engineered tabular modeling.
- Preprocessing with median/mode imputation and one-hot encoding.
- Baseline models: Ridge, RandomForest (optional XGBoost).
- Selection metric: RMSPE.

## Notebook Workflow
1. `01_EDA_Rossmann.ipynb`: data quality, sales behavior, baseline figures.
2. `02_Statistical_Analysis.ipynb`: inference and diagnostics.
3. `03_Feature_Engineering.ipynb`: reusable feature generation.
4. `04_TimeSeries_Modeling.ipynb`: SARIMAX benchmark.
5. `05_ML_Models.ipynb`: model training and prediction comparison.

## Notes
- Raw Kaggle files are intentionally excluded from version control.
- The repository is designed to support iterative improvement: feature tuning, walk-forward CV, and model optimization can be layered on top of the current baseline pipeline.
