# RetailDemand-Modeling

Demand forecasting project based on the Kaggle Rossmann Store Sales competition.

## Objectives
- Build a reproducible pipeline for retail demand forecasting.
- Perform EDA, statistical analysis, time-series modeling, and ML modeling.
- Compare model families and report practical business insights.

## Dataset
- Source: [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/competitions/rossmann-store-sales/overview)
- Expected raw files in `data/raw/`:
  - `train.csv`
  - `test.csv`
  - `store.csv`
  - `sample_submission.csv`

Note: raw Kaggle files are ignored in git and should be downloaded locally.

## Project Structure
```
RetailDemand-Modeling/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── reports/
│   ├── figures/
│   └── analysis_summary.pdf
├── requirements.txt
└── environment.yml
```

## Quick Start
1. Create environment and install dependencies:
   - `pip install -r requirements.txt`
2. Download Kaggle files into `data/raw/`.
3. Run notebook: `notebooks/01_EDA_Rossmann.ipynb`.
