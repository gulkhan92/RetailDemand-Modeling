"""Statistical analysis helpers for Rossmann store-sales data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def promo_uplift_test(df: pd.DataFrame) -> dict[str, float]:
    """Compute promo uplift summary with Welch t-test and Mann-Whitney U."""
    promo_1 = df.loc[df["Promo"] == 1, "Sales"]
    promo_0 = df.loc[df["Promo"] == 0, "Sales"]

    welch = stats.ttest_ind(promo_1, promo_0, equal_var=False)
    mannw = stats.mannwhitneyu(promo_1, promo_0, alternative="two-sided")

    return {
        "mean_sales_promo_1": float(promo_1.mean()),
        "mean_sales_promo_0": float(promo_0.mean()),
        "mean_uplift": float(promo_1.mean() - promo_0.mean()),
        "welch_t_stat": float(welch.statistic),
        "welch_pvalue": float(welch.pvalue),
        "mannwhitney_u": float(mannw.statistic),
        "mannwhitney_pvalue": float(mannw.pvalue),
    }


def fit_robust_ols(df: pd.DataFrame):
    """Fit robust OLS on log-sales with calendar/store controls."""
    data = df.copy()
    if "log_sales" not in data.columns:
        data["log_sales"] = np.log1p(data["Sales"])

    formula = (
        "log_sales ~ Promo + SchoolHoliday + C(StateHoliday) + C(StoreType) + C(Assortment) + "
        "CompetitionDistance + Promo2 + C(DayOfWeek) + C(Month) + Year"
    )

    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    return model


def run_baseline_regression(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run baseline statistical analysis and return coefficients + tests."""
    model = fit_robust_ols(df)
    coef_df = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "p_value": model.pvalues.values,
        }
    ).sort_values("p_value")

    promo_stats = promo_uplift_test(df)
    return coef_df, promo_stats
