"""Evaluation helpers for the AI Maturity Index.

Provides validation utilities used by `notebooks/05_evaluation.ipynb`:
SHAP feature importance, rank correlation against external benchmarks
(HG Insights, IMD AI Maturity), and per-indicator descriptive stats.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats


def rank_correlation(
    df: pd.DataFrame,
    feature: str,
    target: str,
    method: str = "spearman",
) -> dict[str, float]:
    """Compute Spearman or Kendall rank correlation between a feature
    column and a target column, ignoring rows where either is NaN.
    """
    sub = df[[feature, target]].dropna()
    if len(sub) < 3:
        return {"correlation": float("nan"), "pvalue": float("nan"), "n": int(len(sub))}
    if method == "spearman":
        result = stats.spearmanr(sub[feature], sub[target])
    elif method == "kendall":
        result = stats.kendalltau(sub[feature], sub[target])
    else:
        raise ValueError(f"unknown method: {method}")
    return {
        "correlation": float(result.correlation),
        "pvalue": float(result.pvalue),
        "n": int(len(sub)),
    }


def indicator_descriptive_stats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return descriptive statistics for all columns belonging to one
    indicator (identified by the namespaced `<indicator>__` prefix).
    """
    cols = [c for c in df.columns if c.startswith(f"{prefix}__")]
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T
