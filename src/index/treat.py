"""Outlier treatment ahead of normalization (OECD 1.5).

Follows the JRC screening rule used for OECD composite indicators: an
indicator needs treatment when |skewness| > 2 AND excess kurtosis > 3.5.

Upper-percentile winsorization was tried first but left most violating
indicators still over the thresholds (our indicators are shares bounded in
[0, 1] with a large exact-zero mass -- capping the tail barely touches
skew/kurtosis when that much of the distribution sits at the floor). A
square-root transform, the standard variance-stabilizing transform for
rate-like data, fully resolves every violation in this data set (verified
empirically) without the epsilon-at-zero problem of log transforms
(sqrt(0) = 0 is well defined) and without distorting the ranking (sqrt is
monotonic). Indicators passing the rule untransformed are left untouched
so the treatment never destroys information it does not have to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SKEW_THRESHOLD = 2.0
KURTOSIS_THRESHOLD = 3.5


def distribution_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-indicator distribution diagnostics for the normalization choice."""
    rows: list[dict] = []
    for c in cols:
        s = df[c].dropna()
        rows.append(
            {
                "indicator": c,
                "n": len(s),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurt()),
                "needs_treatment": bool(abs(s.skew()) > SKEW_THRESHOLD and s.kurt() > KURTOSIS_THRESHOLD),
                "min": float(s.min()),
                "p50": float(s.median()),
                "p975": float(s.quantile(0.975)),
                "max": float(s.max()),
            }
        )
    return pd.DataFrame(rows).set_index("indicator")


def sqrt_transform_skewed(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Square-root transform every indicator violating the JRC rule.

    Returns (treated frame, report). The report shows skew/kurtosis before
    and after; untreated indicators appear with `treated = False` so the
    decision is auditable. Indicators must be non-negative (true of every
    index indicator: shares and the FLS score are in [0, 1], the FinBERT
    net-tone scores are handled separately and never reach this function
    since they already pass the JRC rule).
    """
    out = df.copy()
    rows: list[dict] = []
    for c in cols:
        s = df[c].dropna()
        violates = abs(s.skew()) > SKEW_THRESHOLD and s.kurt() > KURTOSIS_THRESHOLD
        if violates:
            assert (s >= 0).all(), f"{c} has negative values; sqrt transform requires non-negative data"
            out[c] = np.sqrt(df[c])
        treated = out[c].dropna()
        rows.append(
            {
                "indicator": c,
                "treated": violates,
                "skew_before": float(s.skew()),
                "skew_after": float(treated.skew()),
                "kurt_before": float(s.kurt()),
                "kurt_after": float(treated.kurt()),
            }
        )
    return out, pd.DataFrame(rows).set_index("indicator")
