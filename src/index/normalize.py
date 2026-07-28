"""Within-Fortune-500 normalization for indicator features.

Two scaling options are exposed: z-score (mean 0, sd 1) and min-max
(rescaled to [0, 1]). Both ignore NaN. Boolean / 0-1 indicator flags
are detected by dtype and passed through unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_passthrough(s: pd.Series) -> bool:
    if s.dtype == bool:
        return True
    if s.dropna().isin([0, 1]).all() and s.name and str(s.name).endswith("_flag"):
        return True
    if s.name and str(s.name).split("__")[-1].startswith("has_"):
        return True
    return False


def zscore(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in cols:
        if _is_passthrough(out[c]):
            continue
        s = out[c].astype(float)
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if not sd or np.isnan(sd):
            out[c] = 0.0
        else:
            out[c] = (s - mu) / sd
    return out


def rankscore(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Percentile-rank normalization to [0, 1] (OECD "ranking" method).

    Immune to outliers and distribution shape, at the price of losing all
    level information; used as a sensitivity variant next to min-max.
    NaN stays NaN.
    """
    out = df.copy()
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in cols:
        if _is_passthrough(out[c]):
            continue
        s = out[c].astype(float)
        n = s.notna().sum()
        if n <= 1:
            out[c] = 0.0
        else:
            out[c] = (s.rank(method="average") - 1) / (n - 1)
    return out


def minmax(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in cols:
        if _is_passthrough(out[c]):
            continue
        s = out[c].astype(float)
        lo = s.min(skipna=True)
        hi = s.max(skipna=True)
        if hi == lo or np.isnan(hi - lo):
            out[c] = 0.0
        else:
            out[c] = (s - lo) / (hi - lo)
    return out
