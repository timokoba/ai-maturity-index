"""Combined input table and the Technology / People firm-level indicators.

`build_inputs` produces one wide row per universe firm holding every raw
input, the match provenance for both sources, the four computed shares, and
per-dimension completeness flags. It is the single source of truth for the
dimension notebooks and the HTML review.

Missing data is marked, never dropped or conflated with zero: a share is
NaN wherever an input is missing or a denominator is zero (a firm with no
publications has no defined AI-publication share), while a firm that
publishes but never on AI carries a genuine 0. The indicator frames keep
one row per universe firm, NaN and all; how to treat missing values is
decided at index-composition time.

Each share ships with a companion `<feature>_reason` column so the two ways
a share can be missing are not conflated: "unmatched" (the firm has no
record in the source at all -- no PARAT company, or for the People shares
no Compustat gvkey) versus "zero_denominator" (the firm is matched, but its
total publications/patents/employees is zero or unavailable). The reason is
NaN whenever the share itself is defined.

The indicator frames written to `data_clean/indicators/<universe>/` carry
only the id columns plus the ratio features and their reason columns:
`compose_index` treats every non-id column as a feature, so raw counts and
match provenance stay in the cache and the review page. The reason columns
are strings, so `src/index/normalize.py`'s z-score/min-max (which only
touch numeric columns) pass them through untouched.
"""

from __future__ import annotations

import pandas as pd

from .eto import load_eto_core

INDICATOR = "structured_features"

TECHNOLOGY_FEATURES = ["ai_publication_share", "ai_patent_share"]
PEOPLE_FEATURES = ["tech_team1_worker_share", "ai_worker_share"]
TECHNOLOGY_REASON_COLS = [f"{c}_reason" for c in TECHNOLOGY_FEATURES]
PEOPLE_REASON_COLS = [f"{c}_reason" for c in PEOPLE_FEATURES]
ID_COLS = ["normalized_company_name", "ticker", "company_name"]

REASON_UNMATCHED = "unmatched"
REASON_ZERO_DENOMINATOR = "zero_denominator"


def _share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator with non-positive denominators treated as
    missing (a firm with zero total patents has no defined AI patent share).
    """
    return numerator / denominator.where(denominator > 0)


def _missing_reason(matched: pd.Series, denominator: pd.Series) -> pd.Series:
    """Why a share is NaN: `REASON_UNMATCHED` when the firm has no record in
    the source at all, `REASON_ZERO_DENOMINATOR` when it does but the
    denominator is zero or unavailable. NaN (`pd.NA`) when the share is
    defined -- matched with a positive denominator.
    """
    reason = pd.Series(pd.NA, index=matched.index, dtype="object")
    reason[~matched] = REASON_UNMATCHED
    bad_denominator = matched & (denominator.isna() | (denominator <= 0))
    reason[bad_denominator] = REASON_ZERO_DENOMINATOR
    return reason


def build_inputs(
    universe_df: pd.DataFrame,
    eto_matched: pd.DataFrame,
    wrds_matched: pd.DataFrame,
) -> pd.DataFrame:
    """One row per universe firm: ids, match provenance, raw ETO and
    Compustat inputs, the four computed shares (NaN wherever an input is
    missing or a denominator is zero), and the per-dimension completeness
    flags that implement the drop rule.
    """
    base = universe_df[["rank", "company", "ticker", "normalized_company_name"]].rename(
        columns={"company": "company_name"}
    )
    df = base.merge(eto_matched, on="normalized_company_name", how="left")
    df = df.merge(
        load_eto_core().drop(columns=["eto_name"]),
        on="eto_id",
        how="left",
    )
    df = df.merge(wrds_matched, on="normalized_company_name", how="left")

    df["ai_publication_share"] = _share(df["ai_publications"], df["total_publications"])
    df["ai_patent_share"] = _share(df["ai_patents"], df["total_patents"])
    df["tech_team1_worker_share"] = _share(df["tech_team1_workers"], df["employees_wrds"])
    df["ai_worker_share"] = _share(df["ai_workers"], df["employees_wrds"])

    eto_matched = df["eto_id"].notna()
    df["ai_publication_share_reason"] = _missing_reason(eto_matched, df["total_publications"])
    df["ai_patent_share_reason"] = _missing_reason(eto_matched, df["total_patents"])

    # Both People shares share one numerator source (ETO) and one denominator
    # source (WRDS employees), so a firm missing either has no record for
    # either share -- "unmatched" -- regardless of which side failed; the ETO
    # and WRDS match-method columns in `inputs` (not carried into the
    # indicator parquet) already say which.
    worker_matched = eto_matched & df["gvkey"].notna()
    df["tech_team1_worker_share_reason"] = _missing_reason(worker_matched, df["employees_wrds"])
    df["ai_worker_share_reason"] = _missing_reason(worker_matched, df["employees_wrds"])

    df["technology_complete"] = df[TECHNOLOGY_FEATURES].notna().all(axis=1)
    df["people_complete"] = df[PEOPLE_FEATURES].notna().all(axis=1)
    return df.sort_values("rank").reset_index(drop=True)


def technology_indicator(inputs: pd.DataFrame) -> pd.DataFrame:
    """One row per universe firm; shares are NaN where inputs are missing,
    with a companion `_reason` column distinguishing unmatched firms from
    matched firms with a zero/unavailable denominator."""
    return inputs[ID_COLS + TECHNOLOGY_FEATURES + TECHNOLOGY_REASON_COLS].reset_index(drop=True)


def people_indicator(inputs: pd.DataFrame) -> pd.DataFrame:
    """One row per universe firm; shares are NaN where inputs are missing,
    with a companion `_reason` column distinguishing unmatched firms from
    matched firms with a zero/unavailable denominator."""
    return inputs[ID_COLS + PEOPLE_FEATURES + PEOPLE_REASON_COLS].reset_index(drop=True)
