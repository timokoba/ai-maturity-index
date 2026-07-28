"""Missing-data analysis and imputation for the index matrix (OECD 1.3).

The pipeline marks missing values upstream instead of dropping firms, and
each indicator carries its missing mechanism (structured: `*_reason`
columns; NLP: `item_parsed` / `n_ai_sentences`). That lets the imputation
distinguish structural absence from coverage gaps:

- "zero_denominator" is structural (NMAR modelled explicitly, as the OECD
  handbook requires): a firm with zero total patents demonstrably has zero
  AI patents, so the share is imputed as 0 and flagged.
- a tone missing because a parsed Item contains no AI sentences is also
  structural, but has no defensible point value -- it stays NaN and the
  dimension is down-weighted at aggregation time instead.
- coverage gaps ("unmatched" source, unparsed Item, no 10-K) stay NaN:
  imputing both indicators of a dimension from other firms would make the
  dimension score fully synthetic (Dempster & Rubin's "dangerous" case).
  A sector-median imputation exists only as a sensitivity scenario.
"""

from __future__ import annotations

import pandas as pd

from ..indicators.common.io import DATA_RAW
from .schema import DIMENSIONS, INDEX_INDICATORS, NLP_DIMENSIONS

MECH_DEFINED = "defined"
MECH_NO_FILING = "no_filing"
MECH_ITEM_UNPARSED = "item_unparsed"
MECH_NO_AI_SENTENCES = "no_ai_sentences"
MECH_UNMATCHED = "unmatched"
MECH_ZERO_DENOMINATOR = "zero_denominator"


def missing_mechanism(df: pd.DataFrame, indicator: str) -> pd.Series:
    """Categorise each firm's status on one indicator using its companion
    columns. Returns one of the MECH_* labels per row."""
    out = pd.Series(MECH_DEFINED, index=df.index, dtype="object")
    missing = df[indicator].isna()

    reason_col = f"{indicator}_reason"
    if reason_col in df.columns:
        out[missing] = df.loc[missing, reason_col]
        return out

    dimension = indicator.split("__")[0]
    parsed = df.get(f"{dimension}__item_parsed")
    n_ai = df.get(f"{dimension}__n_ai_sentences")
    if parsed is None:
        out[missing] = MECH_NO_FILING
        return out
    out[missing & parsed.isna()] = MECH_NO_FILING
    out[missing & (parsed == 0)] = MECH_ITEM_UNPARSED
    if n_ai is not None:
        out[missing & (parsed == 1) & (n_ai == 0)] = MECH_NO_AI_SENTENCES
    return out


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per index indicator: coverage plus the mechanism breakdown."""
    rows: list[dict] = []
    for indicator in INDEX_INDICATORS:
        mech = missing_mechanism(df, indicator)
        counts = mech.value_counts()
        rows.append(
            {
                "indicator": indicator,
                "n_defined": int(counts.get(MECH_DEFINED, 0)),
                "pct_missing": float(df[indicator].isna().mean()),
                **{
                    m: int(counts.get(m, 0))
                    for m in (
                        MECH_ZERO_DENOMINATOR,
                        MECH_UNMATCHED,
                        MECH_NO_AI_SENTENCES,
                        MECH_ITEM_UNPARSED,
                        MECH_NO_FILING,
                    )
                },
            }
        )
    return pd.DataFrame(rows).set_index("indicator")


def impute_structural_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set indicators whose reason is "zero_denominator" to 0.0.

    Returns (imputed frame, flags frame). The flags frame has one bool
    column `<indicator>_imputed` per index indicator (False everywhere for
    indicators without a reason column).
    """
    out = df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=[f"{c}_imputed" for c in INDEX_INDICATORS])
    for indicator in INDEX_INDICATORS:
        reason_col = f"{indicator}_reason"
        if reason_col not in df.columns:
            continue
        mask = df[reason_col] == MECH_ZERO_DENOMINATOR
        out.loc[mask, indicator] = 0.0
        flags.loc[mask, f"{indicator}_imputed"] = True
    return out, flags


def load_gics_sector(universe_df: pd.DataFrame) -> pd.Series:
    """GICS sector per firm (index-aligned with `universe_df`), via gvkey
    from the Compustat fundamentals file. NaN where no gvkey or no sector.
    Used only by the sector-median sensitivity scenario."""
    w = pd.read_csv(
        DATA_RAW / "wrds" / "fundamentals_annual.csv",
        usecols=["gvkey", "fyear", "gsector"],
        dtype={"gvkey": str},
    )
    w = w.dropna(subset=["gsector"]).sort_values("fyear")
    sector_by_gvkey = w.drop_duplicates("gvkey", keep="last").set_index("gvkey")["gsector"]
    return universe_df["gvkey"].map(sector_by_gvkey)


def impute_sector_median(
    df: pd.DataFrame, sector: pd.Series, min_group: int = 5
) -> pd.DataFrame:
    """Sensitivity scenario only: fill every remaining NaN indicator with
    its GICS-sector median (groups below `min_group` non-missing values
    fall back to the overall median)."""
    out = df.copy()
    for indicator in INDEX_INDICATORS:
        overall = out[indicator].median()
        medians = out.groupby(sector)[indicator].transform(
            lambda s: s.median() if s.notna().sum() >= min_group else overall
        )
        out[indicator] = out[indicator].fillna(medians).fillna(overall)
    return out
