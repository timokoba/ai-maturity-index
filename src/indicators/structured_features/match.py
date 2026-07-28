"""Source-agnostic matching primitives for the structured feature pipeline."""

from __future__ import annotations

import re

import pandas as pd

MATCH_GVKEY = "gvkey"
MATCH_CIK = "cik"
MATCH_TICKER = "ticker"
MATCH_NAME = "name"
MATCH_ALIAS = "alias"
MATCH_AMBIGUOUS = "ambiguous"
MATCH_NONE = "unmatched"


def normalize_ticker(ticker: object) -> str:
    """Reduce a ticker to its alphanumeric core so exchange-specific
    spellings collide ("BRK.A" == "BRK-A" == "BRKA") while share classes
    stay distinct. Missing values and the Fortune placeholder "~" map to "".
    """
    if ticker is None or (isinstance(ticker, float) and pd.isna(ticker)):
        return ""
    s = str(ticker).upper().strip()
    if s in ("", "~", "NAN"):
        return ""
    return re.sub(r"[^A-Z0-9]", "", s)


def unambiguous_map(df: pd.DataFrame, key: str, value: str) -> tuple[dict, set]:
    """Build a key -> value lookup keeping only keys that map to exactly
    one distinct value.

    Returns (mapping, ambiguous_keys). Ambiguous keys are never resolved
    silently; callers let the affected firms fall through to the next
    matching pass or report them for manual adjudication.
    """
    pairs = df[[key, value]].dropna().drop_duplicates()
    if pairs[key].dtype == object:
        pairs = pairs[pairs[key] != ""]
    counts = pairs.groupby(key)[value].nunique()
    ambiguous = set(counts[counts > 1].index)
    ok = pairs[~pairs[key].isin(ambiguous)]
    return dict(zip(ok[key], ok[value])), ambiguous
