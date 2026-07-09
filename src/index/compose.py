"""Compose per-indicator parquets into the AI Maturity Index feature matrix.

This module is intentionally indicator-agnostic: it discovers every
parquet in `data_clean/indicators/` and joins their feature columns
onto the master Fortune 500 row set. New indicators dropped into that
folder are picked up automatically as long as they expose a
`normalized_company_name` join key.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..indicators.common.company_ids import load_fortune500
from ..indicators.common.io import (
    INDEX_OUTPUT,
    INDICATORS_DIR,
    list_indicators,
    read_indicator,
)

log = logging.getLogger(__name__)

JOIN_KEY = "normalized_company_name"
NON_FEATURE_COLS = {
    "cik",
    "ticker",
    "company_name",
    "normalized_company_name",
    "accession_number",
    "fiscal_year",
}


def _prefix_features(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    rename_map = {c: f"{indicator}__{c}" for c in feature_cols}
    keep = [JOIN_KEY] + feature_cols
    return df[keep].rename(columns=rename_map)


def compose_index(write: bool = True) -> pd.DataFrame:
    """Join all per-indicator parquets onto the Fortune 500 base table.

    Each indicator's feature columns are namespaced with its indicator
    name (e.g. `strategy__net_tone`, `governance__fls_share`) to avoid
    collisions across indicators. The result is keyed on
    `normalized_company_name`.
    """
    base = load_fortune500()[["rank", "company", "ticker", "industry", "normalized_company_name"]]

    indicators = list_indicators()
    if not indicators:
        log.warning("No indicator parquets found in %s", INDICATORS_DIR)
    else:
        log.info("Composing %d indicators: %s", len(indicators), ", ".join(indicators))

    out = base
    for ind in indicators:
        df = read_indicator(ind)
        if JOIN_KEY not in df.columns:
            log.warning("Indicator %s lacks %s column; skipping", ind, JOIN_KEY)
            continue
        prefixed = _prefix_features(df, ind)
        out = out.merge(prefixed, on=JOIN_KEY, how="left")

    if write:
        INDEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(INDEX_OUTPUT, index=False)
        log.info("Wrote %d rows x %d cols to %s", len(out), out.shape[1], INDEX_OUTPUT)
    return out
