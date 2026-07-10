"""Compose per-indicator parquets into the AI Maturity Index feature matrix.

This module is intentionally indicator-agnostic: it discovers every
parquet in `data_clean/indicators/<universe>/` and joins their feature
columns onto that universe's base row set. New indicators dropped into
that folder are picked up automatically as long as they expose a
`normalized_company_name` join key. Which universe feeds the index is a
single argument.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..indicators.common.universe import load_universe
from ..indicators.common.io import (
    INDICATORS_DIR,
    index_output_path,
    list_indicators,
    read_indicator,
)

log = logging.getLogger(__name__)

JOIN_KEY = "normalized_company_name"
BASE_COLS = ["rank", "company", "ticker", "industry", "normalized_company_name"]
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


def compose_index(universe: str, write: bool = True) -> pd.DataFrame:
    """Join all per-indicator parquets of one universe onto its base table.

    Each indicator's feature columns are namespaced with its indicator
    name (e.g. `strategy__net_tone`, `technology__ai_patent_share`) to
    avoid collisions across indicators. The result is keyed on
    `normalized_company_name` and written to
    `data_clean/ai_maturity_index_<universe>.parquet`.
    """
    universe_df = load_universe(universe)
    base = universe_df[[c for c in BASE_COLS if c in universe_df.columns]]

    indicators = list_indicators(universe)
    if not indicators:
        log.warning("No indicator parquets found in %s", INDICATORS_DIR / universe)
    else:
        log.info("Composing %d indicators for %s: %s", len(indicators), universe, ", ".join(indicators))

    out = base
    for ind in indicators:
        df = read_indicator(ind, universe)
        if JOIN_KEY not in df.columns:
            log.warning("Indicator %s lacks %s column; skipping", ind, JOIN_KEY)
            continue
        prefixed = _prefix_features(df, ind)
        out = out.merge(prefixed, on=JOIN_KEY, how="left")

    if write:
        out_path = index_output_path(universe)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        log.info("Wrote %d rows x %d cols to %s", len(out), out.shape[1], out_path)
    return out
