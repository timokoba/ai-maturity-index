"""Structured feature engine for the Technology and People dimensions.

Resolves the listed Fortune 500 universe to ETO PARAT companies and to
Compustat FY2024 fundamentals, assembles one wide raw-input table, and
derives four firm-level shares: AI publications and AI patents relative to
each firm's totals (Technology), and Tech Team 1 / AI workers relative to
total employees (People). The same front end feeds both dimension
notebooks and the HTML input review.
"""

from .eto import load_eto_aliases, load_eto_core, load_eto_tickers, match_to_eto
from .features import (
    ID_COLS,
    INDICATOR,
    PEOPLE_FEATURES,
    PEOPLE_REASON_COLS,
    REASON_UNMATCHED,
    REASON_ZERO_DENOMINATOR,
    TECHNOLOGY_FEATURES,
    TECHNOLOGY_REASON_COLS,
    build_inputs,
    people_indicator,
    technology_indicator,
)
from .match import normalize_ticker, unambiguous_map
from .review import build_structured_inputs_review
from .wrds import TARGET_FYEAR, dedupe_wrds, load_wrds_fy, match_to_wrds

__all__ = [
    "load_eto_core",
    "load_eto_tickers",
    "load_eto_aliases",
    "match_to_eto",
    "load_wrds_fy",
    "dedupe_wrds",
    "match_to_wrds",
    "TARGET_FYEAR",
    "build_inputs",
    "technology_indicator",
    "people_indicator",
    "INDICATOR",
    "TECHNOLOGY_FEATURES",
    "PEOPLE_FEATURES",
    "TECHNOLOGY_REASON_COLS",
    "PEOPLE_REASON_COLS",
    "REASON_UNMATCHED",
    "REASON_ZERO_DENOMINATOR",
    "ID_COLS",
    "normalize_ticker",
    "unambiguous_map",
    "build_structured_inputs_review",
]
