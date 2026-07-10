"""Shared NLP feature engine for the 10-K AI-disclosure dimensions.

Resolves one 10-K per universe firm, extracts Items 1 / 1A / 7, segments and
AI-filters their sentences, and scores tone. The same front end feeds three
dimensions defined in `dimensions.py` (strategy, operations, governance),
each aggregated to a firm-level indicator by `aggregate_dimension`.
"""

from .edgar import resolve_filings
from .parse import extract_items, fetch_filing, assemble_sections
from .filter import filter_ai_sentences, AI_KEYWORDS
from .sentiment import score_sentences
from .forward_looking import score_forward_looking
from .dimensions import Dimension, DIMENSIONS
from .aggregate import aggregate_dimension
from .review import build_ai_sentence_review

__all__ = [
    "resolve_filings",
    "extract_items",
    "fetch_filing",
    "assemble_sections",
    "filter_ai_sentences",
    "AI_KEYWORDS",
    "score_sentences",
    "score_forward_looking",
    "Dimension",
    "DIMENSIONS",
    "aggregate_dimension",
    "build_ai_sentence_review",
]
