"""Index schema: which composed columns form the AI Maturity Index.

Five dimensions with two indicators each, all oriented higher-is-better:

- the six NLP indicators measure how much a firm discloses about AI in its
  10-K (extensive `ai_sentence_share` per Item) and how it frames that
  disclosure (intensive tone). `net_tone` reads more-positive sentiment as
  stronger strategic/operational commitment to AI; `fls_score` reads a more
  forward-looking framing of AI risk as more anticipatory governance. Both
  are valence assumptions and are documented as such in the thesis.
- the four structured indicators measure revealed AI activity: research
  (publications), invention (patents), and workforce composition.

Column names follow `compose_index`'s `<indicator>__<feature>` convention.
The companion columns (`*_reason`, `item_parsed`, `n_ai_sentences`) carry
the missing-data mechanisms and never enter the index itself.
"""

from __future__ import annotations

DIMENSIONS: dict[str, list[str]] = {
    "strategy": ["strategy__ai_sentence_share", "strategy__net_tone"],
    "operations": ["operations__ai_sentence_share", "operations__net_tone"],
    "governance": ["governance__ai_sentence_share", "governance__fls_score"],
    "technology": ["technology__ai_publication_share", "technology__ai_patent_share"],
    "people": ["people__tech_team1_worker_share", "people__ai_worker_share"],
}

INDEX_INDICATORS: list[str] = [c for cols in DIMENSIONS.values() for c in cols]

NLP_DIMENSIONS = ("strategy", "operations", "governance")
STRUCTURED_DIMENSIONS = ("technology", "people")

ID_COLS = ["rank", "company", "ticker", "normalized_company_name"]


def qa_columns(dimension: str) -> list[str]:
    """Companion columns used to diagnose why an indicator is missing."""
    if dimension in NLP_DIMENSIONS:
        return [f"{dimension}__item_parsed", f"{dimension}__n_ai_sentences"]
    return [f"{c}_reason" for c in DIMENSIONS[dimension]]


ALL_QA_COLS: list[str] = [c for d in DIMENSIONS for c in qa_columns(d)]
