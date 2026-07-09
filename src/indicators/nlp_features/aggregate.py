"""Aggregate sentence-level tone scores to firm-year level, per dimension.

Each dimension (strategy, operations, governance) is operationalised on a
single 10-K Item following the extensive/intensive margin decomposition of
Babina, Fedyk, He, Hodson (2024, JFE):

- `ai_sentence_share` (extensive margin) -- share of the Item's cleaned
  sentences that mention AI. Length-normalized to neutralize cross-firm
  differences in 10-K size (Loughran-McDonald 2011 convention).
- an intensive-margin tone, conditional on at least
  `MIN_AI_SENTENCES_FOR_TONE` AI sentences in the Item (below that the mean is
  too noisy to compare across firms and is set to NaN). Every tone is a mean of
  FinBERT softmax probabilities over the Item's AI sentences, so all three
  dimensions are on a comparable, continuous footing:
    - sentiment dimensions (strategy, operations) report `net_tone`, the mean
      of (P(positive) - P(negative)) in [-1, 1] (FinBERT-tone; Huang, Wang,
      Yang 2023, CAR).
    - the forward-looking dimension (governance) reports `fls_score`, the mean
      P(forward-looking) in [0, 1] (FinBERT-FLS) -- how prospectively, rather
      than merely descriptively, the firm frames its AI risk.

Each dimension thus contributes exactly two indicators: the extensive
`ai_sentence_share` and one intensive tone score.

Each dimension draws on a single Item, so `MIN_AI_SENTENCES_FOR_TONE` bites
harder than a pooled indicator would (especially governance, where Item 1A
carries fewer AI mentions); the notebooks' sanity cell reports the pass-rate at
3/5/10/20 so it can be revisited. Output is written to
`data_clean/indicators/<dimension>.parquet`.
"""

from __future__ import annotations

import pandas as pd

from ..common.io import write_indicator
from .dimensions import Dimension

MIN_AI_SENTENCES_FOR_TONE = 5

# A filing is "parse-complete" only if all three Items yielded at least
# MIN_ITEM_SENTENCES_FOR_PARSE clean sentences. Genuine sections run to hundreds
# of sentences (median ~250), so this floor discards edgartools truncation stubs
# (a handful of sentences) without touching a real section. The flag is written
# identically into every dimension's output so the same firms drop everywhere.
ALL_ITEMS = ("item_1", "item_1a", "item_7")
MIN_ITEM_SENTENCES_FOR_PARSE = 25


def _net_tone(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float((df["pos"].sum() - df["neg"].sum()) / len(df))


def aggregate_dimension(
    dimension: Dimension,
    filings: pd.DataFrame,
    scored_sentences: pd.DataFrame,
    sentence_totals: pd.DataFrame,
    min_ai_sentences_for_tone: int = MIN_AI_SENTENCES_FOR_TONE,
) -> pd.DataFrame:
    """Aggregate one dimension's per-sentence scores to one row per firm.

    `dimension.item` selects the source Item and `dimension.tone` the tone
    metric. `scored_sentences` is the long table of AI sentences (with an
    `item` column) scored by `sentiment.score_sentences` (columns `pos`, `neg`,
    `label`) or `forward_looking.score_forward_looking` (`p_specific`,
    `p_nonspecific`). `sentence_totals` supplies each Item's total cleaned
    sentence count (`filter.filter_ai_sentences`), the denominator of
    `ai_sentence_share`. Firms with fewer than `min_ai_sentences_for_tone` AI
    sentences get NaN tone (unreliable mean; handle at index-composition time).

    Returns one row per firm -- extensive margin, intensive-margin tone, and a
    `parse_complete` flag identical across dimensions so the same incomplete
    firms drop everywhere -- and writes `data_clean/indicators/<name>.parquet`.
    """
    item = dimension.item
    totals_col = f"n_sentences_{item}"

    base = filings[
        ["cik", "ticker", "company_name", "normalized_company_name", "accession_number", "fiscal_year"]
    ].copy()

    if len(sentence_totals) > 0:
        totals_by_acc = sentence_totals.set_index("accession_number")
    else:
        totals_by_acc = None

    if len(scored_sentences) == 0 or "item" not in scored_sentences.columns:
        item_scored = scored_sentences.iloc[0:0]
    else:
        item_scored = scored_sentences[scored_sentences["item"] == item]

    rows: list[dict] = []
    for _, f in base.iterrows():
        sub = item_scored[item_scored["accession_number"] == f["accession_number"]]
        n_ai = len(sub)

        if totals_by_acc is not None and f["accession_number"] in totals_by_acc.index:
            t = totals_by_acc.loc[f["accession_number"]]
            n_total = int(t.get(totals_col, 0))
            parse_complete = all(
                int(t.get(f"n_sentences_{it}", 0)) >= MIN_ITEM_SENTENCES_FOR_PARSE
                for it in ALL_ITEMS
            )
        else:
            n_total = 0
            parse_complete = False

        row = dict(
            cik=f["cik"],
            ticker=f["ticker"],
            company_name=f["company_name"],
            normalized_company_name=f["normalized_company_name"],
            accession_number=f["accession_number"],
            fiscal_year=f["fiscal_year"],
            parse_complete=int(parse_complete),
            has_ai_mention=int(n_ai > 0),
            n_ai_sentences=int(n_ai),
            n_total_sentences=int(n_total),
            ai_sentence_share=float(n_ai / n_total) if n_total > 0 else 0.0,
        )

        has_tone = n_ai >= min_ai_sentences_for_tone
        if dimension.tone == "sentiment":
            row["net_tone"] = _net_tone(sub) if has_tone else float("nan")
        elif dimension.tone == "forward_looking":
            row["fls_score"] = (
                float((sub["p_specific"] + sub["p_nonspecific"]).mean()) if has_tone else float("nan")
            )
        else:
            raise ValueError(f"unknown tone metric: {dimension.tone!r}")

        rows.append(row)

    out = pd.DataFrame(rows)
    write_indicator(out, dimension.name)
    return out
