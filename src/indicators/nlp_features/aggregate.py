"""Aggregate sentence-level tone scores to firm-year level, per dimension.

Each dimension (strategy, operations, governance) is operationalised on a
single 10-K Item following the extensive/intensive margin decomposition of
Babina, Fedyk, He, Hodson (2024, JFE):

- `ai_sentence_share` (extensive margin) -- share of the Item's cleaned
  sentences that mention AI. Length-normalized to neutralize cross-firm
  differences in 10-K size (Loughran-McDonald 2011 convention).
- an intensive-margin tone, the mean of FinBERT softmax probabilities over
  the Item's AI sentences, so all three dimensions are on a comparable,
  continuous footing:
    - sentiment dimensions (strategy, operations) report `net_tone`, the mean
      of (P(positive) - P(negative)) in [-1, 1] (FinBERT-tone; Huang, Wang,
      Yang 2023, CAR).
    - the forward-looking dimension (governance) reports `fls_score`, the mean
      P(forward-looking) in [0, 1] (FinBERT-FLS) -- how prospectively, rather
      than merely descriptively, the firm frames its AI risk.

Missing data is marked, never conflated with zero:

- an Item that did not parse (fewer than `MIN_ITEM_SENTENCES_FOR_PARSE`
  clean sentences) has no defined share -- `ai_sentence_share` and
  `has_ai_mention` are NaN, and `item_parsed` is 0. A parsed Item with no
  AI sentences is a genuine 0.
- a tone over zero AI sentences is undefined -- NaN, never 0. Beyond that
  there is no minimum-sentence threshold: a firm with little AI text is
  already reflected in `ai_sentence_share`, and gating the tone on top would
  penalise it twice while leaving nothing sensible to impute. The per-firm
  `n_ai_sentences` ships with every row so noisy small-sample tones can be
  weighted or filtered downstream if needed.

Each dimension thus contributes exactly two indicators: the extensive
`ai_sentence_share` and one intensive tone score. Output is written to
`data_clean/indicators/<universe>/<dimension>.parquet`.
"""

from __future__ import annotations

import pandas as pd

from ..common.io import write_indicator
from .dimensions import Dimension

# A filing is "parse-complete" only if all three Items yielded at least
# MIN_ITEM_SENTENCES_FOR_PARSE clean sentences. Genuine sections run to hundreds
# of sentences (median ~250), so this floor discards edgartools truncation stubs
# (a handful of sentences) without touching a real section. The flag is written
# identically into every dimension's output; the per-dimension `item_parsed`
# flag marks whether this dimension's own Item cleared the floor.
ALL_ITEMS = ("item_1", "item_1a", "item_7")
MIN_ITEM_SENTENCES_FOR_PARSE = 25


def _net_tone(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float((df["pos"].sum() - df["neg"].sum()) / len(df))


def aggregate_dimension(
    dimension: Dimension,
    filings: pd.DataFrame,
    scored_sentences: pd.DataFrame,
    sentence_totals: pd.DataFrame,
    universe: str,
) -> pd.DataFrame:
    """Aggregate one dimension's per-sentence scores to one row per firm.

    `dimension.item` selects the source Item and `dimension.tone` the tone
    metric. `scored_sentences` is the long table of AI sentences (with an
    `item` column) scored by `sentiment.score_sentences` (columns `pos`, `neg`,
    `label`) or `forward_looking.score_forward_looking` (`p_specific`,
    `p_nonspecific`). `sentence_totals` supplies each Item's total cleaned
    sentence count (`filter.filter_ai_sentences`), the denominator of
    `ai_sentence_share`.

    Returns one row per firm and writes
    `data_clean/indicators/<universe>/<name>.parquet`. Firms whose Item did
    not parse keep their row with NaN share/tone and `item_parsed = 0`;
    nothing is dropped here -- missing values are handled at
    index-composition time.
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

        item_parsed = n_total >= MIN_ITEM_SENTENCES_FOR_PARSE

        row = dict(
            cik=f["cik"],
            ticker=f["ticker"],
            company_name=f["company_name"],
            normalized_company_name=f["normalized_company_name"],
            accession_number=f["accession_number"],
            fiscal_year=f["fiscal_year"],
            parse_complete=int(parse_complete),
            item_parsed=int(item_parsed),
            n_ai_sentences=int(n_ai),
            n_total_sentences=int(n_total),
            has_ai_mention=float(n_ai > 0) if item_parsed else float("nan"),
            ai_sentence_share=float(n_ai / n_total) if item_parsed else float("nan"),
        )

        has_tone = item_parsed and n_ai > 0
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
    write_indicator(out, dimension.name, universe)
    return out
