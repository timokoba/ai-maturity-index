"""Aggregate sentence-level FinBERT-tone scores to firm-year level.

Produces the canonical Strategy & Governance indicator table written
to `data_clean/indicators/strategy_governance.parquet`.

The indicator follows the extensive/intensive margin decomposition
established by Babina, Fedyk, He, Hodson (2024, JFE), with two
firm-level components:

- `ai_sentence_share` (extensive margin) — share of AI-keyword sentences
  in the cleaned narrative text across Items 1, 1A, and 7. Length-
  normalized to neutralize cross-firm differences in 10-K size
  (Loughran-McDonald 2011 convention). Item 7A is excluded — it is
  dominated by quantitative market-risk content, Loughran-McDonald
  (2016, JAR) flag it as unreliable for tone analysis, and Babina et
  al. (2024) report near-zero AI mentions there.
- `net_tone_finbert` (intensive margin) — FinBERT-tone mean per
  Huang, Wang, Yang (2023, CAR), conditional on at least
  `MIN_AI_SENTENCES_FOR_TONE` AI sentences. Below that threshold the
  3-class mean is too noisy to compare across firms (SE > 0.27 at
  n=5) and is set to NaN.
"""

from __future__ import annotations

import pandas as pd

from ..common.io import write_indicator

INDICATOR_NAME = "strategy_governance"

ITEM_COLS = ("item_1", "item_1a", "item_7")

MIN_AI_SENTENCES_FOR_TONE = 5


def _net_tone(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return 0.0
    return float((df["pos"].sum() - df["neg"].sum()) / len(df))


def aggregate_firm_level(
    filings: pd.DataFrame,
    scored_sentences: pd.DataFrame,
    sentence_totals: pd.DataFrame,
    min_ai_sentences_for_tone: int = MIN_AI_SENTENCES_FOR_TONE,
) -> pd.DataFrame:
    """Aggregate per-sentence scores to one row per firm.

    Parameters
    ----------
    filings:
        Output of `edgar.resolve_fortune500_filings`, with columns
        `cik`, `ticker`, `company_name`, `normalized_company_name`,
        `accession_number`, `fiscal_year`, `filing_date`.
    scored_sentences:
        Long table of AI-relevant sentences with FinBERT scores and an
        `item` column. Columns: `cik`, `accession_number`, `item`,
        `sentence`, `pos`, `neu`, `neg`, `label`.
    sentence_totals:
        Per-filing counts of total cleaned sentences in each Item, as
        emitted by `filter.filter_ai_sentences`. Columns:
        `accession_number`, `cik`, `n_sentences_item_1`,
        `n_sentences_item_1a`, `n_sentences_item_7`. Used as the
        denominator of `ai_sentence_share`.
    min_ai_sentences_for_tone:
        Minimum AI-sentence count required to compute the FinBERT-tone
        mean. Firms below this threshold receive `net_tone_finbert = NaN`,
        signaling that the tone signal is unreliable for them and should
        be handled (drop or median-impute) at index-composition time.
        Default 5 follows Huang, Wang, Yang (2023, CAR).

    Returns
    -------
    pd.DataFrame
        One row per firm with extensive- and intensive-margin features
        plus per-Item descriptive columns for robustness reporting.
    """
    base = filings[
        ["cik", "ticker", "company_name", "normalized_company_name", "accession_number", "fiscal_year"]
    ].copy()

    if len(sentence_totals) > 0:
        totals_by_acc = sentence_totals.set_index("accession_number")
    else:
        totals_by_acc = None

    if len(scored_sentences) == 0:
        scored = pd.DataFrame(
            columns=["cik", "accession_number", "item", "pos", "neu", "neg", "label"]
        )
    else:
        scored = scored_sentences.copy()

    rows: list[dict] = []
    for _, f in base.iterrows():
        sub = scored[scored["accession_number"] == f["accession_number"]]
        n_ai = len(sub)

        if totals_by_acc is not None and f["accession_number"] in totals_by_acc.index:
            t = totals_by_acc.loc[f["accession_number"]]
            per_item_totals = {item: int(t.get(f"n_sentences_{item}", 0)) for item in ITEM_COLS}
        else:
            per_item_totals = {item: 0 for item in ITEM_COLS}
        n_total = sum(per_item_totals.values())

        row = dict(
            cik=f["cik"],
            ticker=f["ticker"],
            company_name=f["company_name"],
            normalized_company_name=f["normalized_company_name"],
            accession_number=f["accession_number"],
            fiscal_year=f["fiscal_year"],
            has_ai_mention=int(n_ai > 0),
            n_ai_sentences=int(n_ai),
            n_total_sentences=int(n_total),
            ai_sentence_share=float(n_ai / n_total) if n_total > 0 else 0.0,
        )

        if n_ai >= min_ai_sentences_for_tone:
            labels = sub["label"]
            row["net_tone_finbert"] = _net_tone(sub)
            row["pos_share"] = float((labels == "positive").mean())
            row["neu_share"] = float((labels == "neutral").mean())
            row["neg_share"] = float((labels == "negative").mean())
        else:
            row["net_tone_finbert"] = float("nan")
            row["pos_share"] = float("nan")
            row["neu_share"] = float("nan")
            row["neg_share"] = float("nan")

        for item in ITEM_COLS:
            item_sub = sub[sub["item"] == item]
            n_ai_item = len(item_sub)
            n_total_item = per_item_totals[item]
            row[f"n_ai_sentences_{item}"] = int(n_ai_item)
            row[f"n_total_sentences_{item}"] = int(n_total_item)
            row[f"ai_share_{item}"] = float(n_ai_item / n_total_item) if n_total_item > 0 else 0.0
            if n_ai_item >= min_ai_sentences_for_tone:
                row[f"net_tone_{item}"] = _net_tone(item_sub)
            else:
                row[f"net_tone_{item}"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    write_indicator(out, INDICATOR_NAME)
    return out
