"""Measurement reliability of the tone indicators.

A firm's tone is the mean of FinBERT scores over its AI sentences in one
Item, so it is a sample estimate whose precision depends on how many
sentences it rests on. Some firms have one sentence, others fifty; treating
both as equally precise lets a coin flip on a single sentence move a
dimension as much as a well-evidenced average.

The correction is the standard variance-components / empirical-Bayes
result. Writing sentence j of firm i as

    y_ij ~ (theta_i, sigma^2)      sentence-level noise
    theta_i ~ (mu, tau^2)          genuine spread between firms

the firm mean based on n_i sentences carries reliability

    lambda_i = n_i / (n_i + k),    k = sigma^2 / tau^2

so k is *estimated*, not chosen: it is the ratio of within-firm to
between-firm variance, both identified from the cached sentence scores by a
one-way random-effects model. The intraclass correlation ICC =
tau^2 / (tau^2 + sigma^2) is the same quantity expressed per sentence --
how much one sentence says about the firm rather than about which sentence
happened to be sampled.

The share indicators get reliability 1: a share is a complete count over
the Item, not an estimate from a sample of its sentences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.common.io import load_cached_step
from .schema import DIMENSIONS, NLP_DIMENSIONS

# Which cached column carries the per-sentence value behind each firm-level
# tone. Must mirror how `nlp_features.aggregate` builds the indicator.
TONE_SENTENCE_VALUE = {
    "strategy": lambda d: d["pos"] - d["neg"],
    "operations": lambda d: d["pos"] - d["neg"],
    "governance": lambda d: d["p_specific"] + d["p_nonspecific"],
}


def _variance_components(scored: pd.DataFrame, value: pd.Series) -> dict:
    """One-way random-effects decomposition of sentence scores by firm."""
    d = scored.assign(_y=value)
    grouped = d.groupby("accession_number")["_y"]
    n_i, mean_i, var_i = grouped.size(), grouped.mean(), grouped.var(ddof=1)
    n_total, n_firms = len(d), len(n_i)

    ms_between = float((n_i * (mean_i - d["_y"].mean()) ** 2).sum()) / (n_firms - 1)
    ms_within = float(((n_i - 1) * var_i.fillna(0)).sum()) / (n_total - n_firms)
    # effective group size for unbalanced designs
    n0 = (n_total - (n_i ** 2).sum() / n_total) / (n_firms - 1)
    tau2 = max((ms_between - ms_within) / n0, 1e-12)

    # method of moments as an independent cross-check on tau^2
    multi = n_i > 1
    sigma2_mom = float(((n_i[multi] - 1) * var_i[multi]).sum() / (n_i[multi] - 1).sum())
    tau2_mom = max(float(mean_i.var(ddof=1) - sigma2_mom * (1.0 / n_i).mean()), 1e-12)

    return {
        "n_sentences": n_total,
        "n_firms": n_firms,
        "median_n": float(n_i.median()),
        "sigma2": ms_within,
        "tau2": tau2,
        "k": ms_within / tau2,
        "icc": tau2 / (tau2 + ms_within),
        "k_moments_check": sigma2_mom / tau2_mom,
    }


def estimate_tone_reliability(universe: str) -> pd.DataFrame:
    """Estimate k and the ICC for each tone indicator from the scored caches.

    One row per NLP dimension. `k` is the shrinkage constant used by
    `indicator_reliability`; `k_moments_check` is the same quantity from a
    method-of-moments estimator, reported so the reader can see the two
    agree in order of magnitude.
    """
    rows: list[dict] = []
    for dim in NLP_DIMENSIONS:
        scored = load_cached_step(dim, "scored", universe)
        if scored is None or scored.empty:
            raise RuntimeError(
                f"No scored cache for {dim!r} in universe {universe!r}. "
                f"Run notebooks/02_feature_engineering/{dim}.ipynb first."
            )
        stats = _variance_components(scored, TONE_SENTENCE_VALUE[dim](scored))
        rows.append({"dimension": dim, "indicator": DIMENSIONS[dim][1], **stats})
    return pd.DataFrame(rows).set_index("dimension")


def indicator_reliability(matrix: pd.DataFrame, k_by_dimension: dict[str, float]) -> pd.DataFrame:
    """Per-firm reliability weight in [0, 1] for every index indicator.

    Tone indicators are weighted by `n / (n + k)` with n the number of AI
    sentences behind them; every other present indicator gets 1, and a
    missing value gets 0. That makes the existing missing-data rule the
    limiting case of the same mechanism: a firm with no AI sentences has
    tone reliability 0, so its dimension rests on the share alone at half
    weight -- exactly as before, but now reached continuously instead of
    across a jump.

    Pass the frame whose presence pattern should count, i.e. the imputed
    one: a structurally imputed zero is an observed value and must keep
    full weight, not be read as missing.
    """
    reliability = pd.DataFrame(index=matrix.index)
    for dim, cols in DIMENSIONS.items():
        for col in cols:
            present = matrix[col].notna()
            is_tone = dim in NLP_DIMENSIONS and col == cols[1]
            if is_tone:
                n = matrix[f"{dim}__n_ai_sentences"].astype(float)
                lam = n / (n + k_by_dimension[dim])
                reliability[col] = lam.where(present, 0.0).fillna(0.0)
            else:
                reliability[col] = present.astype(float)
    return reliability
