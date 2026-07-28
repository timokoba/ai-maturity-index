"""Weighting and aggregation of the AI Maturity Index (OECD 1.6).

Hierarchical equal weighting, with a deliberate split between the two
levels at which a "weight" could enter.

*Within* a dimension the question is an estimation one -- how best to read
this dimension off the indicators available for this firm -- so the
per-indicator reliability weight r in [0, 1] from `reliability.py` applies:

    dimension score = sum(r_j * x_j) / sum(r_j)
    coverage        = sum(r_j) / 2

An indicator that is missing has r = 0 and drops out of the average
entirely; it is never averaged in as a zero. Because r varies continuously,
"no tone at all" and "a tone resting on one sentence" are no longer
separated by a jump. `coverage` records how much indicator evidence the
dimension actually rests on -- 1.0 for two fully reliable indicators, 0.5
for one, 0.0 for none. It is a data-quality diagnostic, **not** a weight.

*Across* dimensions the question is a definitional one -- how much does this
facet matter for the construct -- and the answer is that all five matter
equally. Dimension weights are therefore 1 for every observed dimension and
0 only where nothing was observed, which under the availability rule
excludes the firm. Letting coverage weight the composite instead would make
an unintended value judgement ("dimensions we measured better matter
more"), would contradict the formative framing, and would treat an observed
zero as if it were partly unknown.

Weights are renormalized per firm, so the composite is always a proper
weighted mean of what is actually observed. Linear aggregation is the main
specification (full compensability across dimensions, appropriate for an
economic maturity measure); a geometric variant with reduced
compensability at the low end is provided for the sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DIMENSIONS


def dimension_scores(
    norm_df: pd.DataFrame, reliability: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-firm dimension scores and their indicator coverage.

    `norm_df` holds the ten normalized indicators in [0, 1]. `reliability`
    gives each indicator a weight in [0, 1] per firm; omit it to weight
    every present indicator equally, which is the plain present/missing
    rule. Returns (scores, coverage), each with one column per dimension;
    a dimension with no usable indicator scores NaN at coverage 0.

    `coverage` says how much indicator evidence a dimension rests on, not
    how much it should count: pass `(coverage > 0).astype(float)` to
    `composite_index` for the equal dimension weighting the framework calls
    for. Feeding coverage itself in as weights is available for the
    sensitivity analysis but is not the specification.
    """
    if reliability is None:
        reliability = norm_df.notna().astype(float)
    # an unspecified weight for a present indicator means "no reliability
    # information", which is full weight -- never silently zero
    reliability = reliability.reindex(index=norm_df.index, columns=norm_df.columns).fillna(1.0)
    # an indicator that is missing cannot carry weight, whatever was passed in
    reliability = reliability.where(norm_df.notna(), 0.0)

    scores = pd.DataFrame(index=norm_df.index)
    coverage = pd.DataFrame(index=norm_df.index)
    for dim, cols in DIMENSIONS.items():
        r = reliability[cols]
        total = r.sum(axis=1)
        weighted = (norm_df[cols].fillna(0.0) * r).sum(axis=1)
        scores[dim] = weighted / total.where(total > 0)
        coverage[dim] = total / len(cols)
    return scores, coverage


def composite_index(scores: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Linear composite on a 0-100 scale.

    NaN for firms where any dimension has weight 0 (no indicator at all in
    that dimension), per the availability rule.
    """
    eligible = (weights > 0).all(axis=1)
    weighted = (scores * weights).sum(axis=1) / weights.sum(axis=1)
    return (weighted * 100).where(eligible)


def geometric_index(scores: pd.DataFrame, weights: pd.DataFrame, floor: float = 0.01) -> pd.Series:
    """Geometric composite on a 0-100 scale (sensitivity variant).

    Dimension scores are floored at `floor` because the geometric mean is
    zero (and its log undefined) at 0; the floor bounds how hard a single
    empty dimension can pull the composite down. Same eligibility rule as
    the linear composite.
    """
    eligible = (weights > 0).all(axis=1)
    clipped = scores.clip(lower=floor)
    log_mean = (np.log(clipped) * weights).sum(axis=1) / weights.sum(axis=1)
    return (np.exp(log_mean) * 100).where(eligible)
