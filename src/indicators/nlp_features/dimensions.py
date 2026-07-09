"""Registry of the three AI-disclosure dimensions.

Each dimension shares the same front end (resolve filings -> parse items ->
segment sentences -> AI keyword filter) and is then operationalised on a
single 10-K Item as an extensive margin (relative occurrence of AI
sentences) plus an intensive margin (tone of those sentences). The only
things that vary across dimensions are which Item supplies the text and
which tone model scores it, so a dimension is expressed here as one line
of configuration rather than duplicated pipeline code.

- Strategy   -> Item 1 (Business): where a firm frames AI as part of what
  it does. Tone via FinBERT-tone sentiment.
- Operations -> Item 7 (MD&A): where a firm discusses AI in results and
  operations. Tone via FinBERT-tone sentiment.
- Governance -> Item 1A (Risk Factors): where a firm discloses AI risk.
  Sentiment is uninformative here (risk factors are uniformly negative),
  so tone is the forward-looking orientation of the disclosure, scored
  with FinBERT-FLS (Huang, Wang, Yang 2023, CAR).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    name: str  # output indicator name, e.g. "strategy"
    item: str  # source 10-K Item: "item_1" | "item_1a" | "item_7"
    tone: str  # tone model: "sentiment" | "forward_looking"


DIMENSIONS: dict[str, Dimension] = {
    "strategy": Dimension("strategy", "item_1", "sentiment"),
    "operations": Dimension("operations", "item_7", "sentiment"),
    "governance": Dimension("governance", "item_1a", "forward_looking"),
}
