"""Company identifier resolution shared across indicators.

The canonical join key in this project is `normalized_company_name`
(lowercase, stripped of legal suffixes and punctuation), matching the
convention established in `01_ingest_clean.ipynb` and the master table at
`data_clean/ai_maturity_master.csv`.
"""

from __future__ import annotations

import re

import pandas as pd

from .io import DATA_RAW

LEGAL_SUFFIXES = (
    " inc",
    " inc.",
    " incorporated",
    " corp",
    " corp.",
    " corporation",
    " co",
    " co.",
    " company",
    " ltd",
    " ltd.",
    " limited",
    " plc",
    " llc",
    " l.l.c.",
    " l.p.",
    " lp",
    " holdings",
    " group",
    " the",
)


def normalize_company_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"[\.,'’&]", "", s)
    s = re.sub(r"\s+", " ", s)
    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_fortune500() -> pd.DataFrame:
    """Return the top 500 rows of the Fortune 1000 US dataset.

    Adds a `normalized_company_name` column for joining against the master
    table. The original `rank` and `ticker` columns are preserved.
    """
    src = DATA_RAW / "fortune_1000_us.csv"
    df = pd.read_csv(src)
    df = df.sort_values("rank").head(500).reset_index(drop=True)
    df["normalized_company_name"] = df["company"].map(normalize_company_name)
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df
