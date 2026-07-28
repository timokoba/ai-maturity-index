"""Firm-universe loaders.

Both universes expose the same core schema so the whole pipeline is
universe-agnostic: `rank`, `company`, `ticker`, `normalized_company_name`,
plus `gvkey` and `cik` where the source provides them.

- "fortune500": top 500 of the Fortune 1000 US list (rank = revenue rank).
  Privately held firms carry ticker "~"; the EDGAR resolution and the
  structured matching drop them naturally.
- "sp500": S&P 500 constituents from `data_raw/base/s&p_500.csv` (Compustat
  export with gvkey/tic). Multi-share-class firms appear once per ticker
  and are deduplicated on gvkey, keeping the first row (the A class).
  `rank` is just the file's alphabetical position, not an economic rank.
"""

from __future__ import annotations

import pandas as pd

from .company_ids import load_fortune500, normalize_company_name
from .io import DATA_RAW, UNIVERSES

__all__ = ["UNIVERSES", "load_universe"]


def _load_sp500() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "base" / "s&p_500.csv", dtype={"gvkey": str})
    df = df.drop_duplicates("gvkey").reset_index(drop=True)
    df["ticker"] = df["tic"].astype(str).str.strip().str.upper()
    # Strip a leading article before normalizing ("The Home Depot, Inc." must
    # meet PARAT's "Home Depot"). Loader-local on purpose: changing
    # normalize_company_name itself would break the join keys already stored
    # in the fortune500 caches.
    company = df["companyname"].astype(str).str.replace(r"^[Tt]he\s+", "", regex=True)
    df["company"] = company
    df["normalized_company_name"] = company.map(normalize_company_name)
    df["rank"] = range(1, len(df) + 1)
    df["cik"] = _cik_by_gvkey().reindex(df["gvkey"]).to_numpy()
    return df[["rank", "company", "ticker", "normalized_company_name", "gvkey", "cik"]]


def _cik_by_gvkey() -> pd.Series:
    """gvkey -> CIK from the Compustat fundamentals file (latest fiscal year
    with a populated CIK per gvkey)."""
    w = pd.read_csv(
        DATA_RAW / "wrds" / "fundamentals_annual.csv",
        usecols=["gvkey", "fyear", "cik"],
        dtype={"gvkey": str, "cik": str},
    )
    w = w.dropna(subset=["cik"]).sort_values("fyear")
    return w.drop_duplicates("gvkey", keep="last").set_index("gvkey")["cik"]


def load_universe(name: str) -> pd.DataFrame:
    if name == "fortune500":
        return load_fortune500()
    if name == "sp500":
        return _load_sp500()
    raise ValueError(f"unknown universe {name!r}; expected one of {UNIVERSES}")
