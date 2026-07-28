"""Compustat fundamentals loading and the universe -> gvkey match.

`emp` is reported in thousands of employees; `employees_wrds` converts it
to headcount (a reported 0 is kept as 0 so it can be flagged as a zero
denominator downstream). Duplicate fiscal-year rows per gvkey (industrial
vs financial-services format twins, currency variants) are collapsed by
`dedupe_wrds` before any matching.
"""

from __future__ import annotations

import pandas as pd

from ..common.company_ids import normalize_company_name
from ..common.io import DATA_RAW, load_cached_step
from .match import (
    MATCH_AMBIGUOUS,
    MATCH_CIK,
    MATCH_GVKEY,
    MATCH_NAME,
    MATCH_NONE,
    MATCH_TICKER,
    normalize_ticker,
    unambiguous_map,
)

WRDS_FUNDAMENTALS = DATA_RAW / "wrds" / "fundamentals_annual.csv"
TARGET_FYEAR = 2024


def dedupe_wrds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per gvkey: prefer the industrial over the financial-services
    format, USD, standard consolidated data, a populated emp, then the
    largest total assets.
    """
    ranked = df.copy()
    ranked["_pref"] = (
        (ranked["indfmt"] == "INDL").astype(int) * 16
        + (ranked["curcd"] == "USD").astype(int) * 8
        + (ranked["datafmt"] == "STD").astype(int) * 4
        + (ranked["consol"] == "C").astype(int) * 2
        + ranked["emp"].notna().astype(int)
    )
    ranked = ranked.sort_values(["gvkey", "_pref", "at"], ascending=[True, False, False])
    return ranked.drop_duplicates("gvkey", keep="first").drop(columns="_pref")


def load_wrds_fy(fyear: int = TARGET_FYEAR) -> pd.DataFrame:
    """Fundamentals for one fiscal year, one row per gvkey, with the
    normalized identifiers used for matching (norm_ticker,
    normalized_conm, cik as nullable integer).
    """
    df = pd.read_csv(WRDS_FUNDAMENTALS, dtype={"gvkey": str, "cik": str})
    df = df[df["fyear"] == fyear].copy()
    df = dedupe_wrds(df)
    df["norm_ticker"] = df["tic"].map(normalize_ticker)
    df["normalized_conm"] = df["conm"].map(normalize_company_name)
    df["cik_int"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def _universe_ciks(universe_df: pd.DataFrame, universe: str) -> dict:
    """normalized_company_name -> CIK (int), taken from the EDGAR filing
    resolution cached by nlp_features_setup.ipynb for the same universe.
    Empty if that cache is missing; the CIK pass is then simply skipped.
    """
    filings = load_cached_step("nlp_features", "filings", universe)
    if filings is None:
        return {}
    sub = filings[["normalized_company_name", "cik"]].dropna().drop_duplicates()
    sub["cik_int"] = pd.to_numeric(sub["cik"], errors="coerce")
    sub = sub.dropna(subset=["cik_int"])
    names = set(universe_df["normalized_company_name"])
    sub = sub[sub["normalized_company_name"].isin(names)]
    return dict(zip(sub["normalized_company_name"], sub["cik_int"].astype(int)))


def _match_by_gvkey(universe_df: pd.DataFrame, wrds: pd.DataFrame) -> pd.DataFrame:
    """Direct join for universes that already carry a gvkey (e.g. the S&P 500
    constituent list is a Compustat export) — no cascade needed."""
    by_gvkey = wrds.set_index("gvkey")
    rows: list[dict] = []
    for _, firm in universe_df.iterrows():
        gvkey = firm["gvkey"]
        hit = pd.notna(gvkey) and gvkey in by_gvkey.index
        emp = by_gvkey.at[gvkey, "emp"] if hit else float("nan")
        rows.append(
            {
                "normalized_company_name": firm["normalized_company_name"],
                "gvkey": gvkey if hit else None,
                "wrds_conm": by_gvkey.at[gvkey, "conm"] if hit else None,
                "employees_wrds": emp * 1000.0 if pd.notna(emp) else float("nan"),
                "wrds_match_method": MATCH_GVKEY if hit else MATCH_NONE,
            }
        )
    return pd.DataFrame(rows)


def match_to_wrds(
    universe_df: pd.DataFrame,
    wrds: pd.DataFrame | None = None,
    universe: str = "fortune500",
) -> pd.DataFrame:
    """Resolve each universe firm to a Compustat gvkey.

    If the universe frame carries a `gvkey` column the match is a direct
    join (method "gvkey"). Otherwise three ordered passes, first
    unambiguous hit wins: CIK (via the EDGAR filing resolution for the same
    universe), ticker, normalized company name — with the same
    never-silently-pick rule as the ETO match, including the duplicate
    guard: if several firms resolve to the same gvkey, only the strongest
    match method keeps it (cik > ticker > name).

    Returns one row per input firm: normalized_company_name, gvkey
    (nullable), wrds_conm, employees_wrds (headcount; NaN if emp missing),
    wrds_match_method.
    """
    if wrds is None:
        wrds = load_wrds_fy()

    if "gvkey" in universe_df.columns:
        return _match_by_gvkey(universe_df, wrds)

    cik_map, cik_amb = unambiguous_map(wrds, "cik_int", "gvkey")
    tic_map, tic_amb = unambiguous_map(wrds, "norm_ticker", "gvkey")
    name_map, name_amb = unambiguous_map(wrds, "normalized_conm", "gvkey")
    firm_ciks = _universe_ciks(universe_df, universe)

    by_gvkey = wrds.set_index("gvkey")

    rows: list[dict] = []
    for _, firm in universe_df.iterrows():
        norm_name = firm["normalized_company_name"]
        norm_ticker = normalize_ticker(firm["ticker"])
        cik = firm_ciks.get(norm_name)

        gvkey = cik_map.get(cik) if cik is not None else None
        method = MATCH_CIK
        hit_ambiguous = cik is not None and cik in cik_amb
        if gvkey is None:
            gvkey = tic_map.get(norm_ticker)
            method = MATCH_TICKER
            hit_ambiguous = hit_ambiguous or norm_ticker in tic_amb
        if gvkey is None:
            gvkey = name_map.get(norm_name)
            method = MATCH_NAME
            hit_ambiguous = hit_ambiguous or norm_name in name_amb
        if gvkey is None:
            method = MATCH_AMBIGUOUS if hit_ambiguous else MATCH_NONE

        emp = by_gvkey.at[gvkey, "emp"] if gvkey is not None else float("nan")
        rows.append(
            {
                "normalized_company_name": norm_name,
                "gvkey": gvkey,
                "wrds_conm": by_gvkey.at[gvkey, "conm"] if gvkey is not None else None,
                "employees_wrds": emp * 1000.0 if pd.notna(emp) else float("nan"),
                "wrds_match_method": method,
            }
        )

    return _demote_weaker_duplicates(pd.DataFrame(rows))


def _demote_weaker_duplicates(out: pd.DataFrame) -> pd.DataFrame:
    rank = {MATCH_CIK: 0, MATCH_TICKER: 1, MATCH_NAME: 2}
    matched = out[out["gvkey"].notna()]
    for _, grp in matched.groupby("gvkey"):
        if len(grp) == 1:
            continue
        ranks = grp["wrds_match_method"].map(rank)
        losers = grp.index[ranks > ranks.min()] if ranks.nunique() > 1 else grp.index
        out.loc[losers, "gvkey"] = None
        out.loc[losers, "wrds_conm"] = None
        out.loc[losers, "employees_wrds"] = float("nan")
        out.loc[losers, "wrds_match_method"] = MATCH_AMBIGUOUS
    return out
