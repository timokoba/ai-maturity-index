"""ETO PARAT loaders and the universe -> PARAT company match.

PARAT metrics in `core.csv` are company-level aggregates (publications and
patents are lifetime totals; workforce counts are a ~2024 LinkedIn-based
snapshot). `ticker.csv` bridges PARAT IDs to stock tickers, `alias.csv`
supplies alternate and subsidiary names. `id .csv` (PermID / LinkedIn /
regex identifiers) is deliberately unused: tickers plus normalized names
and aliases resolve the Fortune universe, and those identifiers have no
counterpart in the project's other sources.
"""

from __future__ import annotations

import pandas as pd

from ..common.company_ids import normalize_company_name
from ..common.io import DATA_RAW
from .match import (
    MATCH_ALIAS,
    MATCH_AMBIGUOUS,
    MATCH_NAME,
    MATCH_NONE,
    MATCH_TICKER,
    normalize_ticker,
    unambiguous_map,
)

ETO_DIR = DATA_RAW / "eto"

# Verified corporate renames PARAT has not caught up with: the PARAT record
# (old name / old ticker) is the same legal entity as today's constituent,
# so the automatic passes cannot see the link. Keyed by
# normalized_company_name; every entry was checked by hand.
MANUAL_MATCHES = {
    # Everest Group, Ltd. (EG) was Everest Re Group (RE) until 2023; PARAT
    # still lists it as "Everest Re" with the old ticker.
    "everest": 2312,
}

ETO_CORE_COLS = {
    "ID": "eto_id",
    "Name": "eto_name",
    "PARAT link": "parat_link",
    "Publications: AI publications": "ai_publications",
    "Publications: Total publications": "total_publications",
    "Publications: AI publication percentage": "eto_ai_publication_pct",
    "Patents: AI patents": "ai_patents",
    "Patents: Total patents": "total_patents",
    "Patents: AI patent percentage": "eto_ai_patent_pct",
    "Workforce: AI workers": "ai_workers",
    "Workforce: Tech Team 1 workers": "tech_team1_workers",
}


def load_eto_core() -> pd.DataFrame:
    df = pd.read_csv(ETO_DIR / "core.csv", usecols=list(ETO_CORE_COLS))
    df = df.rename(columns=ETO_CORE_COLS)
    df["eto_id"] = df["eto_id"].astype(int)
    return df


def load_eto_tickers() -> pd.DataFrame:
    df = pd.read_csv(ETO_DIR / "ticker.csv")
    df = df.rename(columns={"ID": "eto_id"})
    df["norm_ticker"] = df["Ticker"].map(normalize_ticker)
    return df


def load_eto_aliases() -> pd.DataFrame:
    df = pd.read_csv(ETO_DIR / "alias.csv")
    df = df.rename(columns={"ID": "eto_id"})
    df["normalized_alias"] = df["Alias"].map(normalize_company_name)
    return df


def match_to_eto(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve each universe firm to a PARAT company ID.

    Three ordered passes, first unambiguous hit wins: ticker, normalized
    company name, normalized alias. Keys that map to more than one PARAT
    ID within a pass fall through to the next; a firm whose every
    candidate key was ambiguous ends up `ambiguous` (reported, never
    silently picked), otherwise `unmatched`. Candidate maps are restricted
    to IDs present in `core.csv`, since only those carry metrics.

    When several Fortune firms resolve to the same PARAT ID, only the one
    with the strongest match method keeps it (ticker > name > alias); the
    others are demoted to `ambiguous`. PARAT aliases include former names
    (e.g. Leidos was "Science Applications International Corporation"
    before the 2013 SAIC spin-off), so a weaker duplicate is more likely a
    historical-name collision than a real subsidiary aggregation.

    Corporate-action successors stay unmatched on purpose: entities like
    FedEx Freight, Honeywell Aerospace, Paramount Skydance, Qnity, Sandisk,
    Smurfit Westrock, or Expand Energy postdate PARAT's snapshot, and the
    PARAT record of their predecessor (FedEx, Honeywell International,
    Paramount Global, ...) describes a different corporate scope — matching
    them would attribute the old conglomerate's AI activity to the
    spun-off/merged entity.

    Returns one row per input firm: normalized_company_name, eto_id
    (nullable Int64), eto_name, eto_match_method.
    """
    core = load_eto_core()
    core_ids = set(core["eto_id"])

    tickers = load_eto_tickers()
    tickers = tickers[tickers["eto_id"].isin(core_ids)]
    ticker_map, ticker_amb = unambiguous_map(tickers, "norm_ticker", "eto_id")

    names = core[["eto_id", "eto_name"]].copy()
    names["normalized_name"] = names["eto_name"].map(normalize_company_name)
    name_map, name_amb = unambiguous_map(names, "normalized_name", "eto_id")

    aliases = load_eto_aliases()
    aliases = aliases[aliases["eto_id"].isin(core_ids)]
    alias_map, alias_amb = unambiguous_map(aliases, "normalized_alias", "eto_id")

    id_to_name = dict(zip(core["eto_id"], core["eto_name"]))

    rows: list[dict] = []
    for _, firm in universe_df.iterrows():
        norm_ticker = normalize_ticker(firm["ticker"])
        norm_name = firm["normalized_company_name"]

        eto_id = MANUAL_MATCHES.get(norm_name)
        method = "manual"
        hit_ambiguous = False
        if eto_id is None:
            eto_id = ticker_map.get(norm_ticker)
            method = MATCH_TICKER
            hit_ambiguous = norm_ticker in ticker_amb
        if eto_id is None:
            eto_id = name_map.get(norm_name)
            method = MATCH_NAME
            hit_ambiguous = hit_ambiguous or norm_name in name_amb
        if eto_id is None:
            eto_id = alias_map.get(norm_name)
            method = MATCH_ALIAS
            hit_ambiguous = hit_ambiguous or norm_name in alias_amb
        if eto_id is None:
            method = MATCH_AMBIGUOUS if hit_ambiguous else MATCH_NONE

        rows.append(
            {
                "normalized_company_name": norm_name,
                "eto_id": eto_id,
                "eto_name": id_to_name.get(eto_id),
                "eto_match_method": method,
            }
        )

    out = pd.DataFrame(rows)
    out["eto_id"] = out["eto_id"].astype("Int64")
    return _demote_weaker_duplicates(out)


def _demote_weaker_duplicates(out: pd.DataFrame) -> pd.DataFrame:
    rank = {"manual": -1, MATCH_TICKER: 0, MATCH_NAME: 1, MATCH_ALIAS: 2}
    matched = out[out["eto_id"].notna()]
    for _, grp in matched.groupby("eto_id"):
        if len(grp) == 1:
            continue
        ranks = grp["eto_match_method"].map(rank)
        losers = grp.index[ranks > ranks.min()] if ranks.nunique() > 1 else grp.index
        out.loc[losers, "eto_id"] = pd.NA
        out.loc[losers, "eto_name"] = None
        out.loc[losers, "eto_match_method"] = MATCH_AMBIGUOUS
    return out
