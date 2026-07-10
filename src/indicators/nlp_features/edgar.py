"""Resolve one 10-K filing per universe firm from SEC EDGAR.

By default takes each firm's most recently filed 10-K. Pass `fiscal_year` to
instead pin every firm to the 10-K that *reports on* that fiscal year, so the
cross-section is comparable across firms with different fiscal-year ends
(e.g. a January-ending filer's FY2025 vs. a December-ending filer's FY2025)
rather than mixing whatever each firm happened to file most recently.

Uses `edgartools`, which handles ticker -> CIK lookup, respects SEC fair-access
rate limits, and parses iXBRL filings. EDGAR requires a contact email in the
user-agent: set the `EDGAR_IDENTITY` environment variable (e.g. "Name email")
before calling anything here.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def ensure_identity() -> None:
    """Configure the edgartools client with the SEC-required identity string.

    Reads the `EDGAR_IDENTITY` environment variable (e.g. "Name email") and
    forwards it to `edgar.set_identity`. Public because it is also called
    from sibling modules (e.g. `parse.py`).
    """
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise RuntimeError(
            "Set the EDGAR_IDENTITY environment variable to a 'Name email' "
            "string before calling EDGAR. Required by SEC fair-access policy."
        )
    from edgar import set_identity

    set_identity(identity)


def _filing_fiscal_year(filing) -> int | None:
    """Fiscal year a filing reports on, taken from its `period_of_report`
    (e.g. '2025-01-31' -> 2025). Returns None when EDGAR supplied no
    period-of-report, so callers can decide how to handle it rather than
    silently falling back to the filing date (which, for most firms, lands
    in the calendar year *after* the fiscal year and would mislabel it).
    """
    por = getattr(filing, "period_of_report", None)
    if not por:
        return None
    try:
        return int(str(por)[:4])
    except (ValueError, TypeError):
        return None


def resolve_filings(
    universe: pd.DataFrame,
    fiscal_year: int | None = None,
    form: str = "10-K",
) -> pd.DataFrame:
    """Look up one 10-K per universe firm and return one row per
    resolved filing with columns cik, ticker, company_name,
    normalized_company_name, accession_number, fiscal_year, filing_date, form.

    `universe` needs `company`, `ticker`, `normalized_company_name`; when a
    `cik` column is present and populated the EDGAR lookup goes through the
    CIK, which is robust against Compustat ticker spellings (e.g. "FDXF")
    that EDGAR's ticker table does not know. Ticker lookup is the fallback.

    With `fiscal_year=None` (default) each firm's most recently filed 10-K is
    taken. With `fiscal_year` set, only 10-Ks whose `period_of_report` falls in
    that fiscal year are considered, matched on the same year derivation that
    populates the `fiscal_year` column so the filter criterion and the stored
    value never disagree. Firms with no 10-K reporting on that year (e.g. a
    deregistered filer whose last 10-K predates it) are dropped and logged.

    Foreign filers without a 10-K (e.g. firms that file 20-F) are dropped
    and logged. Tickers that EDGAR cannot resolve are also dropped.

    EDGAR's `form="10-K"` filter also returns `10-K/A` amendments, and an
    amendment frequently only restates a narrow part of the filing (e.g. Part
    III director/compensation data) without Items 1, 1A, or 7. An amendment
    can also be filed after the original, so picking the single most recent
    filing without excluding `/A` forms risks silently selecting a document
    that lacks the sections this pipeline needs. Amendments are therefore
    filtered out before taking the latest.
    """
    ensure_identity()
    from edgar import Company

    has_cik = "cik" in universe.columns
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for _, r in universe.iterrows():
        ticker = r["ticker"]
        cik = r["cik"] if has_cik and pd.notna(r["cik"]) else None
        if cik is None and (not ticker or ticker.lower() in {"nan", "none", "-", "~"}):
            skipped.append((r["company"], "no ticker"))
            continue
        try:
            company = Company(int(cik)) if cik is not None else Company(ticker)
        except Exception as exc:  # noqa: BLE001
            skipped.append((ticker, f"lookup failed: {exc}"))
            continue
        try:
            filings = company.get_filings(form=form)
            exact_form = [x for x in filings if x.form == form]
            if not exact_form:
                skipped.append((ticker, f"no exact-form {form} filings (only amendments)"))
                continue
            if fiscal_year is not None:
                exact_form = [x for x in exact_form if _filing_fiscal_year(x) == fiscal_year]
                if not exact_form:
                    skipped.append((ticker, f"no {form} reporting on fiscal year {fiscal_year}"))
                    continue
            f = max(exact_form, key=lambda x: x.filing_date)
        except Exception as exc:  # noqa: BLE001
            skipped.append((ticker, f"filings query failed: {exc}"))
            continue
        fy = _filing_fiscal_year(f)
        rows.append(
            dict(
                cik=str(company.cik).zfill(10),
                ticker=ticker,
                company_name=r["company"],
                normalized_company_name=r["normalized_company_name"],
                accession_number=str(f.accession_number),
                fiscal_year=fy if fy is not None else int(str(f.filing_date)[:4]),
                filing_date=str(f.filing_date),
                form=str(f.form),
            )
        )

    if skipped:
        log.warning("Skipped %d tickers during resolution", len(skipped))
        for tk, reason in skipped[:25]:
            log.warning("  %s: %s", tk, reason)
    return pd.DataFrame(rows)
