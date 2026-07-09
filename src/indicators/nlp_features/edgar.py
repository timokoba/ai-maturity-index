"""Resolve the latest 10-K filing for each Fortune 500 ticker from SEC EDGAR.

Uses `edgartools`, which handles ticker -> CIK lookup, respects SEC fair-access
rate limits, and parses iXBRL filings. EDGAR requires a contact email in the
user-agent: set the `EDGAR_IDENTITY` environment variable (e.g. "Name email")
before calling anything here.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from ..common.company_ids import load_fortune500

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


def resolve_fortune500_filings(
    fiscal_year: int | None = None,
    form: str = "10-K",
) -> pd.DataFrame:
    """Look up the most recent (or `fiscal_year`-specific) 10-K for each
    Fortune 500 ticker and return one row per resolved filing with columns
    cik, ticker, company_name, normalized_company_name, accession_number,
    fiscal_year, filing_date, form.

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

    f500 = load_fortune500()
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for _, r in f500.iterrows():
        ticker = r["ticker"]
        if not ticker or ticker.lower() in {"nan", "none", "-"}:
            skipped.append((r["company"], "no ticker"))
            continue
        try:
            company = Company(ticker)
        except Exception as exc:  # noqa: BLE001
            skipped.append((ticker, f"lookup failed: {exc}"))
            continue
        try:
            filings = company.get_filings(form=form)
            if fiscal_year is not None:
                filings = filings.filter(date=f"{fiscal_year}-01-01:{fiscal_year + 1}-06-30")
            exact_form = [x for x in filings if x.form == form]
            if not exact_form:
                skipped.append((ticker, f"no exact-form {form} filings (only amendments)"))
                continue
            f = max(exact_form, key=lambda x: x.filing_date)
        except Exception as exc:  # noqa: BLE001
            skipped.append((ticker, f"filings query failed: {exc}"))
            continue
        rows.append(
            dict(
                cik=str(company.cik).zfill(10),
                ticker=ticker,
                company_name=r["company"],
                normalized_company_name=r["normalized_company_name"],
                accession_number=str(f.accession_number),
                fiscal_year=int(getattr(f, "period_of_report", str(f.filing_date))[:4]),
                filing_date=str(f.filing_date),
                form=str(f.form),
            )
        )

    if skipped:
        log.warning("Skipped %d tickers during resolution", len(skipped))
        for tk, reason in skipped[:25]:
            log.warning("  %s: %s", tk, reason)
    return pd.DataFrame(rows)
