"""Resolve and download 10-K filings for the Fortune 500 from SEC EDGAR.

Uses the `edgartools` library, which handles ticker -> CIK lookup via
EDGAR's company-tickers JSON, respects SEC fair-access rate limits, and
parses iXBRL filings into navigable section objects.

EDGAR requires a user-agent containing a contact email; set the
`EDGAR_IDENTITY` environment variable (e.g. "Timo Koba kab.timo3@gmail.com")
before calling any function in this module.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from ..common.company_ids import load_fortune500
from ..common.io import EDGAR_CACHE

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
            if len(filings) == 0:
                skipped.append((ticker, f"no {form} filings"))
                continue
            f = filings.latest()
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


def _filing_dir(cik: str, accession: str) -> Path:
    return EDGAR_CACHE / cik / accession.replace("-", "")


def download_filing(cik: str, accession: str) -> Path:
    """Download a single filing into the local cache and return the dir.

    The raw filing JSON (from edgartools' structured representation) is
    written to `<cache>/<cik>/<accession>/filing.json`. The function is
    idempotent: if the file already exists, it is not redownloaded.
    """
    ensure_identity()
    from edgar import find

    out_dir = _filing_dir(cik, accession)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "filing.json"
    if target.exists():
        return out_dir
    filing = find(accession)
    payload = {
        "cik": cik,
        "accession_number": accession,
        "form": str(filing.form),
        "filing_date": str(filing.filing_date),
        "company": str(filing.company),
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return out_dir


def filing_cache_dir(cik: str, accession: str) -> Path:
    return _filing_dir(cik, accession)
