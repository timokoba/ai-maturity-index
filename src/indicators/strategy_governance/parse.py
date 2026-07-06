"""Extract Item sections from a 10-K filing.

Primary path uses `edgartools`' structured 10-K view, which exposes the
business, risk_factors, and MD&A sections directly. A regex fallback
recovers Item 1, 1A, and 7 from the raw text when the structured view
fails (this happens for ~5-10% of filings with non-standard formatting).

Item 7A (Quantitative and Qualitative Disclosures About Market Risk) is
deliberately excluded: it is dominated by financial-market risk (FX,
interest-rate, commodity exposures) and tabular content rather than
narrative-strategic disclosure, and Loughran-McDonald (2016, JAR) flag
it as unreliable for tone analysis. Babina, Fedyk, He, Hodson (2024)
report effectively zero AI mentions in 7A.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .edgar import ensure_identity

log = logging.getLogger(__name__)


ITEM_PATTERNS = {
    "item_1": re.compile(r"^\s*item\s*1\.?\s*[^a]", re.IGNORECASE | re.MULTILINE),
    "item_1a": re.compile(r"^\s*item\s*1a\.?", re.IGNORECASE | re.MULTILINE),
    "item_7": re.compile(r"^\s*item\s*7\.?\s*[^a]", re.IGNORECASE | re.MULTILINE),
    "item_7a": re.compile(r"^\s*item\s*7a\.?", re.IGNORECASE | re.MULTILINE),
    "item_8": re.compile(r"^\s*item\s*8\.?", re.IGNORECASE | re.MULTILINE),
}

# Item 7A and Item 8 are matched only to serve as closing boundaries for
# Item 7 in the regex fallback; their bodies are not returned.
ITEM_ORDER = ["item_1", "item_1a", "item_7", "item_7a", "item_8"]
RETURN_ITEMS = ("item_1", "item_1a", "item_7")


@dataclass(frozen=True)
class FilingSections:
    accession_number: str
    cik: str
    sections: dict[str, str]


def _regex_split_items(text: str) -> dict[str, str]:
    """Split raw 10-K text into Items 1, 1A, 7 using header regex.

    Items 7A and 8 are matched only to serve as closing boundaries for
    Item 7; their bodies are not returned (Item 7A is excluded
    deliberately, see module docstring).
    """
    matches: dict[str, list[int]] = {}
    for key, pattern in ITEM_PATTERNS.items():
        matches[key] = [m.start() for m in pattern.finditer(text)]
    starts: dict[str, int] = {}
    for key in ITEM_ORDER:
        if matches.get(key):
            starts[key] = matches[key][-1]
    if not starts:
        return {}
    keys_in_order = [k for k in ITEM_ORDER if k in starts]
    sections: dict[str, str] = {}
    for i, key in enumerate(keys_in_order):
        if key not in RETURN_ITEMS:
            continue
        start = starts[key]
        end = starts[keys_in_order[i + 1]] if i + 1 < len(keys_in_order) else len(text)
        body = text[start:end].strip()
        if len(body) >= 200:
            sections[key] = body
    return sections


def extract_items(accession_number: str) -> FilingSections:
    """Return the Item 1, 1A, and 7 sections for a 10-K accession.

    The structured edgartools view is tried first; if any of the three
    sections is missing, the regex fallback fills the gap from the raw
    document text. Item 7A is deliberately excluded (see module
    docstring).
    """
    ensure_identity()
    from edgar import find

    filing = find(accession_number)
    cik = str(getattr(filing, "cik", "")).zfill(10)

    sections: dict[str, str] = {}
    try:
        tenk = filing.obj()
    except Exception as exc:  # noqa: BLE001
        log.warning("structured view unavailable for %s: %s", accession_number, exc)
        tenk = None

    if tenk is not None:
        for src_attr, key in (
            ("business", "item_1"),
            ("risk_factors", "item_1a"),
            ("mdna", "item_7"),
        ):
            text = getattr(tenk, src_attr, None)
            if text and isinstance(text, str) and len(text) > 200:
                sections[key] = text

    missing = [k for k in RETURN_ITEMS if k not in sections]
    if missing:
        try:
            raw = filing.text() if hasattr(filing, "text") else str(filing)
        except Exception as exc:  # noqa: BLE001
            log.warning("raw text unavailable for %s: %s", accession_number, exc)
            raw = ""
        if raw:
            fallback = _regex_split_items(raw)
            for key, body in fallback.items():
                if key in missing:
                    sections[key] = body

    return FilingSections(
        accession_number=accession_number,
        cik=cik,
        sections=sections,
    )
