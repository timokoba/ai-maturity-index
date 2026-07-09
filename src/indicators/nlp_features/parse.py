"""Extract Items 1, 1A, and 7 from a 10-K filing.

Each Item is taken from edgartools' structured view (new section parser or legacy
attribute, whichever is longer and valid). For Items the structured view misses,
a title-anchored regex fallback over the raw filing text fills the gap. Every
candidate -- structured or regex -- must survive the same validation: plausible
length, not over-captured into a later Item, content that reads like the Item, and
no overlap with a sibling section. Anything doubtful is dropped, not guessed. Each
returned Item records which parser produced it (`sources`: "edgartools" or
"regex"), so the lower-confidence regex sections can be audited downstream.

Correctness is prioritised over coverage: a missing section is dropped downstream
via `parse_complete`, whereas a section that is present but wrong would corrupt a
firm's counts.

Item 7A is deliberately excluded: it is dominated by tabular market-risk content
rather than narrative disclosure, which Loughran-McDonald (2016, JAR) flag as
unreliable for tone analysis and Babina et al. (2024) find near-empty of AI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .edgar import ensure_identity

log = logging.getLogger(__name__)

RETURN_ITEMS = ("item_1", "item_1a", "item_7")

# Regex fallback (used only for Items the structured view misses). A section is
# recognised by a *title-anchored* header: the Item number followed -- separators
# only, no intervening words -- by the Item's title. That is what separates a real
# heading ("Item 7. Management's Discussion ...") from the cross-references that
# pepper these filings ("Item 7 of the Company's Form 10-K ...", "Item 7 of
# Exhibit 99.2 ..."), which would otherwise be captured as spurious section starts.
# Item 1 also matches the combined "Items 1 and 2. Business and Properties" header
# (incl. "Items 1. and 2.") that oil/gas and pipeline filers use in place of a
# standalone Item 1. The lookaheads keep "1" from matching "1A"/"10" and "7" from
# matching "7A".
_ITEM_NUM = {
    "item_1": r"1(?![0-9a])[.\)]?(?:[\s\xa0]*(?:and|&|,)[\s\xa0]*[2-4][.\)]?){0,2}",
    "item_1a": r"1[\s\xa0]*a(?![0-9a-z])",
    "item_7": r"7(?![0-9a])",
}
# Separators allowed between the Item number and its title: punctuation and long
# whitespace runs only -- never words, so "Item 7 of Exhibit ..." stays excluded.
_HEADER_SEP = r"[.\):;\-–—\s\xa0]"
_ITEM_TITLE = {
    "item_1": r"business",
    "item_1a": r"risk[\s\xa0]+factors?",
    "item_7": r"management|discussion[\s\xa0]+and[\s\xa0]+analysis",
}
# Boundaries that end a section: the next Item's title-anchored header, plus the
# auditor's report and the financial-statement index -- both mark the start of
# Item 8, so they stop Item 7 from running on into the financial statements.
_AUDITOR_HEADING = r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm"
_FIN_STMT_INDEX = r"index\s+to[\s\S]{0,40}?financial\s+statements"
_ITEM_BOUNDARIES = {
    "item_1": [
        r"item[\s\xa0]*1[\s\xa0]*a\b[.\)\s\xa0]{0,40}?risk[\s\xa0]+factors?",
        r"item[\s\xa0]*1[\s\xa0]*b\b[.\)\s\xa0]{0,40}?unresolved",
        r"item[\s\xa0]*2\b[.\)\s\xa0]{0,40}?propert",
    ],
    "item_1a": [
        r"item[\s\xa0]*1[\s\xa0]*b\b[.\)\s\xa0]{0,40}?unresolved",
        r"item[\s\xa0]*1[\s\xa0]*c\b[.\)\s\xa0]{0,40}?cyber",
        r"item[\s\xa0]*2\b[.\)\s\xa0]{0,40}?propert",
    ],
    "item_7": [
        r"item[\s\xa0]*7[\s\xa0]*a\b[.\)\s\xa0]{0,40}?(?:quantitative|market[\s\xa0]+risk)",
        r"item[\s\xa0]*8\b[.\)\s\xa0]{0,40}?financial[\s\xa0]+statements",
        _AUDITOR_HEADING,
        _FIN_STMT_INDEX,
    ],
}
# An Item whose body merely points elsewhere (annual report / proxy / exhibit)
# rather than containing the disclosure; rejected so incorporation stubs are not
# mistaken for a real section.
_INCORP_RE = re.compile(
    r"incorporated\s+(?:herein\s+)?by\s+reference"
    r"|(?:can\s+be\s+found|set\s+forth|is\s+contained|refer(?:\s+to)?|see)\b[\s\S]{0,60}?"
    r"(?:annual\s+report|proxy\s+statement|exhibit\s+13|pages?\s+\d)",
    re.IGNORECASE,
)
# A header line that continues as a reference ("... of this Annual Report on Form
# 10-K", "... in the annual report") is a cross-reference sentence, not the start
# of the section, even though its title sits adjacent to the number. Detected on
# the opening of a candidate so those starts are rejected regardless of length.
_XREF_HEAD_RE = re.compile(
    r"of\s+(?:this|the)\s+[\s\S]{0,20}?(?:annual\s+report|form\s+10-?k|report\b)"
    r"|in\s+the\s+annual\s+report"
    r"|incorporated\s+(?:herein\s+)?by\s+reference",
    re.IGNORECASE,
)
_MIN_BODY_GAP = 200  # a boundary sitting right on the header is a table-of-contents line

# A genuine section runs to thousands of characters; below MIN it is a
# table-of-contents line or an incorporated-by-reference stub, above MAX it is
# an over-capture of the whole document. Both are rejected.
MIN_SECTION_LEN = 500
MAX_SECTION_LEN = 1_000_000


@dataclass(frozen=True)
class FilingSections:
    accession_number: str
    cik: str
    sections: dict[str, str]
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchedFiling:
    """Raw material for one filing, cached so section assembly stays offline.

    `edgar_sections` are the validated edgartools picks; `raw_text` is the full
    filing text (the regex fallback's input), cached for every filing so the
    regex can be re-run over any of them offline. `assemble_sections` turns this
    into the final sections + provenance.
    """

    cik: str
    edgar_sections: dict[str, str]
    raw_text: str


def _valid_len(text: str | None) -> bool:
    return bool(text) and MIN_SECTION_LEN <= len(text) <= MAX_SECTION_LEN


# Over-capture: a section that ran past its boundary into a later Item. The
# auditor's report belongs only to Item 8, so two or more of its headings mean
# the section spilled into the financial statements. An Item 1 that is both
# oversized and restates the MD&A heading has swallowed Item 7 (an embedded
# table of contents lists that heading once, hence the length gate).
_AUDITOR_RE = re.compile(
    r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm", re.IGNORECASE
)
_MDNA_TITLE_RE = re.compile(
    r"management\W{0,4}s?\W{0,5}discussion\s+and\s+analysis\s+of\s+financial\s+condition",
    re.IGNORECASE,
)
_MAX_ITEM_1_LEN = 450_000


def _overcaptured(item: str, text: str) -> bool:
    if len(_AUDITOR_RE.findall(text)) >= 2:
        return True
    return item == "item_1" and len(text) > _MAX_ITEM_1_LEN and len(_MDNA_TITLE_RE.findall(text)) >= 2


# Content check: the section must read like the Item it claims to be. Lenient by
# design -- Item 1A (Risk Factors) is saturated with "risk", Item 7 (MD&A)
# discusses results/liquidity; Item 1 (Business) has no reliable positive marker.
_RISK_HINT_RE = re.compile(r"\brisk", re.IGNORECASE)
_MDNA_HINT_RE = re.compile(
    r"results\s+of\s+operations|liquidity|cash\s+flows?|financial\s+condition", re.IGNORECASE
)


def _looks_like(item: str, text: str) -> bool:
    if item == "item_1a":
        return len(_RISK_HINT_RE.findall(text)) >= 10
    if item == "item_7":
        return bool(_MDNA_HINT_RE.search(text))
    return True


def _containment(shorter: str, longer: str, windows: int = 25) -> float:
    """Fraction of evenly-spaced 200-char windows of `shorter` found verbatim in
    `longer`. A high value means `longer` contains `shorter`, i.e. they overlap."""
    span = len(shorter) - 200
    if span <= windows:
        return 0.0
    step = span // windows
    return sum(shorter[i * step : i * step + 200] in longer for i in range(windows)) / windows


def _drop_overlapping(sections: dict[str, str], sources: dict[str, str]) -> None:
    """Drop any section that largely contains a shorter sibling: the longer one
    over-captured it, so the firm falls out via `parse_complete` rather than
    being kept with corrupted counts. Provenance is dropped alongside."""
    present = [k for k in RETURN_ITEMS if k in sections]
    to_drop: set[str] = set()
    for a in present:
        for b in present:
            if a == b or b in to_drop:
                continue
            if len(sections[a]) < len(sections[b]) and _containment(sections[a], sections[b]) > 0.5:
                to_drop.add(b)
    for k in to_drop:
        sections.pop(k, None)
        sources.pop(k, None)


# The new parser namespaces Items by Part (e.g. "part_ii_item_7"), so we match on
# the suffix. Legacy attributes are a fallback: edgartools is mid-migration and
# each source succeeds on filings the other misses.
_ITEM_SUFFIX = {"1": "item_1", "1a": "item_1a", "7": "item_7"}
_LEGACY_ATTRS = (("business", "item_1"), ("risk_factors", "item_1a"), ("mdna", "item_7"))


def _section_text(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    if callable(text):
        try:
            return text()
        except Exception:  # noqa: BLE001
            return None
    return text if isinstance(text, str) else str(value)


def _newparser_sections(tenk) -> dict[str, str]:
    try:
        secs = tenk.doc.sections
        keys = list(secs.keys())
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, str] = {}
    for k in keys:
        item = _ITEM_SUFFIX.get(k.split("_item_")[-1])
        if item is not None and _valid_len(text := _section_text(secs[k])):
            out[item] = text
    return out


def _legacy_sections(tenk) -> dict[str, str]:
    out: dict[str, str] = {}
    for attr, item in _LEGACY_ATTRS:
        text = getattr(tenk, attr, None)
        if isinstance(text, str) and _valid_len(text):
            out[item] = text
    return out


def _raw_text(filing) -> str:
    try:
        return filing.text() if hasattr(filing, "text") else str(filing)
    except Exception as exc:  # noqa: BLE001
        log.warning("raw text unavailable: %s", exc)
        return ""


def _header_starts(text: str, item: str) -> list[int]:
    """Offsets of every title-anchored header for `item` in the raw text
    (``items?`` also matches the plural "Items 1 and 2 ..." form; the separator
    class tolerates "Item 7:" / "Item 7 -" and long right-aligned whitespace runs
    while still admitting only punctuation between number and title)."""
    pat = re.compile(
        rf"items?[\s\xa0]*{_ITEM_NUM[item]}{_HEADER_SEP}{{0,80}}?(?:{_ITEM_TITLE[item]})",
        re.IGNORECASE,
    )
    return [m.start() for m in pat.finditer(text)]


def _boundary_starts(text: str, item: str) -> list[int]:
    out: list[int] = []
    for pattern in _ITEM_BOUNDARIES[item]:
        out += [m.start() for m in re.finditer(pattern, text, re.IGNORECASE)]
    return out


def _is_incorporation_stub(text: str) -> bool:
    """True when a short body only points to the disclosure elsewhere (annual
    report / proxy / exhibit) instead of containing it."""
    return len(text) < 3000 and bool(_INCORP_RE.search(text[:700]))


def _is_crossref_head(text: str) -> bool:
    """True when a candidate opens as a cross-reference sentence ("... of this
    Annual Report on Form 10-K", "... in the annual report") rather than a real
    section heading, so it is rejected even if it is long."""
    return bool(_XREF_HEAD_RE.search(text[:130]))


def _regex_candidates(text: str, item: str) -> list[str]:
    """Candidate bodies for `item`, longest first: each title-anchored header up
    to the next section boundary. The real section outruns table-of-contents
    entries (tiny bodies), and cross-references never match the title anchor."""
    bounds = _boundary_starts(text, item)
    bodies: list[str] = []
    for start in _header_starts(text, item):
        ends = [b for b in bounds if b > start + _MIN_BODY_GAP]
        end = min(ends) if ends else len(text)
        bodies.append(text[start:end].strip())
    return sorted(set(bodies), key=len, reverse=True)


def _edgartools_sections(tenk) -> dict[str, str]:
    """Longest valid new-parser / legacy candidate per Item."""
    new_secs = _newparser_sections(tenk)
    legacy_secs = _legacy_sections(tenk)
    out: dict[str, str] = {}
    for key in RETURN_ITEMS:
        candidates = [
            s
            for s in (new_secs.get(key), legacy_secs.get(key))
            if s and not _overcaptured(key, s) and _looks_like(key, s)
        ]
        if candidates:
            out[key] = max(candidates, key=len)
    return out


def fetch_filing(accession_number: str) -> FetchedFiling:
    """Download one 10-K's edgartools sections and its full raw text. This is the
    only EDGAR-bound step; the regex fallback and provenance are applied offline
    by `assemble_sections`, so the regex can be re-tuned against any filing from
    the cache without re-fetching.
    """
    ensure_identity()
    from edgar import find

    filing = find(accession_number)
    cik = str(getattr(filing, "cik", "")).zfill(10)

    try:
        tenk = filing.obj()
    except Exception as exc:  # noqa: BLE001
        log.warning("structured view unavailable for %s: %s", accession_number, exc)
        tenk = None

    edgar_sections = _edgartools_sections(tenk) if tenk is not None else {}
    return FetchedFiling(cik, edgar_sections, _raw_text(filing))


def assemble_sections(
    edgar_sections: dict[str, str], raw_text: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Combine edgartools sections with a validated regex fallback and return
    ``(sections, sources)`` with per-Item provenance. Pure and offline, so it can
    be re-run over the cache while iterating on the regex.

    edgartools over-captures are dropped *first*, so an Item that the structured
    view swallowed into a sibling can still be recovered cleanly by the regex
    rather than left missing. The regex then fills any absent Item; its output
    passes the same validation, and a final overlap pass guards it too."""
    sections = dict(edgar_sections)
    sources = {k: "edgartools" for k in edgar_sections}
    _drop_overlapping(sections, sources)

    missing = [k for k in RETURN_ITEMS if k not in sections]
    added = False
    if missing and raw_text:
        for key in missing:
            # longest candidate that reads like the Item, is not over-captured,
            # and is not a mere incorporation-by-reference stub
            for body in _regex_candidates(raw_text, key):
                if (
                    _valid_len(body)
                    and not _overcaptured(key, body)
                    and _looks_like(key, body)
                    and not _is_incorporation_stub(body)
                    and not _is_crossref_head(body)
                ):
                    sections[key] = body
                    sources[key] = "regex"
                    added = True
                    break
    if added:
        _drop_overlapping(sections, sources)

    return sections, sources


def extract_items(accession_number: str) -> FilingSections:
    """Fetch + assemble a filing's Items in one live call (used where an online
    fetch is acceptable, e.g. the validation sample). The cached pipeline instead
    calls `fetch_filing` once and `assemble_sections` locally."""
    fetched = fetch_filing(accession_number)
    sections, sources = assemble_sections(fetched.edgar_sections, fetched.raw_text)
    return FilingSections(accession_number, fetched.cik, sections, sources)
