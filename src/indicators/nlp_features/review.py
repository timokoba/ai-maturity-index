"""Build a static HTML document for manual validation of parsed AI sentences.

One self-contained page listing every parse-complete firm and, per Item, the
AI-relevant sentences in document order (runs of non-AI sentences are elided with
a count). Each Item header is tagged with the parser that produced the section;
regex-sourced Items are flagged red so their lower-confidence extraction can be
eyeballed and dropped by hand. The page is fully expanded (no click-to-open) with
a sticky ticker / company-name filter.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

from .aggregate import ALL_ITEMS, MIN_ITEM_SENTENCES_FOR_PARSE

ITEM_LABEL = {
    "item_1": "Item 1 — Business (Strategy)",
    "item_1a": "Item 1A — Risk Factors (Governance)",
    "item_7": "Item 7 — MD&A (Operations)",
}

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0; line-height: 1.5; }
.topbar { position: sticky; top: 0; background: Canvas; border-bottom: 1px solid #8884; padding: 12px 24px; z-index: 10; }
.topbar h1 { font-size: 1.15rem; margin: 0 0 6px; }
.topbar .meta { color: #888; font-size: 0.85rem; margin: 0 0 8px; }
input#filter { width: 100%; padding: 8px; font-size: 1rem; }
.jumplist { margin-top: 8px; max-height: 90px; overflow-y: auto; font-size: 0.82rem; }
.jumplist a { margin-right: 10px; white-space: nowrap; }
main { max-width: 1100px; margin: 0 auto; padding: 8px 24px 40px; }
section.firm { border: 1px solid #8883; border-radius: 6px; margin: 14px 0; padding: 12px 16px; }
section.firm h2 { font-size: 1.05rem; margin: 0 0 8px; }
section.firm h2 a.edgar { font-size: 0.78rem; font-weight: normal; margin-left: 8px; }
div.itemblock { margin: 10px 0 16px 4px; border-left: 3px solid #6669; padding-left: 12px; }
div.itemblock h3 { font-size: 0.92rem; color: #468; margin: 0 0 4px; font-weight: 600; }
span.badge { font-size: 0.72rem; font-weight: 600; padding: 1px 7px; border-radius: 10px; margin-left: 8px; vertical-align: middle; }
span.src-edgartools { color: #6b7280; border: 1px solid #6b728077; }
span.src-regex { color: #c0392b; border: 1px solid #c0392b; font-weight: 700; }
ol.sentlist { margin: 4px 0; padding-left: 1.6em; }
ol.sentlist li { margin: 3px 0; }
.idx { color: #999; font-size: 0.8em; }
.gap-note { color: #999; font-style: italic; font-size: 0.85em; }
.none { color: #999; font-style: italic; }
"""


def _esc(value) -> str:
    return _html.escape(str(value))


def _kept_firms(filings: pd.DataFrame, sentence_totals: pd.DataFrame) -> pd.DataFrame:
    """Parse-complete firms (every Item clears the sentence gate), joined to
    filing metadata and sorted by ticker."""
    count_cols = [f"n_sentences_{i}" for i in ALL_ITEMS]
    complete = sentence_totals[count_cols].ge(MIN_ITEM_SENTENCES_FOR_PARSE).all(axis=1)
    kept = sentence_totals[complete].drop(columns=["cik"], errors="ignore")
    return kept.merge(filings, on="accession_number", how="left").sort_values("ticker").reset_index(drop=True)


def _sentence_list(sub: pd.DataFrame) -> str:
    lis: list[str] = []
    prev_idx: int | None = None
    for _, r in sub.sort_values("sentence_idx").iterrows():
        idx = int(r["sentence_idx"])
        if prev_idx is not None and idx - prev_idx > 1:
            lis.append(f'<li class="gap-note">… {idx - prev_idx - 1} non-AI sentence(s) omitted …</li>')
        lis.append(f'<li><span class="idx">[{idx}]</span> {_esc(r["sentence"])}</li>')
        prev_idx = idx
    return '<ol class="sentlist">' + "".join(lis) + "</ol>"


def _source_badge(source: str | None) -> str:
    if not source:
        return ""
    cls = "src-regex" if source == "regex" else "src-edgartools"
    return f'<span class="badge {cls}">{_esc(source)}</span>'


def _firm_section(row, per_firm_item, source_map) -> str:
    acc, tk, name, cik = row["accession_number"], row["ticker"], row["company_name"], row["cik"]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
    out = [
        f'<section class="firm" id="firm-{_esc(tk)}" data-search="{_esc(tk).lower()} {_esc(name).lower()}">',
        f'<h2>{_esc(tk)} — {_esc(name)} <a class="edgar" href="{url}" target="_blank">[EDGAR]</a></h2>',
    ]
    for item in ALL_ITEMS:
        sub = per_firm_item.get((acc, item))
        n = 0 if sub is None else len(sub)
        badge = _source_badge(source_map.get((acc, item)))
        out.append(f'<div class="itemblock"><h3>{ITEM_LABEL[item]} — {n} AI sentence(s){badge}</h3>')
        out.append(_sentence_list(sub) if sub is not None and len(sub) else '<p class="none">none</p>')
        out.append("</div>")
    out.append("</section>")
    return "".join(out)


def build_ai_sentence_review(
    sentences: pd.DataFrame,
    filings: pd.DataFrame,
    sentence_totals: pd.DataFrame,
    sections: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Write the AI-sentence review page and return its path.

    `sections` supplies the per-Item parser provenance (columns
    ``accession_number, item, source``) used to badge each Item block.
    """
    kept = _kept_firms(filings, sentence_totals)
    per_firm_item = {
        (acc, item): grp for (acc, item), grp in sentences.groupby(["accession_number", "item"])
    }
    source_map = {
        (r["accession_number"], r["item"]): r["source"] for _, r in sections.iterrows()
    }

    jump = "".join(f'<a href="#firm-{_esc(r.ticker)}">{_esc(r.ticker)}</a>' for _, r in kept.iterrows())
    body = "".join(_firm_section(r, per_firm_item, source_map) for _, r in kept.iterrows())
    n_regex = sum(1 for v in source_map.values() if v == "regex")
    meta = (
        f"{len(kept)} parse-complete firms · AI-relevant sentences only "
        f"(index = position among all cleaned sentences; gaps = non-AI text omitted). "
        f"<b style='color:#c0392b'>{n_regex} section(s) parsed by regex are flagged red</b> — verify these."
    )
    filter_js = (
        "const f=document.getElementById('filter');"
        "f.addEventListener('input',()=>{const q=f.value.toLowerCase();"
        "document.querySelectorAll('section.firm').forEach(d=>{"
        "d.style.display=d.dataset.search.includes(q)?'':'none';});});"
    )
    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Parse review — {len(kept)} firms (AI sentences)</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<div class='topbar'><h1>Parse review — AI sentences</h1>"
        f"<p class='meta'>{meta}</p>"
        f"<input id='filter' placeholder='Filter by ticker or company name…'>"
        f"<div class='jumplist'>{jump}</div></div>"
        f"<main>{body}</main><script>{filter_js}</script></body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
