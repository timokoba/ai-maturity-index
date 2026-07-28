"""Build a static HTML document for manual review of the structured inputs.

One self-contained page with a single table, one row per listed universe
firm: match provenance for ETO PARAT and Compustat, every raw input behind
the four Technology / People indicators, the computed shares, and the
firm's complete / missing status per dimension. Missing inputs are flagged
red, zero denominators are flagged red showing the literal 0, and worker
shares above 1 are flagged orange. Nothing is dropped: firms with missing
inputs keep their indicator rows with NaN features, and this page is where
those NaNs can be traced back to the input that caused them. A sticky
header, a text filter, and complete/missing toggles support scanning the
full universe.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import pandas as pd

from .features import PEOPLE_FEATURES, TECHNOLOGY_FEATURES

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0; line-height: 1.4; }
.topbar { padding: 12px 24px; border-bottom: 1px solid #8884; }
.topbar h1 { font-size: 1.15rem; margin: 0 0 6px; }
.topbar .meta { color: #888; font-size: 0.85rem; margin: 0 0 8px; }
.topbar .controls { display: flex; gap: 8px; align-items: center; }
input#filter { flex: 1; padding: 8px; font-size: 1rem; }
.controls button { padding: 7px 12px; font-size: 0.85rem; cursor: pointer; }
.controls button.active { font-weight: 700; outline: 2px solid #468; }
main { padding: 0 12px 40px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
thead th { position: sticky; top: 0; background: Canvas; border-bottom: 2px solid #888;
           padding: 6px 8px; text-align: right; white-space: nowrap; z-index: 5; }
thead th.left { text-align: left; }
td { padding: 3px 8px; border-bottom: 1px solid #8883; text-align: right; white-space: nowrap; }
td.left { text-align: left; }
td.missing { background: #c0392b2e; color: #c0392b; font-weight: 600; }
td.zero-denom { background: #c0392b2e; color: #c0392b; font-weight: 700; }
td.warn { background: #e67e222e; color: #b9650f; font-weight: 600; }
td.complete { color: #27862a; font-weight: 600; }
td.incomplete { color: #c0392b; font-weight: 700; }
span.badge { font-size: 0.7rem; font-weight: 600; padding: 1px 7px; border-radius: 10px; margin-right: 6px; }
span.m-cik, span.m-ticker { color: #6b7280; border: 1px solid #6b728077; }
span.m-name, span.m-alias, span.m-manual { color: #468; border: 1px solid #46888; }
span.m-ambiguous, span.m-unmatched { color: #c0392b; border: 1px solid #c0392b; font-weight: 700; }
"""


def _esc(value) -> str:
    return _html.escape(str(value))


def _int_cell(value, zero_is_denom: bool = False) -> str:
    if pd.isna(value):
        return '<td class="missing">—</td>'
    if zero_is_denom and value == 0:
        return f'<td class="zero-denom">{int(value):,}</td>'
    return f"<td>{int(value):,}</td>"


def _share_cell(value, warn_above_one: bool = False) -> str:
    if pd.isna(value):
        return '<td class="missing">—</td>'
    cls = ' class="warn"' if warn_above_one and value > 1 else ""
    return f"<td{cls}>{value:.4f}</td>"


def _match_cell(method, label, link=None) -> str:
    badge = f'<span class="badge m-{_esc(method)}">{_esc(method)}</span>'
    if label is None or pd.isna(label):
        return f'<td class="left missing">{badge}</td>'
    name = _esc(label)
    if link is not None and not pd.isna(link):
        name = f'<a href="{_esc(link)}" target="_blank">{name}</a>'
    return f'<td class="left">{badge}{name}</td>'


def _status_cell(complete: bool) -> str:
    return '<td class="complete">complete</td>' if complete else '<td class="incomplete">missing</td>'


def _firm_row(r) -> str:
    search = " ".join(
        str(v).lower() for v in (r["ticker"], r["company_name"], r["eto_name"], r["wrds_conm"]) if pd.notna(v)
    )
    tech = "complete" if r["technology_complete"] else "missing"
    people = "complete" if r["people_complete"] else "missing"
    cells = [
        f"<td>{int(r['rank'])}</td>",
        f'<td class="left">{_esc(r["ticker"])}</td>',
        f'<td class="left">{_esc(r["company_name"])}</td>',
        _match_cell(r["eto_match_method"], r["eto_name"], r.get("parat_link")),
        _int_cell(r["ai_publications"]),
        _int_cell(r["total_publications"], zero_is_denom=True),
        _int_cell(r["ai_patents"]),
        _int_cell(r["total_patents"], zero_is_denom=True),
        _int_cell(r["ai_workers"]),
        _int_cell(r["tech_team1_workers"]),
        _match_cell(r["wrds_match_method"], r["wrds_conm"]),
        _int_cell(r["employees_wrds"], zero_is_denom=True),
        _share_cell(r["ai_publication_share"]),
        _share_cell(r["ai_patent_share"]),
        _share_cell(r["tech_team1_worker_share"], warn_above_one=True),
        _share_cell(r["ai_worker_share"], warn_above_one=True),
        _status_cell(r["technology_complete"]),
        _status_cell(r["people_complete"]),
    ]
    return (
        f'<tr data-search="{_esc(search)}" data-tech="{tech}" data-people="{people}">'
        + "".join(cells)
        + "</tr>"
    )


_HEADERS = (
    '<tr><th>Rank</th><th class="left">Ticker</th><th class="left">Company</th>'
    '<th class="left">ETO match</th>'
    "<th>AI pubs</th><th>Total pubs</th><th>AI patents</th><th>Total patents</th>"
    "<th>AI workers</th><th>TT1 workers</th>"
    '<th class="left">WRDS match (FY2024)</th><th>Employees</th>'
    "<th>AI pub share</th><th>AI patent share</th><th>TT1 share</th><th>AI worker share</th>"
    "<th>Technology</th><th>People</th></tr>"
)

_FILTER_JS = (
    "const f=document.getElementById('filter');let mode='all';"
    "function apply(){const q=f.value.toLowerCase();"
    "document.querySelectorAll('tbody tr').forEach(tr=>{"
    "const okQ=tr.dataset.search.includes(q);"
    "const okM=mode==='all'||(mode==='tech'&&tr.dataset.tech==='missing')"
    "||(mode==='people'&&tr.dataset.people==='missing');"
    "tr.style.display=okQ&&okM?'':'none';});}"
    "f.addEventListener('input',apply);"
    "document.querySelectorAll('.controls button').forEach(b=>{"
    "b.addEventListener('click',()=>{mode=b.dataset.mode;"
    "document.querySelectorAll('.controls button').forEach(x=>x.classList.toggle('active',x===b));"
    "apply();});});"
)


def build_structured_inputs_review(inputs: pd.DataFrame, out_path: str | Path) -> Path:
    """Write the structured-inputs review page and return its path.

    `inputs` is the frame produced by `features.build_inputs` (one row per
    listed Fortune 500 firm, already sorted by Fortune rank).
    """
    n = len(inputs)
    n_eto = int(inputs["eto_id"].notna().sum())
    n_wrds = int(inputs["gvkey"].notna().sum())
    n_tech = int(inputs["technology_complete"].sum())
    n_people = int(inputs["people_complete"].sum())
    n_warn = int(
        ((inputs[PEOPLE_FEATURES] > 1).any(axis=1)).sum()
    )

    meta = (
        f"{n} listed universe firms · {n_eto} matched to ETO PARAT · "
        f"{n_wrds} matched to Compustat FY2024 · "
        f"complete: {n_tech} Technology / {n_people} People. "
        f"<b style='color:#c0392b'>Red cells are missing inputs or zero denominators</b> "
        f"(the affected indicators are NaN — marked, not dropped); "
        f"<b style='color:#b9650f'>orange marks worker shares above 1</b> "
        f"({n_warn} firm(s); ETO snapshot vs. consolidated headcount)."
    )
    body = "".join(_firm_row(r) for _, r in inputs.iterrows())
    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Structured inputs review — {n} firms</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<div class='topbar'><h1>Structured inputs review — Technology &amp; People</h1>"
        f"<p class='meta'>{meta}</p>"
        f"<div class='controls'>"
        f"<input id='filter' placeholder='Filter by ticker or company name…'>"
        f"<button data-mode='all' class='active'>All</button>"
        f"<button data-mode='tech'>Missing: Technology</button>"
        f"<button data-mode='people'>Missing: People</button>"
        f"</div></div>"
        f"<main><table><thead>{_HEADERS}</thead><tbody>{body}</tbody></table></main>"
        f"<script>{_FILTER_JS}</script></body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
