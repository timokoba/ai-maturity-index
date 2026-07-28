"""Self-contained HTML report for the AI Maturity Index ranking.

One table row per ranked firm: composite 0-100, the five dimension scores
(0-100, green-shaded by value, marked with a badge where the dimension
rests on a single indicator), and the number of imputed indicators. The
dimensions themselves are weighted equally in the composite; the coverage
shown here records how much indicator evidence each one rests on, which is
a data-quality note rather than a weight. Firms without a composite (a dimension contributed no
indicator at all) are listed in a second table with the offending
dimensions named, so exclusion is as auditable as inclusion.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DIMENSIONS

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 0; line-height: 1.4; }
.topbar { padding: 12px 24px; border-bottom: 1px solid #8884; }
.topbar h1 { font-size: 1.15rem; margin: 0 0 6px; }
.topbar .meta { color: #888; font-size: 0.85rem; margin: 0 0 8px; }
input#filter { width: 100%; padding: 8px; font-size: 1rem; }
main { padding: 0 12px 40px; overflow-x: auto; }
h2 { font-size: 1rem; padding: 12px 12px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
thead th { position: sticky; top: 0; background: Canvas; border-bottom: 2px solid #888;
           padding: 6px 8px; text-align: right; white-space: nowrap; z-index: 5; }
thead th.left { text-align: left; }
td { padding: 4px 8px; border-bottom: 1px solid #8883; text-align: right; white-space: nowrap; }
td.left { text-align: left; }
td.index { font-weight: 700; }
td.missing { color: #c0392b; font-weight: 700; }
span.half { font-size: 0.7rem; font-weight: 700; color: #b9650f; border: 1px solid #b9650f;
            border-radius: 8px; padding: 0 5px; margin-left: 5px; }
span.cluster { font-size: 0.72rem; color: #468; border: 1px solid #46888; border-radius: 8px;
               padding: 1px 7px; white-space: nowrap; }
tr.firm.expandable { cursor: pointer; }
tr.firm.expandable:hover { background: #8881; }
td.caret { color: #888; width: 1em; }
tr.detail { display: none; }
tr.detail.open { display: table-row; }
tr.detail > td { background: #8880; padding: 10px 14px; }
.dimgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.dimblock { border-left: 3px solid #46886; padding: 2px 0 2px 9px; }
.dimblock .head { font-weight: 700; font-size: 0.8rem; }
.dimblock .sub { color: #888; font-size: 0.74rem; }
.dimblock .ind { display: flex; justify-content: space-between; gap: 8px; font-size: 0.78rem;
                 margin-top: 3px; }
.dimblock .ind .val { font-variant-numeric: tabular-nums; white-space: nowrap; }
.dimblock .ind .rel { color: #888; font-size: 0.72rem; }
.dimblock .ind.dropped { color: #c0392b; }
"""


def _esc(value) -> str:
    return _html.escape(str(value))


def _score_cell(score: float, coverage: float) -> str:
    if pd.isna(score) or coverage == 0:
        return '<td class="missing">—</td>'
    shade = f"background: rgba(39, 134, 42, {0.05 + 0.35 * score / 100:.2f});"
    badge = '<span class="half">½</span>' if coverage <= 0.5 else ""
    return f'<td style="{shade}">{score:.1f}{badge}</td>'


def _cluster_cell(r) -> str:
    label = r.get("cluster_label")
    if label is None or pd.isna(label):
        return '<td class="left">—</td>'
    return f'<td class="left"><span class="cluster">{_esc(label)}</span></td>'


def _indicator_line(label: str, value, reliability) -> str:
    """One indicator inside a dimension block: its normalized score and the
    reliability with which it entered the dimension average."""
    if pd.isna(value):
        return (f'<div class="ind dropped"><span>{_esc(label)}</span>'
                f'<span class="val">missing</span></div>')
    rel = "" if pd.isna(reliability) else f'<span class="rel">r {reliability:.2f}</span>'
    return (f'<div class="ind"><span>{_esc(label)}</span>'
            f'<span class="val">{value:.1f} {rel}</span></div>')


def _detail_row(r, colspan: int, indicators: pd.DataFrame, reliability: pd.DataFrame | None) -> str:
    """Hidden row breaking each dimension down into its two indicators, so a
    dimension score can be traced to what produced it."""
    key = r["normalized_company_name"]
    blocks: list[str] = []
    for dim, cols in DIMENSIONS.items():
        score, cov = r[f"score_{dim}"], r[f"coverage_{dim}"]
        head = f'{dim.capitalize()} — {"—" if pd.isna(score) else f"{score:.1f}"}'
        sub = f"indicator coverage {cov:.2f}" if pd.notna(cov) else ""
        lines = []
        for col in cols:
            value = indicators.at[key, col] if key in indicators.index and col in indicators.columns else np.nan
            w = (
                reliability.at[key, col]
                if reliability is not None and key in reliability.index and col in reliability.columns
                else np.nan
            )
            lines.append(_indicator_line(col.split("__", 1)[1].replace("_", " "), value, w))
        blocks.append(
            f'<div class="dimblock"><div class="head">{_esc(head)}</div>'
            f'<div class="sub">{_esc(sub)}</div>{"".join(lines)}</div>'
        )
    search = " ".join(str(v).lower() for v in (r["ticker"], r["company_name"]) if pd.notna(v))
    return (f'<tr class="detail" data-search="{_esc(search)}">'
            f'<td colspan="{colspan}"><div class="dimgrid">{"".join(blocks)}</div></td></tr>')


def _ranked_row(r, expandable: bool) -> str:
    cells = [f'<td class="caret">{"›" if expandable else ""}</td>'] if expandable else []
    cells += [
        f"<td>{int(r['rank'])}</td>",
        f'<td class="left">{_esc(r["ticker"])}</td>',
        f'<td class="left">{_esc(r["company_name"])}</td>',
        f'<td class="index">{r["index_0_100"]:.1f}</td>',
    ]
    for dim in DIMENSIONS:
        cells.append(_score_cell(r[f"score_{dim}"], r[f"coverage_{dim}"]))
    cells.append(_cluster_cell(r))
    cells.append(f"<td>{int(r['n_imputed'])}</td>")
    search = " ".join(
        str(v).lower() for v in (r["ticker"], r["company_name"], r.get("cluster_label")) if pd.notna(v)
    )
    cls = "firm expandable" if expandable else "firm"
    return f'<tr class="{cls}" data-search="{_esc(search)}">' + "".join(cells) + "</tr>"


def _excluded_row(r, expandable: bool) -> str:
    empty_dims = [d for d in DIMENSIONS if r[f"coverage_{d}"] == 0]
    cells = [f'<td class="caret">{"›" if expandable else ""}</td>'] if expandable else []
    cells += [
        f'<td class="left">{_esc(r["ticker"])}</td>',
        f'<td class="left">{_esc(r["company_name"])}</td>',
        f'<td class="left missing">{", ".join(empty_dims)}</td>',
    ]
    for dim in DIMENSIONS:
        cells.append(_score_cell(r[f"score_{dim}"] , r[f"coverage_{dim}"]))
    search = f'{r["ticker"]} {r["company_name"]}'.lower()
    cls = "firm expandable" if expandable else "firm"
    return f'<tr class="{cls}" data-search="{_esc(search)}">' + "".join(cells) + "</tr>"


_JS = (
    # filtering only sets display; a detail row falls back to its CSS default
    # (hidden) unless it carries the open class, so the two never fight
    "const f=document.getElementById('filter');"
    "f.addEventListener('input',()=>{const q=f.value.toLowerCase();"
    "document.querySelectorAll('tbody tr').forEach(tr=>{"
    "tr.style.display=tr.dataset.search.includes(q)?'':'none';});});"
    "document.querySelectorAll('tr.firm.expandable').forEach(tr=>{"
    "tr.addEventListener('click',()=>{const d=tr.nextElementSibling;"
    "if(!d||!d.classList.contains('detail'))return;"
    "const open=d.classList.toggle('open');"
    "const c=tr.querySelector('td.caret');if(c)c.textContent=open?'⌄':'›';});});"
)


def build_index_report(
    result: pd.DataFrame,
    out_path: str | Path,
    indicators: pd.DataFrame | None = None,
    reliability: pd.DataFrame | None = None,
) -> Path:
    """Write the ranking report and return its path.

    `result` needs: normalized_company_name, rank, ticker, company_name,
    index_0_100, score_<dim> (0-100), coverage_<dim>, n_imputed -- one row
    per firm, with NaN index/rank for firms failing the availability rule.
    `cluster_label` is optional; when present it is shown as a badge and is
    searchable alongside ticker and company name.

    Pass `indicators` (normalized indicator values on a 0-100 scale, indexed
    by normalized_company_name) to make every row expandable into a
    per-dimension breakdown of the two indicators behind its score, with
    `reliability` supplying the weight each one carried inside its
    dimension.
    """
    ranked = result[result["index_0_100"].notna()].sort_values("rank")
    excluded = result[result["index_0_100"].isna()].sort_values("ticker")
    expandable = indicators is not None

    caret_head = '<th class="caret"></th>' if expandable else ""
    dim_headers = "".join(f"<th>{_esc(d.capitalize())}</th>" for d in DIMENSIONS)
    ranked_head = (
        f'<tr>{caret_head}<th>Rank</th><th class="left">Ticker</th><th class="left">Company</th>'
        f'<th>Index</th>{dim_headers}<th class="left">Cluster</th><th>Imputed</th></tr>'
    )
    excluded_head = (
        f'<tr>{caret_head}<th class="left">Ticker</th><th class="left">Company</th>'
        f'<th class="left">Empty dimensions</th>{dim_headers}</tr>'
    )
    n_ranked_cols = (1 if expandable else 0) + 4 + len(DIMENSIONS) + 2
    n_excluded_cols = (1 if expandable else 0) + 3 + len(DIMENSIONS)

    def body(rows, render, colspan) -> str:
        out = []
        for _, r in rows.iterrows():
            out.append(render(r, expandable))
            if expandable:
                out.append(_detail_row(r, colspan, indicators, reliability))
        return "".join(out)

    meta = (
        f"{len(ranked)} ranked firms · {len(excluded)} without a composite "
        f"(a dimension contributed no indicator). Dimension scores 0-100; "
        f"<span class='half'>½</span> marks a dimension resting on one of its two "
        f"indicators. The five dimensions are weighted equally in the composite; "
        f"coverage is reported as a data-quality note, not a weight. "
        f"'Imputed' counts structural-zero imputations."
        + (" Click any row to expand the two indicators behind each dimension, "
           "with the reliability each one carried." if expandable else "")
    )
    html_doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>AI Maturity Index — {len(ranked)} firms</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"<div class='topbar'><h1>AI Maturity Index</h1>"
        f"<p class='meta'>{meta}</p>"
        f"<input id='filter' placeholder='Filter by ticker or company name…'></div>"
        f"<main><table><thead>{ranked_head}</thead><tbody>"
        + body(ranked, _ranked_row, n_ranked_cols)
        + f"</tbody></table>"
        f"<h2>Without composite ({len(excluded)})</h2>"
        f"<table><thead>{excluded_head}</thead><tbody>"
        + body(excluded, _excluded_row, n_excluded_cols)
        + f"</tbody></table></main><script>{_JS}</script></body></html>"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
