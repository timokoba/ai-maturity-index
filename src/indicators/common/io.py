"""Path constants and parquet I/O helpers shared across indicators.

Every artifact is scoped by a firm universe ("fortune500" or "sp500") so
both universes coexist side by side and switching between them is a single
variable in the notebooks:

    data_cache/indicators/<universe>/<indicator>/<step>.parquet
    data_clean/indicators/<universe>/<name>.parquet
    data_clean/ai_maturity_index_<universe>.parquet

The sentence-level FinBERT caches under data_raw/edgar/ are deliberately
NOT universe-scoped: they key on sha256 of the sentence text, so scores
computed for one universe are reused verbatim by the other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT: Path = _project_root()
DATA_RAW: Path = PROJECT_ROOT / "data_raw"
DATA_CLEAN: Path = PROJECT_ROOT / "data_clean"
DATA_CACHE: Path = PROJECT_ROOT / "data_cache"
INDICATORS_DIR: Path = DATA_CLEAN / "indicators"
INDICATORS_CACHE_DIR: Path = DATA_CACHE / "indicators"
EDGAR_CACHE: Path = DATA_RAW / "edgar"

UNIVERSES = ("fortune500", "sp500")


def _check_universe(universe: str) -> str:
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}; expected one of {UNIVERSES}")
    return universe


def index_output_path(universe: str) -> Path:
    return DATA_CLEAN / f"ai_maturity_index_{_check_universe(universe)}.parquet"


def indicator_path(name: str, universe: str) -> Path:
    return INDICATORS_DIR / _check_universe(universe) / f"{name}.parquet"


def write_indicator(df: pd.DataFrame, name: str, universe: str) -> Path:
    out = indicator_path(name, universe)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_indicator(name: str, universe: str) -> pd.DataFrame:
    return pd.read_parquet(indicator_path(name, universe))


def list_indicators(universe: str) -> list[str]:
    base = INDICATORS_DIR / _check_universe(universe)
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.parquet"))


def cache_path(indicator: str, step: str, universe: str) -> Path:
    return INDICATORS_CACHE_DIR / _check_universe(universe) / indicator / f"{step}.parquet"


def load_cached_step(indicator: str, step: str, universe: str) -> pd.DataFrame | None:
    """Return the cached parquet for an indicator's pipeline step,
    or None if the cache file does not exist.
    """
    p = cache_path(indicator, step, universe)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save_cached_step(df: pd.DataFrame, indicator: str, step: str, universe: str) -> Path:
    """Write a DataFrame to the indicator's pipeline cache. Creates
    parent directories as needed. Returns the written path.
    """
    p = cache_path(indicator, step, universe)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def clear_cache(indicator: str, universe: str, step: str | None = None) -> int:
    """Delete cached step file(s) for an indicator within one universe.

    If `step` is given, deletes only that one. If None, deletes every
    cached step for the indicator. Returns the number of files removed.
    """
    if step is not None:
        p = cache_path(indicator, step, universe)
        if p.exists():
            p.unlink()
            return 1
        return 0
    base = INDICATORS_CACHE_DIR / _check_universe(universe) / indicator
    if not base.exists():
        return 0
    files = list(base.glob("*.parquet"))
    for f in files:
        f.unlink()
    return len(files)
