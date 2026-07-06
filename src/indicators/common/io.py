"""Path constants and parquet I/O helpers shared across indicators."""

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
INDEX_OUTPUT: Path = DATA_CLEAN / "ai_maturity_index.parquet"


def indicator_path(name: str) -> Path:
    return INDICATORS_DIR / f"{name}.parquet"


def write_indicator(df: pd.DataFrame, name: str) -> Path:
    INDICATORS_DIR.mkdir(parents=True, exist_ok=True)
    out = indicator_path(name)
    df.to_parquet(out, index=False)
    return out


def read_indicator(name: str) -> pd.DataFrame:
    return pd.read_parquet(indicator_path(name))


def list_indicators() -> list[str]:
    if not INDICATORS_DIR.exists():
        return []
    return sorted(p.stem for p in INDICATORS_DIR.glob("*.parquet"))


def cache_path(indicator: str, step: str) -> Path:
    return INDICATORS_CACHE_DIR / indicator / f"{step}.parquet"


def load_cached_step(indicator: str, step: str) -> pd.DataFrame | None:
    """Return the cached parquet for an indicator's pipeline step,
    or None if the cache file does not exist.
    """
    p = cache_path(indicator, step)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def save_cached_step(df: pd.DataFrame, indicator: str, step: str) -> Path:
    """Write a DataFrame to the indicator's pipeline cache. Creates
    parent directories as needed. Returns the written path.
    """
    p = cache_path(indicator, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def clear_cache(indicator: str, step: str | None = None) -> int:
    """Delete cached step file(s) for an indicator.

    If `step` is given, deletes only that one. If None, deletes every
    cached step for the indicator. Returns the number of files removed.
    """
    if step is not None:
        p = cache_path(indicator, step)
        if p.exists():
            p.unlink()
            return 1
        return 0
    base = INDICATORS_CACHE_DIR / indicator
    if not base.exists():
        return 0
    files = list(base.glob("*.parquet"))
    for f in files:
        f.unlink()
    return len(files)
