"""Modeling helpers for the AI Maturity Index.

Skeleton populated as additional indicators come online. The modeling
layer reads the composed index parquet (`data_clean/ai_maturity_index.parquet`)
plus the existing maturity-score columns (`hg_score`, `imd_score`) from
`ai_maturity_master.csv` and trains gradient-boosted models per the
README methodology.
"""

from __future__ import annotations

import pandas as pd

from ..indicators.common.io import DATA_CLEAN, INDEX_OUTPUT


def load_modeling_frame() -> pd.DataFrame:
    """Join the composed index with the master maturity-score table.

    Returns one row per Fortune 500 firm with all indicator features
    plus `hg_score` and `imd_score` as candidate target variables.
    """
    index_df = pd.read_parquet(INDEX_OUTPUT)
    master = pd.read_csv(DATA_CLEAN / "ai_maturity_master.csv")
    keep = ["company_name", "hg_score", "imd_score"]
    master_subset = master[keep].rename(columns={"company_name": "normalized_company_name"})
    return index_df.merge(master_subset, on="normalized_company_name", how="left")


def fit_baseline(frame: pd.DataFrame, target: str = "hg_score"):
    """Fit a baseline LightGBM regressor predicting `target` from the
    indicator feature matrix. Returns the fitted estimator and the
    list of feature columns used.
    """
    raise NotImplementedError(
        "Populate once additional indicators are available; see README."
    )
