"""Multivariate analysis of the indicator set (OECD Step 4).

Runs before normalization, on the outlier-treated indicators. Placement is
a matter of methodological fidelity rather than of numbers: min-max is an
affine transform, so Pearson correlations and standardized PCA are
invariant to it (verified to 1e-15 on this data). What does matter is that
the analysis sees the *treated* data, since the square-root transform is
non-linear and shifts correlations by up to 0.19.

The index is a **formative** construct: the two indicators of a dimension
are distinct facets by design (extensive vs. intensive disclosure margin
following Babina et al.), not interchangeable manifestations of one latent
trait. Low within-dimension correlation is therefore expected and is not a
reliability defect. Reflective-model tools (exploratory or confirmatory
factor analysis) are deliberately not used; PCA here serves to check the
dimensional structure and to test for redundancy, not to derive weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DIMENSIONS


def correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairwise-complete Pearson correlations.

    Pairwise rather than complete-case on purpose: complete cases across
    all ten indicators cover only a fifth of the universe (the tone
    indicators are missing wherever a firm has no AI sentences), and that
    subset is selected towards AI-heavy filers.
    """
    return df[cols].corr(method="pearson", min_periods=30)


def within_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the two indicators of each dimension.

    This is the empirical check on averaging a dimension's pair into one
    score. For a formative construct the two facets may legitimately be
    near-uncorrelated; the column is reported so the reader can judge.
    """
    rows: list[dict] = []
    for dim, cols in DIMENSIONS.items():
        pair = df[cols].dropna()
        rows.append(
            {
                "dimension": dim,
                "indicator_a": cols[0],
                "indicator_b": cols[1],
                "r": float(pair[cols[0]].corr(pair[cols[1]])) if len(pair) > 2 else float("nan"),
                "n_pairwise": len(pair),
            }
        )
    return pd.DataFrame(rows).set_index("dimension")


def double_counting_check(corr: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Indicator pairs correlated above `threshold`.

    The OECD handbook warns that combining near-collinear indicators under
    equal weights silently doubles the weight of whatever they jointly
    measure. Anything returned here needs a substantive decision.
    """
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack()
    flagged = pairs[pairs.abs() > threshold]
    return (
        flagged.rename("r")
        .reset_index()
        .rename(columns={"level_0": "indicator_a", "level_1": "indicator_b"})
    )


def kmo_bartlett(df: pd.DataFrame) -> dict:
    """Sampling-adequacy and sphericity tests, the usual PCA prerequisites.

    KMO compares the size of correlations against partial correlations
    (above ~0.6 is commonly considered adequate); Bartlett's test rejects
    the null that the correlation matrix is an identity matrix, i.e. that
    there is nothing to decompose. Computed on complete cases, since both
    need an invertible correlation matrix.
    """
    from scipy import stats

    x = df.dropna()
    n, p = x.shape
    corr = np.corrcoef(x.values, rowvar=False)

    # Bartlett's test of sphericity
    det = np.linalg.det(corr)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    p_value = float(stats.chi2.sf(chi2, dof))

    # KMO from the inverse correlation matrix (anti-image correlations)
    inv = np.linalg.inv(corr)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    corr_off = corr.copy()
    np.fill_diagonal(corr_off, 0.0)
    kmo = float((corr_off ** 2).sum() / ((corr_off ** 2).sum() + (partial ** 2).sum()))

    return {
        "n_complete": int(n),
        "kmo": kmo,
        "bartlett_chi2": float(chi2),
        "bartlett_dof": int(dof),
        "bartlett_p": p_value,
    }


def pca_report(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Standardized PCA. Returns (variance table, loadings, n used).

    Standardization is intrinsic to PCA here, which also makes the result
    invariant to whichever normalization the pipeline applies later.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = df[cols] if cols is not None else df
    x = x.dropna()
    pca = PCA().fit(StandardScaler().fit_transform(x))

    variance = pd.DataFrame(
        {
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        },
        index=[f"PC{i + 1}" for i in range(len(pca.explained_variance_))],
    )
    n_keep = min(3, pca.components_.shape[0])
    loadings = pd.DataFrame(
        pca.components_[:n_keep].T,
        index=x.columns,
        columns=[f"PC{i + 1}" for i in range(n_keep)],
    )
    return variance, loadings, len(x)


def cronbach_alpha(df: pd.DataFrame, standardized: bool = True) -> float:
    """Internal consistency over complete cases.

    The standardized form works off the correlation matrix and is therefore
    scale-free, so it does not depend on where in the pipeline it is
    computed. Alpha assumes a reflective model; for this formative index it
    is reported as a descriptive diagnostic, not as a quantity to maximise.
    """
    x = df.dropna()
    k = x.shape[1]
    if standardized:
        r = x.corr().values
        r_bar = (r.sum() - k) / (k * (k - 1))
        return float(k * r_bar / (1 + (k - 1) * r_bar))
    return float(k / (k - 1) * (1 - x.var(ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def alpha_if_deleted(df: pd.DataFrame) -> pd.Series:
    """Standardized alpha recomputed with each column left out.

    A column whose removal raises alpha is the one least aligned with the
    rest -- informative about the structure, but not on its own a reason to
    drop a theoretically required dimension.
    """
    return pd.Series(
        {c: cronbach_alpha(df.drop(columns=[c])) for c in df.columns},
        name="alpha_if_deleted",
    )


def silhouette_scan(scores: pd.DataFrame, ks: range = range(2, 11), random_state: int = 42) -> pd.DataFrame:
    """Silhouette score and cluster sizes across candidate k.

    Scanned well past the intended range so the choice can be read off a
    curve rather than a single number: silhouette often just grows with k,
    and a solution is only well supported if the score actually peaks
    there. The smallest cluster is reported alongside, since a high score
    bought with a near-empty cluster is not a usable partition.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(scores.dropna())
    rows: list[dict] = []
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(x)
        sizes = np.bincount(labels)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "min_size": int(sizes.min()),
                "sizes": sizes.tolist(),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def cluster_profiles(
    scores: pd.DataFrame, k: int, random_state: int = 42
) -> tuple[pd.Series, pd.DataFrame, float]:
    """k-means over the dimension scores.

    Returns (labels indexed like `scores`, profile table of mean dimension
    scores per cluster in the units of `scores`, silhouette). Firms with a
    missing dimension score are not clustered and get NaN.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    complete = scores.dropna()
    x = StandardScaler().fit_transform(complete)
    fitted = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(x)

    labels = pd.Series(np.nan, index=scores.index, name="cluster")
    labels.loc[complete.index] = fitted.labels_

    profiles = complete.groupby(fitted.labels_).mean()
    profiles.index.name = "cluster"
    # `level` is the cluster's average standing across all dimensions; sorting
    # by it lets the profile table be read as a progression from least to most
    # mature without changing what the clustering found.
    profiles["level"] = profiles.mean(axis=1)
    profiles["n_firms"] = np.bincount(fitted.labels_)
    profiles = profiles.sort_values("level")
    return labels, profiles, float(silhouette_score(x, fitted.labels_))


def label_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """Name each cluster from the dimensions that set it apart.

    Each cluster's dimension scores are compared against the mean profile
    across clusters. A cluster above the mean everywhere is an integrated
    adopter, one below it everywhere is minimally engaged, and the rest are
    named after the dimensions where they stand out most. Reading the
    profile off the data keeps the names honest when k or the underlying
    data changes, instead of hard-coding an expected story.
    """
    dims = [c for c in profiles.columns if c in DIMENSIONS]
    centre = profiles[dims].mean(axis=0)


    names: dict[int, str] = {}
    for cluster in profiles.index:
        above = [d for d in dims if profiles.loc[cluster, d] > centre[d]]
        if len(above) == len(dims):
            names[cluster] = "integrated adopters"
        elif not above:
            names[cluster] = "minimal engagement"
        else:
            margin = (profiles.loc[cluster, above] - centre[above]).sort_values(ascending=False)
            names[cluster] = " + ".join(margin.index[:2]) + " led"
    return names
