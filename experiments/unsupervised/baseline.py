"""Tier 1: K-Means + GMM baseline clustering with PCA visualisation.

Runs in <30 seconds. No heavy ML dependencies beyond scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from experiments.evaluate import ClusterResult, compute_cluster_result
from experiments.plot import plot_scatter_2d, plot_elbow, plot_ari_by_k, SCENARIO_COLORS


def run_kmeans(
    X: np.ndarray,
    y_true: np.ndarray,
    k_values: list[int] | None = None,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> list[ClusterResult]:
    """K-Means for each k in k_values. Returns one ClusterResult per k."""
    if k_values is None:
        k_values = [5, 7, 9, 11]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results: list[ClusterResult] = []
    inertias: list[float] = []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            n_init=20,          # critical: high n_init for small N (avoids bad inits)
            random_state=random_state,
        )
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        results.append(
            compute_cluster_result(f"K-Means (k={k})", labels, y_true, X_scaled)
        )

    if output_dir is not None:
        plot_elbow(k_values, inertias, output_dir / "kmeans_elbow.png")
        plot_ari_by_k("K-Means", k_values,
                      [r.ari for r in results], output_dir / "kmeans_ari_by_k.png")

    return results


def run_gmm(
    X: np.ndarray,
    y_true: np.ndarray,
    k_values: list[int] | None = None,
    pca_components: int = 15,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> list[ClusterResult]:
    """GMM with PCA pre-reduction (mandatory: 81 features / 9 components = singular covariance).

    Pipeline: StandardScaler → PCA(15) → GMM(covariance_type='diag')
    """
    if k_values is None:
        k_values = [5, 7, 9, 11]

    scaler = StandardScaler()
    pca = PCA(n_components=min(pca_components, X.shape[1] - 1), random_state=random_state)
    X_scaled = scaler.fit_transform(X)
    X_pca = pca.fit_transform(X_scaled)

    explained = float(pca.explained_variance_ratio_.sum())

    results: list[ClusterResult] = []
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",   # 'full' is singular with 81 features, 223 samples
            n_init=5,
            random_state=random_state,
        )
        gmm.fit(X_pca)
        labels = gmm.predict(X_pca)
        results.append(
            compute_cluster_result(
                f"GMM diag (k={k})",
                labels, y_true, X_pca,
                note=f"PCA→{pca_components}d ({explained:.0%} var)",
            )
        )

    return results


def run_pca_scatter(
    X: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
    random_state: int = 42,
) -> np.ndarray:
    """PCA to 2D for visual inspection. Saves two plots: coloured by true label."""
    scaler = StandardScaler()
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(scaler.fit_transform(X))

    ev1, ev2 = pca.explained_variance_ratio_
    title = f"PCA 2D — True Scenarios  (PC1={ev1:.1%}, PC2={ev2:.1%})"
    plot_scatter_2d(X_2d, y_true, title, output_dir / "pca_true_scenarios.png")

    return X_2d
