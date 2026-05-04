"""Tier 2: UMAP + HDBSCAN and DTW K-Medoids on ball trajectory.

UMAP gives the best 2D embedding for visual validation.
DTW K-Medoids is time-series aware — ball arc shape differs strongly per Spielzug.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN        # sklearn >= 1.3, no external hdbscan pkg needed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import umap

from experiments.evaluate import ClusterResult, compute_cluster_result
from experiments.plot import plot_scatter_2d, SCENARIO_COLORS


# ── UMAP + HDBSCAN ────────────────────────────────────────────────────────────

def run_umap_hdbscan(
    X: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> ClusterResult:
    """UMAP dimensionality reduction → HDBSCAN density clustering.

    PCA pre-reduction is applied before UMAP:
    - On N=223 (segment): PCA to ~20 dims is fast and improves UMAP quality
    - On N=3700 (window): PCA to 20 dims reduces UMAP from ~10 min to ~30 s
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA pre-reduction: retain 95% variance
    pca = PCA(n_components=0.95, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    n_pca = X_pca.shape[1]

    # UMAP to 2D
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    X_2d = reducer.fit_transform(X_pca)

    # HDBSCAN (sklearn version — no extra package needed on Python 3.13)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = hdb.fit_predict(X_2d)

    n_noise = int((labels == -1).sum())
    result = compute_cluster_result(
        "UMAP+HDBSCAN",
        labels, y_true, X_2d,
        note=f"PCA→{n_pca}d, noise={n_noise}",
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: coloured by true scenario
        plot_scatter_2d(X_2d, y_true,
                        "UMAP — True Scenarios",
                        output_dir / "umap_true_scenarios.png")

        # Plot 2: coloured by HDBSCAN cluster id (integers → convert to string for colours)
        from experiments.evaluate import assign_cluster_names
        cluster_names_map = assign_cluster_names(result)
        named_pred = np.array([cluster_names_map.get(c, "noise") for c in labels])
        plot_scatter_2d(X_2d, named_pred,
                        f"UMAP+HDBSCAN — Cluster Assignments  (ARI={result.ari:.3f})",
                        output_dir / "umap_hdbscan_clusters.png")

    return result


# ── DTW K-Medoids on ball trajectory ─────────────────────────────────────────

def run_dtw_kmedoids(
    trajectories: list[np.ndarray],
    y_true: np.ndarray,
    k: int = 9,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> ClusterResult | None:
    """DTW K-Medoids clustering on ball (x, y) time series.

    Each trajectory is a variable-length array of shape (T, 2).
    tslearn resamples to a fixed length internally when using DTW.

    Why DTW works well here:
    - doppelpass: tight zig-zag arc (8-frame turnaround) → short, fast trajectory
    - kreislaeuferspiel: single large jump forward 5m in 3 frames → spike pattern
    - parallelstos: long straight-forward movement
    - defense plays: slow back-and-forth at similar x positions
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.preprocessing import TimeSeriesScalerMeanVariance
        from tslearn.utils import to_time_series_dataset
    except ImportError as e:
        print(f"  [yellow]tslearn not available ({e}) — skipping DTW clustering[/yellow]")
        return None

    if len(trajectories) == 0:
        return None

    # Resample all trajectories to the median length
    lengths = [len(t) for t in trajectories]
    target_len = int(np.median(lengths))

    resampled = []
    for traj in trajectories:
        if len(traj) == target_len:
            resampled.append(traj)
        else:
            # Linear resampling
            idx_old = np.linspace(0, len(traj) - 1, target_len)
            new_traj = np.stack([
                np.interp(idx_old, np.arange(len(traj)), traj[:, dim])
                for dim in range(traj.shape[1])
            ], axis=1)
            resampled.append(new_traj)

    dataset = to_time_series_dataset(resampled)  # shape (N, T, 2)

    # Normalise each series to zero-mean, unit-variance (per series, not globally)
    scaler = TimeSeriesScalerMeanVariance()
    dataset = scaler.fit_transform(dataset)

    # TimeSeriesKMeans with DTW metric uses DBA (DTW Barycenter Averaging)
    # — effectively DTW K-Means, the standard tslearn approach
    model = TimeSeriesKMeans(
        n_clusters=k,
        metric="dtw",
        random_state=random_state,
        n_init=3,
        verbose=False,
    )
    labels = model.fit_predict(dataset)

    result = compute_cluster_result(
        f"DTW KMeans (k={k})",
        labels, y_true,
        X_scaled=dataset.reshape(len(dataset), -1),  # flattened for silhouette
        note=f"ball traj, resamp→{target_len}f",
    )

    if output_dir is not None:
        # Visualise: reduce medoid positions to 2D with PCA for a spatial plot
        X_flat = dataset.reshape(len(dataset), -1)
        from sklearn.decomposition import PCA as _PCA
        pca2 = _PCA(n_components=2, random_state=random_state)
        X_2d = pca2.fit_transform(X_flat)
        plot_scatter_2d(
            X_2d, y_true,
            "Ball Trajectory PCA — True Scenarios",
            output_dir / "dtw_pca_true_scenarios.png",
        )
        plot_scatter_2d(
            X_2d,
            np.array([f"cluster_{c}" for c in labels]),
            f"DTW K-Medoids  (ARI={result.ari:.3f})",
            output_dir / "dtw_clusters.png",
        )

    return result
