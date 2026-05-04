"""Tier 3: GCN frame embeddings + K-Means (optional, requires torch + torch-geometric).

This tier documents the infrastructure for using trained GCN embeddings as
cluster features. With UNTRAINED (random) weights the expected ARI ≈ 0 —
this is the correct and honest result. It shows what the pipeline looks like
so it can be re-run once a trained checkpoint is available from wels-monorepo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import ClusterResult, compute_cluster_result

# Path to the wels-monorepo ML source code
_WELS_ML_SRC = Path(__file__).parent.parent.parent / "wels-monorepo" / "packages" / "ml" / "src"


def _check_available() -> bool:
    try:
        import torch
        import torch_geometric
        return True
    except ImportError:
        return False


def run_gcn_kmeans(
    conn,
    y_true: np.ndarray,
    meta: pd.DataFrame,
    k: int = 9,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> ClusterResult | None:
    """Extract GCN frame embeddings → K-Means clustering.

    Uses the wels-monorepo ActionPredictor's encode_frame() method.
    Weights are RANDOM (untrained) unless a checkpoint is explicitly loaded.

    Args:
        conn:   DuckDB connection (read-only)
        y_true: scenario name per segment
        meta:   DataFrame with match_id, start_frame columns
        k:      number of clusters

    Returns:
        ClusterResult or None if torch is not available.
    """
    if not _check_available():
        return None

    import torch
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    if str(_WELS_ML_SRC) not in sys.path:
        sys.path.insert(0, str(_WELS_ML_SRC))

    try:
        from ml.data.features import load_frame_window, open_readonly
        from ml.data.graphs import frame_to_graph
        from ml.models.action import ActionPredictor
    except ImportError as e:
        print(f"  Cannot import wels-monorepo ML: {e}")
        return None

    model = ActionPredictor()
    model.eval()

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for _, row in meta.iterrows():
            match_id = row["match_id"]
            start_frame = int(row["start_frame"])
            # Sample the midpoint frame
            sl_row = conn.execute(
                "SELECT end_frame FROM scenario_labels WHERE match_id=? AND start_frame=?",
                [match_id, start_frame],
            ).fetchone()
            if sl_row is None:
                embeddings.append(np.zeros(128))
                continue

            end_frame = int(sl_row[0])
            mid = (start_frame + end_frame) // 2

            frames = load_frame_window(conn, match_id, center_frame=mid, window=25)
            if not frames:
                embeddings.append(np.zeros(128))
                continue

            frame_embs = []
            for frame in frames:
                graph = frame_to_graph(frame, actor_track_id=4)  # CB as focal player
                if graph.x.shape[0] == 0:
                    continue
                emb = model.encode_frame(graph)           # shape (1, 128)
                frame_embs.append(emb.squeeze(0).numpy())

            if frame_embs:
                embeddings.append(np.stack(frame_embs).mean(axis=0))  # mean-pool
            else:
                embeddings.append(np.zeros(128))

    X_emb = np.stack(embeddings)   # (N, 128)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_emb)

    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    result = compute_cluster_result(
        f"GCN K-Means (k={k})",
        labels, y_true, X_scaled,
        note="UNTRAINED weights — ARI≈0 expected. Re-run with checkpoint for real results.",
    )

    if output_dir is not None:
        from sklearn.decomposition import PCA
        from experiments.plot import plot_scatter_2d
        pca2 = PCA(n_components=2, random_state=random_state)
        X_2d = pca2.fit_transform(X_scaled)
        plot_scatter_2d(X_2d, y_true,
                        "GCN Embeddings PCA (untrained) — True Scenarios",
                        output_dir / "gcn_pca_true_scenarios.png")

    return result
