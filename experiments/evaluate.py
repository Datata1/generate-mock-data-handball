"""Clustering evaluation: ARI, NMI, silhouette, confusion matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from rich.console import Console
from rich.table import Table

from experiments.features import SCENARIO_ORDER


@dataclass
class ClusterResult:
    method: str
    n_clusters: int
    ari: float
    nmi: float
    silhouette: float
    cluster_labels: np.ndarray     # integer cluster assignments (length N)
    true_labels: np.ndarray        # scenario name strings (length N)
    note: str = ""                 # optional annotation (e.g. "untrained weights")
    # Exclude arrays from JSON serialisation
    _exclude_from_json: list = field(default_factory=lambda: ["cluster_labels", "true_labels"])

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "ari": round(self.ari, 4),
            "nmi": round(self.nmi, 4),
            "silhouette": round(self.silhouette, 4),
            "note": self.note,
        }


def compute_cluster_result(
    method: str,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    X_scaled: np.ndarray,
    note: str = "",
) -> ClusterResult:
    """Compute ARI, NMI, silhouette and return a ClusterResult.

    Handles HDBSCAN noise points (label == -1) by excluding them from
    ARI/NMI but including them in silhouette if X_scaled is provided.
    """
    mask = cluster_labels != -1
    n_noise = int((~mask).sum())
    n_clusters = len(set(cluster_labels[mask])) if mask.any() else 0

    ari = adjusted_rand_score(true_labels[mask], cluster_labels[mask]) if mask.sum() > 1 else 0.0
    nmi = normalized_mutual_info_score(true_labels[mask], cluster_labels[mask]) if mask.sum() > 1 else 0.0

    if n_clusters >= 2 and X_scaled is not None and mask.sum() > n_clusters:
        sil = float(silhouette_score(X_scaled[mask], cluster_labels[mask]))
    else:
        sil = float("nan")

    result = ClusterResult(
        method=method,
        n_clusters=n_clusters,
        ari=ari,
        nmi=nmi,
        silhouette=sil,
        cluster_labels=cluster_labels,
        true_labels=true_labels,
        note=f"{n_noise} noise pts. {note}".strip(". ") if n_noise else note,
    )
    return result


def assign_cluster_names(result: ClusterResult) -> dict[int, str]:
    """Map each cluster integer to its majority true-label scenario."""
    mapping: dict[int, str] = {}
    for cid in set(result.cluster_labels):
        if cid == -1:
            mapping[cid] = "noise"
            continue
        mask = result.cluster_labels == cid
        scenarios, counts = np.unique(result.true_labels[mask], return_counts=True)
        mapping[cid] = str(scenarios[counts.argmax()])
    return mapping


def build_confusion_df(result: ClusterResult) -> pd.DataFrame:
    """Confusion matrix as DataFrame: rows=true scenario, cols=cluster majority name.

    Values are row-normalised percentages (% of true scenario in each cluster).
    """
    cluster_names = assign_cluster_names(result)
    named_pred = np.array([cluster_names.get(c, "noise") for c in result.cluster_labels])

    all_true = [s for s in SCENARIO_ORDER if s in result.true_labels]
    all_pred = sorted(set(named_pred))

    rows = []
    for ts in all_true:
        mask = result.true_labels == ts
        if not mask.any():
            continue
        row = {}
        for ps in all_pred:
            row[ps] = int((named_pred[mask] == ps).sum())
        rows.append({"true_scenario": ts, **row})

    df = pd.DataFrame(rows).set_index("true_scenario").fillna(0).astype(int)
    # Row-normalise to percentages
    row_sums = df.sum(axis=1).replace(0, 1)
    return (df.div(row_sums, axis=0) * 100).round(1)


def print_metrics_table(results: list[ClusterResult], console: Console, title: str = "") -> None:
    table = Table(title=title or "Clustering Results", show_header=True, header_style="bold cyan")
    table.add_column("Method", style="cyan")
    table.add_column("k", justify="right")
    table.add_column("ARI", justify="right")
    table.add_column("NMI", justify="right")
    table.add_column("Silhouette", justify="right")
    table.add_column("Note", style="dim")

    for r in sorted(results, key=lambda x: -x.ari):
        ari_s = (
            f"[green]{r.ari:.3f}[/green]" if r.ari > 0.6
            else f"[yellow]{r.ari:.3f}[/yellow]" if r.ari > 0.3
            else f"[red]{r.ari:.3f}[/red]"
        )
        sil_s = f"{r.silhouette:.3f}" if not (isinstance(r.silhouette, float) and np.isnan(r.silhouette)) else "—"
        table.add_row(r.method, str(r.n_clusters), ari_s, f"{r.nmi:.3f}", sil_s, r.note)

    console.print(table)
    console.print(
        "  [dim]ARI: >0.60 excellent · 0.30–0.60 meaningful · <0.30 near-random[/dim]"
    )


def save_metrics_json(results: list[ClusterResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    output_path.write_text(json.dumps(data, indent=2))
