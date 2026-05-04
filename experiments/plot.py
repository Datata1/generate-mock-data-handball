"""Reusable plotting helpers for clustering experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import pandas as pd

from experiments.features import SCENARIO_ORDER
from experiments.evaluate import ClusterResult, build_confusion_df

# Fixed colour palette — consistent across all plots so the same scenario
# always maps to the same colour regardless of which plot you're reading.
SCENARIO_COLORS: dict[str, str] = {
    "kreuzung":          "#E91E63",
    "rueckpass":         "#9C27B0",
    "doppelpass":        "#3F51B5",
    "parallelstos":      "#2196F3",
    "kreislaeuferspiel": "#FF5722",   # brightest — should be most distinct
    "defense_60":        "#4CAF50",
    "defense_51":        "#8BC34A",
    "defense_42":        "#CDDC39",
    "transition":        "#9E9E9E",
    "noise":             "#212121",
}

plt.rcParams.update({
    "figure.facecolor": "#0D1B2A",
    "axes.facecolor":   "#0D1B2A",
    "axes.edgecolor":   "#444",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#333",
})


def _legend_patches(labels: np.ndarray) -> list:
    seen = dict.fromkeys(labels)  # preserves insertion order, deduplicates
    return [
        mpatches.Patch(color=SCENARIO_COLORS.get(s, "#aaa"), label=s)
        for s in seen
    ]


def plot_scatter_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    alpha: float = 0.75,
    s: int = 40,
) -> None:
    """2D scatter coloured by scenario name or cluster name."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
    colors = [SCENARIO_COLORS.get(str(lb), "#aaa") for lb in labels]
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(handles=_legend_patches(labels), bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8, framealpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_heatmap(result: ClusterResult, output_path: Path, title: str = "") -> None:
    """Row-normalised confusion heatmap: true scenario vs. majority cluster name."""
    df = build_confusion_df(result)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.2), max(5, len(df) * 0.7)), dpi=120)
    sns.heatmap(
        df, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#333",
        ax=ax, cbar_kws={"label": "% of true scenario"},
        annot_kws={"size": 9},
    )
    ax.set_title(title or f"Confusion — {result.method}  (ARI={result.ari:.3f})", fontsize=11)
    ax.set_xlabel("Assigned cluster name (majority vote)")
    ax.set_ylabel("True scenario")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_elbow(k_range, inertias: list[float], output_path: Path) -> None:
    """K-Means elbow curve."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(list(k_range), inertias, marker="o", color="#2196F3", linewidth=2)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=9, color="#FF5722", linestyle="--", alpha=0.6, label="k=9 (true classes)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_ari_by_k(
    method: str,
    k_values: list[int],
    ari_values: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.bar([str(k) for k in k_values], ari_values, color="#2196F3", alpha=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("ARI")
    ax.set_title(f"{method} — ARI by k", fontsize=11)
    ax.axhline(y=0.6, color="#4CAF50", linestyle="--", alpha=0.7, label="ARI=0.60 (excellent)")
    ax.axhline(y=0.3, color="#FF9800", linestyle="--", alpha=0.7, label="ARI=0.30 (meaningful)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
