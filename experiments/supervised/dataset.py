"""Data loading and train/test splitting for supervised scenario classification.

Critical design constraint: ALWAYS split by MATCH, never by frame or segment.
Splitting by frame causes data leakage — training frames are adjacent to test
frames within the same tactical sequence, making the task artificially easy.

With 3 matches and ~74 segments per match, use leave-one-match-out CV:
  Fold 0: train on matches 1+2, test on match 0
  Fold 1: train on matches 0+2, test on match 1
  Fold 2: train on matches 0+1, test on match 2
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

# Import feature extraction from the unsupervised experiments module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.features import build_segment_features, SCENARIO_ORDER


@dataclass
class Fold:
    fold_id: int
    train_matches: list[str]
    test_match: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_folds(
    conn: duckdb.DuckDBPyConnection,
    exclude_transition: bool = False,
) -> list[Fold]:
    """Build leave-one-match-out folds from the database.

    Args:
        conn:               Read-only DuckDB connection.
        exclude_transition: Whether to drop 'transition' segments.
                            Transition is a positional bridge, not a tactical play.
                            Excluding it gives a cleaner 8-class problem.

    Returns:
        List of Fold objects, one per match in the database.
    """
    X, y, meta = build_segment_features(conn)

    if exclude_transition:
        mask = y != "transition"
        X, y, meta = X[mask], y[mask], meta[mask]

    match_ids = meta["match_id"].unique().tolist()

    folds: list[Fold] = []
    for i, test_match in enumerate(match_ids):
        train_matches = [m for m in match_ids if m != test_match]

        train_mask = meta["match_id"].isin(train_matches).values
        test_mask = meta["match_id"] == test_match

        folds.append(Fold(
            fold_id=i,
            train_matches=train_matches,
            test_match=test_match,
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask.values],
            y_test=y[test_mask.values],
        ))

    return folds


def class_distribution(y: np.ndarray) -> pd.DataFrame:
    """Summary of class counts and percentages."""
    labels, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({"scenario": labels, "count": counts})
    df["pct"] = (df["count"] / df["count"].sum() * 100).round(1)
    # Sort by SCENARIO_ORDER
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    df["_ord"] = df["scenario"].map(order).fillna(99)
    return df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
