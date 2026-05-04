"""Feature extraction from handball_mock.duckdb for unsupervised ML experiments.

Two granularities:
  - Segment-level (223 samples): one vector per scenario_labels row.
    Primary evaluation surface — ground truth is known.
  - Window-level (~3,700 samples): 25-frame sliding windows with 50% overlap.
    More data for UMAP visualization and DTW clustering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

# Field player track IDs — goalkeepers (7, 14) always have court_x=NULL
FIELD_TRACKS_A = [1, 2, 3, 4, 5, 6]
FIELD_TRACKS_B = [8, 9, 10, 11, 12, 13]
FIELD_TRACKS = FIELD_TRACKS_A + FIELD_TRACKS_B

STAT_COLS = ["mean_x", "std_x", "mean_y", "std_y", "mean_speed", "max_speed"]

SCENARIO_ORDER = [
    "kreuzung", "rueckpass", "doppelpass", "parallelstos", "kreislaeuferspiel",
    "defense_60", "defense_51", "defense_42", "transition",
]


# ── Segment-level features ─────────────────────────────────────────────────────

_SEGMENT_SQL = """
WITH player_stats AS (
    SELECT
        sl.match_id,
        sl.start_frame,
        sl.scenario,
        p.track_id,
        p.team,
        AVG(p.court_x)                                               AS mean_x,
        COALESCE(STDDEV(p.court_x), 0.0)                             AS std_x,
        AVG(p.court_y)                                               AS mean_y,
        COALESCE(STDDEV(p.court_y), 0.0)                             AS std_y,
        AVG(SQRT(p.velocity_x * p.velocity_x + p.velocity_y * p.velocity_y)) AS mean_speed,
        MAX(SQRT(p.velocity_x * p.velocity_x + p.velocity_y * p.velocity_y)) AS max_speed
    FROM scenario_labels sl
    JOIN players p
        ON  p.match_id = sl.match_id
        AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
    WHERE p.court_x IS NOT NULL
    GROUP BY sl.match_id, sl.start_frame, sl.scenario, p.track_id, p.team
),

team_a_centroid AS (
    SELECT sl.match_id, sl.start_frame,
        AVG(p.court_x) AS a_cx, AVG(p.court_y) AS a_cy
    FROM scenario_labels sl
    JOIN players p ON p.match_id = sl.match_id
        AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
        AND p.team = 'A' AND p.court_x IS NOT NULL
    GROUP BY sl.match_id, sl.start_frame
),

team_b_stats AS (
    SELECT sl.match_id, sl.start_frame,
        AVG(p.court_x)      AS b_cx,
        AVG(p.court_y)      AS b_cy,
        VARIANCE(p.court_x) AS b_var_x
    FROM scenario_labels sl
    JOIN players p ON p.match_id = sl.match_id
        AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
        AND p.team = 'B' AND p.court_x IS NOT NULL
    GROUP BY sl.match_id, sl.start_frame
),

ball_imputed AS (
    SELECT sl.match_id, sl.start_frame,
        AVG(COALESCE(b.court_x, pmean.fx)) AS ball_x,
        AVG(COALESCE(b.court_y, pmean.fy)) AS ball_y
    FROM scenario_labels sl
    JOIN frames f ON f.match_id = sl.match_id
        AND f.frame_id BETWEEN sl.start_frame AND sl.end_frame
    LEFT JOIN ball b ON b.match_id = f.match_id AND b.frame_id = f.frame_id
    LEFT JOIN (
        SELECT match_id, frame_id,
               AVG(court_x) AS fx, AVG(court_y) AS fy
        FROM players WHERE court_x IS NOT NULL
        GROUP BY match_id, frame_id
    ) pmean ON pmean.match_id = f.match_id AND pmean.frame_id = f.frame_id
    GROUP BY sl.match_id, sl.start_frame
),

possession AS (
    SELECT sl.match_id, sl.start_frame,
        AVG(CASE WHEN p.team = 'A' AND p.has_ball THEN 1.0 ELSE 0.0 END)
            AS possession_a
    FROM scenario_labels sl
    JOIN players p ON p.match_id = sl.match_id
        AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
    GROUP BY sl.match_id, sl.start_frame
),

track6_x AS (
    SELECT sl.match_id, sl.start_frame,
        AVG(p.court_x) AS t6_mean_x
    FROM scenario_labels sl
    JOIN players p ON p.match_id = sl.match_id
        AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
        AND p.track_id = 6 AND p.court_x IS NOT NULL
    GROUP BY sl.match_id, sl.start_frame
)

SELECT
    ps.match_id, ps.start_frame, ps.scenario,
    ps.track_id, ps.team,
    ps.mean_x, ps.std_x, ps.mean_y, ps.std_y, ps.mean_speed, ps.max_speed,
    tac.a_cx, tac.a_cy,
    tbs.b_cx, tbs.b_cy, tbs.b_var_x,
    bi.ball_x, bi.ball_y,
    po.possession_a,
    t6.t6_mean_x
FROM player_stats ps
JOIN team_a_centroid tac USING (match_id, start_frame)
JOIN team_b_stats    tbs USING (match_id, start_frame)
JOIN ball_imputed    bi  USING (match_id, start_frame)
JOIN possession      po  USING (match_id, start_frame)
JOIN track6_x        t6  USING (match_id, start_frame)
ORDER BY ps.match_id, ps.start_frame, ps.track_id
"""


def build_segment_features(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build one feature vector per scenario_labels row.

    Returns:
        X:    float64 array of shape (N, ~81)
        y:    string array of shape (N,) — scenario names
        meta: DataFrame with match_id, start_frame, scenario columns
    """
    long_df = conn.execute(_SEGMENT_SQL).fetchdf()

    # Pivot: long → wide (one row per segment, columns per track×stat)
    pivot = long_df.pivot_table(
        index=["match_id", "start_frame", "scenario"],
        columns="track_id",
        values=STAT_COLS,
        aggfunc="first",
    )
    # Flatten MultiIndex columns: stat_trackid → e.g. mean_x_1
    pivot.columns = [f"{stat}_{tid}" for stat, tid in pivot.columns]
    pivot = pivot.reset_index()

    # Fill any missing track (shouldn't happen with mock data, safety net)
    stat_cols = [c for c in pivot.columns if any(c.startswith(s) for s in STAT_COLS)]
    pivot[stat_cols] = pivot[stat_cols].fillna(pivot[stat_cols].median())

    # Aggregate columns (same value for every track row — take first occurrence)
    agg = (
        long_df.groupby(["match_id", "start_frame"])
        .first()[["a_cx", "a_cy", "b_cx", "b_cy", "b_var_x",
                  "ball_x", "ball_y", "possession_a", "t6_mean_x"]]
        .reset_index()
    )
    wide = pivot.merge(agg, on=["match_id", "start_frame"])

    meta = wide[["match_id", "start_frame", "scenario"]].copy()
    feature_cols = [c for c in wide.columns
                    if c not in ("match_id", "start_frame", "scenario")]

    X = wide[feature_cols].values.astype(np.float64)
    y = wide["scenario"].values

    return X, y, meta


# ── Ball trajectories for DTW ─────────────────────────────────────────────────

def build_ball_trajectories(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return ball (x, y) time series per scenario segment.

    Returns:
        trajectories: list of arrays, each shape (T, 2) — variable length
        y:            string array of scenario labels, same order
    """
    # Pull per-frame ball position per scenario span; impute NULLs
    sql = """
    SELECT sl.match_id, sl.start_frame, sl.scenario,
           f.frame_id,
           COALESCE(b.court_x, pmean.fx) AS bx,
           COALESCE(b.court_y, pmean.fy) AS by
    FROM scenario_labels sl
    JOIN frames f
        ON f.match_id = sl.match_id
        AND f.frame_id BETWEEN sl.start_frame AND sl.end_frame
    LEFT JOIN ball b
        ON b.match_id = f.match_id AND b.frame_id = f.frame_id
    LEFT JOIN (
        SELECT match_id, frame_id,
               AVG(court_x) AS fx, AVG(court_y) AS fy
        FROM players WHERE court_x IS NOT NULL
        GROUP BY match_id, frame_id
    ) pmean ON pmean.match_id = f.match_id AND pmean.frame_id = f.frame_id
    ORDER BY sl.match_id, sl.start_frame, f.frame_id
    """
    df = conn.execute(sql).fetchdf()

    trajectories: list[np.ndarray] = []
    labels: list[str] = []

    for (match_id, start_frame, scenario), grp in df.groupby(
        ["match_id", "start_frame", "scenario"], sort=False
    ):
        traj = grp[["bx", "by"]].values.astype(np.float64)
        # Linear interpolation for any remaining NaNs
        if np.any(np.isnan(traj)):
            for col in range(traj.shape[1]):
                mask = np.isnan(traj[:, col])
                if mask.any() and not mask.all():
                    idx = np.arange(len(traj))
                    traj[mask, col] = np.interp(idx[mask], idx[~mask], traj[~mask, col])
                elif mask.all():
                    traj[:, col] = 20.0 if col == 0 else 10.0  # court centre fallback
        trajectories.append(traj)
        labels.append(scenario)

    return trajectories, np.array(labels)


# ── Window-level features ─────────────────────────────────────────────────────

def build_window_features(
    conn: duckdb.DuckDBPyConnection,
    window: int = 25,
    step: int = 12,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sliding-window features for larger-N UMAP visualization.

    Returns:
        X:    float64 array of shape (~3700, ~81)
        y:    scenario label for each window (majority scenario)
        meta: DataFrame with match_id, center_frame, scenario
    """
    # Build scenario interval index for labeling windows
    spans_df = conn.execute(
        "SELECT match_id, start_frame, end_frame, scenario FROM scenario_labels"
    ).fetchdf()

    # Get all match frame ranges
    match_ranges = conn.execute(
        "SELECT match_id, MIN(frame_id) AS f0, MAX(frame_id) AS f1 FROM frames GROUP BY match_id"
    ).fetchdf()

    rows = []
    for _, mrow in match_ranges.iterrows():
        mid = mrow["match_id"]
        f0, f1 = int(mrow["f0"]), int(mrow["f1"])
        match_spans = spans_df[spans_df["match_id"] == mid]

        for center in range(f0 + window - 1, f1 + 1, step):
            w_start = center - window + 1
            w_end = center

            # Label = scenario with most overlap in this window
            best_scenario = "transition"
            best_overlap = 0
            for _, sp in match_spans.iterrows():
                overlap = max(0, min(w_end, sp["end_frame"]) - max(w_start, sp["start_frame"]) + 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_scenario = sp["scenario"]

            rows.append({"match_id": mid, "center_frame": center, "scenario": best_scenario})

    window_meta = pd.DataFrame(rows)

    # Compute per-window features using the same SQL pattern
    # Build a temporary table of window definitions for batch querying
    windows_input = window_meta.copy()
    windows_input["start_frame"] = windows_input["center_frame"] - window + 1
    windows_input["end_frame"] = windows_input["center_frame"]

    conn.register("_win_def", windows_input[["match_id", "start_frame", "end_frame", "scenario"]])

    win_sql = """
    WITH player_stats AS (
        SELECT
            w.match_id, w.start_frame, w.scenario, p.track_id, p.team,
            AVG(p.court_x)                                               AS mean_x,
            COALESCE(STDDEV(p.court_x), 0.0)                             AS std_x,
            AVG(p.court_y)                                               AS mean_y,
            COALESCE(STDDEV(p.court_y), 0.0)                             AS std_y,
            AVG(SQRT(p.velocity_x*p.velocity_x + p.velocity_y*p.velocity_y)) AS mean_speed,
            MAX(SQRT(p.velocity_x*p.velocity_x + p.velocity_y*p.velocity_y)) AS max_speed
        FROM _win_def w
        JOIN players p ON p.match_id = w.match_id
            AND p.frame_id BETWEEN w.start_frame AND w.end_frame
            AND p.court_x IS NOT NULL
        GROUP BY w.match_id, w.start_frame, w.scenario, p.track_id, p.team
    )
    SELECT * FROM player_stats ORDER BY match_id, start_frame, track_id
    """
    long_win = conn.execute(win_sql).fetchdf()
    conn.unregister("_win_def")

    if long_win.empty:
        return np.empty((0, 0)), np.array([]), pd.DataFrame()

    pivot = long_win.pivot_table(
        index=["match_id", "start_frame", "scenario"],
        columns="track_id",
        values=STAT_COLS,
        aggfunc="first",
    )
    pivot.columns = [f"{stat}_{tid}" for stat, tid in pivot.columns]
    pivot = pivot.reset_index()

    stat_cols_w = [c for c in pivot.columns if any(c.startswith(s) for s in STAT_COLS)]
    pivot[stat_cols_w] = pivot[stat_cols_w].fillna(pivot[stat_cols_w].median())

    meta_out = pivot[["match_id", "start_frame", "scenario"]].copy()
    meta_out = meta_out.rename(columns={"start_frame": "window_start"})
    feature_cols_w = [c for c in pivot.columns if c not in ("match_id", "start_frame", "scenario")]

    X = pivot[feature_cols_w].values.astype(np.float64)
    y = pivot["scenario"].values

    return X, y, meta_out
