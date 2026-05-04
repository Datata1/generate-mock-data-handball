"""DuckDB writer — creates all 8 tables and writes mock match data.

Schema SQL is inlined verbatim from:
  wels-monorepo/packages/ingestion/src/ingestion/storage/schema.py
  wels-monorepo/packages/ml/src/ml/storage/schema.py

Formation and possession logic is inlined to keep this project standalone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .labeler import extract_labels
from .types import MatchFrame

# ── Schema DDL ─────────────────────────────────────────────────────────────────

_INGESTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id       TEXT PRIMARY KEY,
    video_path     TEXT NOT NULL,
    fps            DOUBLE NOT NULL,
    total_frames   INTEGER NOT NULL,
    team_a_name    TEXT,
    team_b_name    TEXT,
    ingested_at    TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS frames (
    match_id       TEXT NOT NULL,
    frame_id       INTEGER NOT NULL,
    timestamp_s    DOUBLE NOT NULL,
    player_count   INTEGER NOT NULL,
    on_court_count INTEGER NOT NULL,
    PRIMARY KEY (match_id, frame_id)
);
CREATE TABLE IF NOT EXISTS players (
    match_id       TEXT NOT NULL,
    frame_id       INTEGER NOT NULL,
    track_id       INTEGER NOT NULL,
    team           TEXT NOT NULL,
    court_x        DOUBLE,
    court_y        DOUBLE,
    pixel_foot_x   DOUBLE NOT NULL,
    pixel_foot_y   DOUBLE NOT NULL,
    velocity_x     DOUBLE NOT NULL DEFAULT 0,
    velocity_y     DOUBLE NOT NULL DEFAULT 0,
    confidence     DOUBLE NOT NULL,
    on_court       BOOLEAN NOT NULL DEFAULT TRUE,
    has_ball       BOOLEAN NOT NULL DEFAULT FALSE,
    bbox_x1        INTEGER NOT NULL,
    bbox_y1        INTEGER NOT NULL,
    bbox_x2        INTEGER NOT NULL,
    bbox_y2        INTEGER NOT NULL,
    PRIMARY KEY (match_id, frame_id, track_id)
);
CREATE TABLE IF NOT EXISTS ball (
    match_id       TEXT NOT NULL,
    frame_id       INTEGER NOT NULL,
    court_x        DOUBLE,
    court_y        DOUBLE,
    pixel_x        DOUBLE NOT NULL,
    pixel_y        DOUBLE NOT NULL,
    confidence     DOUBLE NOT NULL,
    bbox_x1        INTEGER NOT NULL,
    bbox_y1        INTEGER NOT NULL,
    bbox_x2        INTEGER NOT NULL,
    bbox_y2        INTEGER NOT NULL,
    PRIMARY KEY (match_id, frame_id)
);
CREATE TABLE IF NOT EXISTS action_labels (
    match_id       TEXT NOT NULL,
    frame_id       INTEGER NOT NULL,
    track_id       INTEGER NOT NULL,
    action         TEXT NOT NULL,
    annotator      TEXT NOT NULL DEFAULT 'manual',
    PRIMARY KEY (match_id, frame_id, track_id)
);

-- Ground-truth Spielzug labels — primary table for supervised ML on play patterns.
-- Values: kreuzung | rueckpass | doppelpass | parallelstos |
--         kreislaeuferspiel | defense_60 | defense_51 | defense_42 | transition
CREATE TABLE IF NOT EXISTS scenario_labels (
    match_id      TEXT    NOT NULL,
    start_frame   INTEGER NOT NULL,
    end_frame     INTEGER NOT NULL,
    scenario      TEXT    NOT NULL,
    duration_s    DOUBLE  GENERATED ALWAYS AS
                    (CAST(end_frame - start_frame AS DOUBLE) / 25.0),
    PRIMARY KEY (match_id, start_frame)
);

CREATE INDEX IF NOT EXISTS idx_players_match_frame ON players (match_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_ball_match_frame    ON ball    (match_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_labels_action       ON action_labels (action);
CREATE INDEX IF NOT EXISTS idx_scenario_match      ON scenario_labels (match_id, scenario);
"""

_ML_SCHEMA = """
CREATE TABLE IF NOT EXISTS action_predictions (
    match_id         TEXT    NOT NULL,
    frame_id         INTEGER NOT NULL,
    track_id         INTEGER NOT NULL,
    pass_prob        DOUBLE  NOT NULL,
    shot_prob        DOUBLE  NOT NULL,
    dribble_prob     DOUBLE  NOT NULL,
    hold_prob        DOUBLE  NOT NULL,
    predicted_action TEXT    NOT NULL,
    PRIMARY KEY (match_id, frame_id, track_id)
);
CREATE TABLE IF NOT EXISTS formations (
    match_id  TEXT    NOT NULL,
    frame_id  INTEGER NOT NULL,
    team      TEXT    NOT NULL,
    formation TEXT    NOT NULL,
    PRIMARY KEY (match_id, frame_id, team)
);
CREATE TABLE IF NOT EXISTS possession_phases (
    match_id      TEXT    NOT NULL,
    phase_id      INTEGER NOT NULL,
    team          TEXT    NOT NULL,
    start_frame   INTEGER NOT NULL,
    end_frame     INTEGER NOT NULL,
    start_time_s  DOUBLE  NOT NULL,
    end_time_s    DOUBLE  NOT NULL,
    duration_s    DOUBLE  GENERATED ALWAYS AS (end_time_s - start_time_s),
    PRIMARY KEY (match_id, phase_id)
);
CREATE INDEX IF NOT EXISTS idx_action_pred_match_frame ON action_predictions (match_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_formations_match_frame  ON formations (match_id, frame_id);
CREATE INDEX IF NOT EXISTS idx_possession_match        ON possession_phases (match_id, start_frame);
"""

# ── Formation logic (inlined from ml/analysis/formation.py) ───────────────────

_DEF_DEPTH = 14.0
_OFF_DEPTH = 14.0
_MIN_PLAYERS = 4
_COMPACT_Y = 10.0


def _classify_formation(
    positions: list[tuple[float, float]],
    defending_left: bool,
) -> str:
    mapped = [(x, y) for x, y in positions if x is not None and y is not None]
    if len(mapped) < _MIN_PLAYERS:
        return "unknown"

    own_x = 0.0 if defending_left else 40.0
    opp_x = 40.0 if defending_left else 0.0

    def dist_own(x: float) -> float:
        return abs(x - own_x)

    def dist_opp(x: float) -> float:
        return abs(x - opp_x)

    n = len(mapped)
    in_own_half = sum(1 for x, _ in mapped if dist_own(x) < 20.0)
    in_def_zone = sum(1 for x, _ in mapped if dist_own(x) <= _DEF_DEPTH)
    in_att_zone = sum(1 for x, _ in mapped if dist_opp(x) <= _OFF_DEPTH)

    if in_att_zone >= n - 1:
        return "attack"
    if in_own_half < n - 2:
        return "transition"
    if in_def_zone < _MIN_PLAYERS:
        return "transition"

    ys = [y for x, y in mapped if dist_own(x) <= _DEF_DEPTH]
    y_spread = max(ys) - min(ys) if len(ys) >= 2 else 0.0
    xs = sorted(dist_own(x) for x, _ in mapped if dist_own(x) <= _DEF_DEPTH)
    if len(xs) < 2:
        return "unknown"
    depth_gap = max(xs[i + 1] - xs[i] for i in range(len(xs) - 1))

    if y_spread <= _COMPACT_Y and depth_gap <= 2.0:
        return "6-0"
    elif depth_gap >= 3.0 and len(xs) >= 5:
        return "5-1"
    else:
        return "4-2"


# ── Possession logic (inlined from ml/analysis/possession.py) ─────────────────

@dataclass
class _Phase:
    phase_id: int
    team: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float


def _detect_phases(
    seq: list[dict],
    min_duration_s: float = 1.5,
    gap_tolerance_s: float = 1.0,
    fps: float = 25.0,
) -> list[_Phase]:
    if not seq:
        return []
    gap_frames = int(gap_tolerance_s * fps)
    min_frames = int(min_duration_s * fps)

    merged: list[tuple[str, int, int, float, float]] = []
    current_team: str | None = None
    run_start_frame = run_start_time = 0
    last_frame = last_time = 0.0

    for entry in seq:
        fid = entry["frame_id"]
        ts = entry["timestamp_s"]
        team = entry.get("team")

        if team in ("A", "B"):
            if team != current_team:
                if current_team is not None:
                    merged.append((current_team, run_start_frame, int(last_frame), run_start_time, float(last_time)))
                current_team = team
                run_start_frame = fid
                run_start_time = ts
            last_frame = fid
            last_time = ts
        else:
            if current_team is not None and fid - int(last_frame) > gap_frames:
                merged.append((current_team, run_start_frame, int(last_frame), run_start_time, float(last_time)))
                current_team = None

    if current_team is not None:
        merged.append((current_team, run_start_frame, int(last_frame), run_start_time, float(last_time)))

    phases = []
    pid = 0
    for team, sf, ef, st, et in merged:
        if (ef - sf) >= min_frames:
            phases.append(_Phase(pid, team, sf, ef, st, et))
            pid += 1
    return phases


# ── Mock softmax helper ────────────────────────────────────────────────────────

_ACTION_ORDER = ["pass", "shot", "dribble", "hold"]


def _mock_softmax(true_action: str, rng: np.random.Generator) -> tuple[float, float, float, float]:
    logits = rng.normal(0, 1, 4).astype(float)
    idx = _ACTION_ORDER.index(true_action) if true_action in _ACTION_ORDER else 3
    logits[idx] += 3.0
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])


# ── Writer ─────────────────────────────────────────────────────────────────────

class MockDataWriter:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        self.conn.execute(_INGESTION_SCHEMA)
        self.conn.execute(_ML_SCHEMA)
        self._rng = np.random.default_rng(0)

    def write_match(
        self,
        match_id: str,
        frames: list[MatchFrame],
        fps: float,
        team_a: str,
        team_b: str,
        scenario_spans: list[tuple[int, int, str]] | None = None,
    ) -> None:
        total = len(frames)

        # Single transaction for the entire match — critical for performance
        self.conn.execute("BEGIN")

        # matches
        self.conn.execute(
            "INSERT INTO matches VALUES (?,?,?,?,?,?,current_timestamp)",
            [match_id, f"mock://{match_id}.mp4", fps, total, team_a, team_b],
        )

        # Pre-extract labels once (before batching)
        labels = extract_labels(frames)

        # frames + players + ball in batches of 500
        frame_buf, player_buf, ball_buf = [], [], []
        label_buf, pred_buf = [], []

        for mf in frames:
            frame_buf.append([
                match_id, mf.frame_id, mf.timestamp_s,
                len(mf.players),
                sum(1 for p in mf.players if p.on_court),
            ])

            for p in mf.players:
                player_buf.append([
                    match_id, mf.frame_id, p.track_id, p.team,
                    p.court_x, p.court_y,
                    p.pixel_foot_x, p.pixel_foot_y,
                    p.velocity_x, p.velocity_y,
                    p.confidence, p.on_court, p.has_ball,
                    p.bbox_x1, p.bbox_y1, p.bbox_x2, p.bbox_y2,
                ])
                key = (mf.frame_id, p.track_id)
                if key in labels:
                    action = labels[key]
                    label_buf.append([match_id, mf.frame_id, p.track_id, action, "mock_generator"])
                    pp, sp, dp, hp = _mock_softmax(action, self._rng)
                    predicted = _ACTION_ORDER[int(np.argmax([pp, sp, dp, hp]))]
                    pred_buf.append([match_id, mf.frame_id, p.track_id, pp, sp, dp, hp, predicted])

            if mf.ball is not None:
                b = mf.ball
                ball_buf.append([
                    match_id, mf.frame_id,
                    b.court_x, b.court_y,
                    b.pixel_x, b.pixel_y,
                    b.confidence,
                    b.bbox_x1, b.bbox_y1, b.bbox_x2, b.bbox_y2,
                ])

            if len(frame_buf) >= 2000:
                self._flush(frame_buf, player_buf, ball_buf)
                frame_buf, player_buf, ball_buf = [], [], []

        self._flush(frame_buf, player_buf, ball_buf)
        self._flush_labels(label_buf, pred_buf)

        # formations — sampled every 5 frames
        form_buf = []
        for mf in frames:
            if mf.frame_id % 5 != 0:
                continue
            for team in ("A", "B"):
                positions = [
                    (p.court_x, p.court_y)
                    for p in mf.players
                    if p.team == team and p.court_x is not None
                ]
                # Team A defends left (own_goal=0), Team B defends right (own_goal=40)
                defending_left = team == "A"
                formation = _classify_formation(positions, defending_left)
                form_buf.append([match_id, mf.frame_id, team, formation])

        if form_buf:
            self.conn.executemany("INSERT INTO formations VALUES (?,?,?,?)", form_buf)

        # possession phases
        poss_seq = []
        for mf in frames:
            team = next((p.team for p in mf.players if p.has_ball), None)
            poss_seq.append({"frame_id": mf.frame_id, "timestamp_s": mf.timestamp_s, "team": team})

        phases = _detect_phases(poss_seq, fps=fps)
        phase_buf = [
            [match_id, ph.phase_id, ph.team, ph.start_frame, ph.end_frame, ph.start_time_s, ph.end_time_s]
            for ph in phases
        ]
        if phase_buf:
            self.conn.executemany(
                "INSERT INTO possession_phases "
                "(match_id, phase_id, team, start_frame, end_frame, start_time_s, end_time_s) "
                "VALUES (?,?,?,?,?,?,?)",
                phase_buf,
            )

        # scenario_labels — ground truth for supervised ML on play patterns
        if scenario_spans:
            self.conn.executemany(
                "INSERT INTO scenario_labels (match_id, start_frame, end_frame, scenario) "
                "VALUES (?,?,?,?)",
                [[match_id, s, e, n] for s, e, n in scenario_spans],
            )

        self.conn.commit()

    _FRAME_COLS = ["match_id", "frame_id", "timestamp_s", "player_count", "on_court_count"]
    _PLAYER_COLS = [
        "match_id", "frame_id", "track_id", "team", "court_x", "court_y",
        "pixel_foot_x", "pixel_foot_y", "velocity_x", "velocity_y",
        "confidence", "on_court", "has_ball",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]
    _BALL_COLS = [
        "match_id", "frame_id", "court_x", "court_y",
        "pixel_x", "pixel_y", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]

    def _flush(self, frames: list, players: list, balls: list) -> None:
        if frames:
            df = pd.DataFrame(frames, columns=self._FRAME_COLS)
            self.conn.execute("INSERT INTO frames SELECT * FROM df")
        if players:
            df = pd.DataFrame(players, columns=self._PLAYER_COLS)
            self.conn.execute("INSERT INTO players SELECT * FROM df")
        if balls:
            df = pd.DataFrame(balls, columns=self._BALL_COLS)
            self.conn.execute("INSERT INTO ball SELECT * FROM df")

    _LABEL_COLS = ["match_id", "frame_id", "track_id", "action", "annotator"]
    _PRED_COLS = [
        "match_id", "frame_id", "track_id",
        "pass_prob", "shot_prob", "dribble_prob", "hold_prob", "predicted_action",
    ]

    def _flush_labels(self, labels: list, preds: list) -> None:
        if labels:
            df = pd.DataFrame(labels, columns=self._LABEL_COLS)
            self.conn.execute("INSERT INTO action_labels SELECT * FROM df")
        if preds:
            df = pd.DataFrame(preds, columns=self._PRED_COLS)
            self.conn.execute("INSERT INTO action_predictions SELECT * FROM df")

    def close(self) -> None:
        self.conn.close()
