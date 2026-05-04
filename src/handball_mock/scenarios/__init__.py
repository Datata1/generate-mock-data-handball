"""Scenario base class, shared positions, and frame-assembly helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..physics import drift_path, extend_path
from ..types import ActionEvent, BallFrame, MatchFrame, PlayerFrame

# ── Team assignment ────────────────────────────────────────────────────────────
TEAM_MAP: dict[int, str] = {
    1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "A", 7: "A",
    8: "B", 9: "B", 10: "B", 11: "B", 12: "B", 13: "B", 14: "B",
}

# ── Team A attack base positions (attacking right, toward x=40) ───────────────
# Court reference: 9m line from right goal = x=31, 6m line = x=34
#
# formation.py "attack" rule: in_attacking_zone >= n-1 = 5
#   in_attacking_zone = players with dist_opp(x) <= 14 = players with x >= 26
#   → need at least 5 of 6 players at x >= 26 to get "attack" label.
#
# Backcourt at x=27 → dist_opp=13 <= 14 ✓ (reliably in attack zone even with drift)
ATTACK_A_BASE: dict[int, tuple[float, float]] = {
    1: (33.0,  1.5),   # LW — left wing, near right goal
    2: (33.0, 18.5),   # RW — right wing, near right goal
    3: (27.0,  6.0),   # LB — left back, 13m from goal (x>=26, in attack zone)
    4: (28.0, 10.0),   # CB — centre back, 12m from goal
    5: (27.0, 14.0),   # RB — right back, 13m from goal
    6: (32.0, 10.0),   # Pivot (Kreisläufer) — between 9m (x=31) and 6m (x=34)
}

# ── Team B defensive formations (defending right goal, own_goal_x=40) ─────────
#
# Formation thresholds from ml/analysis/formation.py:
#   dist_own(x) = |x - 40|
#   defensive zone: dist_own <= 14  →  x >= 26
#   "6-0": y_spread <= 10  AND  depth_gap <= 2
#   "5-1": depth_gap >= 3  AND  len(xs) >= 5
#   "4-2": else (fallback)
#
# Defenders stand at x≈32-33 (between 6m at x=34 and 9m at x=31).

# 6-0: CURVED ARC matching the goal-area shape.
# In real handball the 6-0 line is not straight — wing defenders sit closer to
# the goal (x≈34) and center defenders push furthest into the court (x≈32).
# This approximates the arc of the 6m goal-area line.
#
# Formation compliance (defending right, own_goal=40):
#   dist_own values: [6,6,7,7,8,8] → depth_gap=1 ≤ 2 ✓, y_spread=9 ≤ 10 ✓ → "6-0"
DEFENSE_60_BASE: dict[int, tuple[float, float]] = {
    8:  (34.0,  5.5),   # left wing  — close to goal-area line (dist_own=6)
    9:  (33.0,  7.5),   # left mid   — (dist_own=7)
    10: (32.0,  9.5),   # left center — furthest into court (dist_own=8)
    11: (32.0, 10.5),   # right center
    12: (33.0, 12.5),   # right mid
    13: (34.0, 14.5),   # right wing — close to goal-area line
}
# y_spread = 14.5-5.5 = 9 ≤ 10 ✓
# dist_own sorted: [6,6,7,7,8,8] → depth_gap=1 ≤ 2 ✓ → "6-0"

# 5-1: curved 5-line + Ausputzer pressed toward midfield.
# depth_gap between Ausputzer (dist=12) and 5-line (dist≈6-7) = 5 ≥ 3, len=6 ≥ 5 → "5-1"
DEFENSE_51_BASE: dict[int, tuple[float, float]] = {
    8:  (28.0, 10.0),   # Ausputzer — pressed out (dist_own=12)
    9:  (34.5,  5.0),   # 5-line wing left  (dist_own=5.5)
    10: (33.5,  7.5),   # 5-line mid left   (dist_own=6.5)
    11: (33.0, 10.0),   # 5-line center     (dist_own=7)
    12: (33.5, 12.5),   # 5-line mid right
    13: (34.5, 15.0),   # 5-line wing right
}

# 4-2: two staggered rows, wide y-spread.
# dist_own alternates 6.5/8.5 → depth_gap=2 ≤ 2 (fails 6-0 because y_spread=18>10),
# depth_gap=2 < 3 (fails 5-1) → "4-2"
DEFENSE_42_BASE: dict[int, tuple[float, float]] = {
    8:  (33.5,  1.5),
    9:  (31.5,  6.0),
    10: (33.5, 10.0),
    11: (31.5, 14.0),
    12: (33.5, 17.5),
    13: (31.5, 19.5),
}

PATH = list[tuple[float, float]]
NULLABLE_PATH = list[tuple[float | None, float | None]]


def defense_paths(
    base: dict[int, tuple[float, float]],
    n_frames: int,
    rng: np.random.Generator,
    amp: float = 0.25,
) -> dict[int, PATH]:
    return {
        tid: drift_path(bx, by, n_frames, rng, amp=amp)
        for tid, (bx, by) in base.items()
    }


def attack_base_paths(
    n_frames: int,
    rng: np.random.Generator,
    overrides: dict[int, PATH] | None = None,
) -> dict[int, PATH]:
    paths: dict[int, PATH] = {}
    for tid, (bx, by) in ATTACK_A_BASE.items():
        if overrides and tid in overrides:
            paths[tid] = overrides[tid]
        else:
            paths[tid] = drift_path(bx, by, n_frames, rng, amp=0.6)
    return paths


def build_frames(
    start_frame: int,
    player_paths: dict[int, PATH | NULLABLE_PATH],
    ball_path: list[tuple[float, float] | None],
    action_ranges: list[tuple[int, int, int, str]],
    fps: float,
) -> list[MatchFrame]:
    """Assemble MatchFrame objects from precomputed paths and ball trajectory."""
    n = len(ball_path)

    # Build per-frame action lookup
    frame_actions: dict[int, dict[int, str]] = {}
    for s, e, tid, act in action_ranges:
        for f in range(max(0, s), min(e, n)):
            frame_actions.setdefault(f, {})[tid] = act

    frames: list[MatchFrame] = []
    for i in range(n):
        frame_id = start_frame + i
        ts = frame_id / fps

        players: list[PlayerFrame] = []
        for tid in sorted(TEAM_MAP.keys()):
            path = player_paths.get(tid, [])
            pos = path[i] if i < len(path) else (path[-1] if path else (None, None))
            cx, cy = pos
            players.append(
                PlayerFrame(
                    track_id=tid,
                    team=TEAM_MAP[tid],  # type: ignore[arg-type]
                    court_x=float(cx) if cx is not None else None,
                    court_y=float(cy) if cy is not None else None,
                )
            )

        ball_pos = ball_path[i]
        ball: BallFrame | None = (
            BallFrame(court_x=float(ball_pos[0]), court_y=float(ball_pos[1]))
            if ball_pos is not None
            else None
        )

        events = [
            ActionEvent(track_id=tid, action=act)  # type: ignore[arg-type]
            for tid, act in frame_actions.get(i, {}).items()
        ]
        frames.append(MatchFrame(frame_id=frame_id, timestamp_s=ts, players=players, ball=ball, action_events=events))

    return frames


class ScenarioBase(ABC):
    name: str = "base"

    def __init__(
        self,
        rng: np.random.Generator,
        fps: float,
        config: "GeneratorConfig | None" = None,  # noqa: F821
    ) -> None:
        self.rng = rng
        self.fps = fps
        self.config = config

    @abstractmethod
    def generate(self, start_frame: int) -> list[MatchFrame]:
        ...
