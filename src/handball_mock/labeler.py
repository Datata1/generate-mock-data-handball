"""Auto action-label extraction from scenario events + kinematic fallback."""

from __future__ import annotations

import math

from .types import ActionLabel, MatchFrame


def extract_labels(frames: list[MatchFrame]) -> dict[tuple[int, int], ActionLabel]:
    """
    Return {(frame_id, track_id): action} for all labeled frames.

    Priority:
    1. Explicit ActionEvents set by the scenario
    2. Kinematic fallback for ball carriers without an explicit label
    """
    labels: dict[tuple[int, int], ActionLabel] = {}

    # Pass 1: collect explicit scenario events
    for mf in frames:
        for ev in mf.action_events:
            labels[(mf.frame_id, ev.track_id)] = ev.action

    # Pass 2: kinematic fallback for ball carriers with no explicit label
    # We look at a window of ±3 frames around each unlabeled ball carrier.
    frame_map = {mf.frame_id: mf for mf in frames}
    sorted_fids = sorted(frame_map.keys())

    for i, fid in enumerate(sorted_fids):
        mf = frame_map[fid]
        carrier = _ball_carrier(mf)
        if carrier is None:
            continue
        if (fid, carrier.track_id) in labels:
            continue

        # Kinematic rules
        ball = mf.ball
        if ball is None:
            labels[(fid, carrier.track_id)] = "hold"
            continue

        # Check if ball is heading fast toward goal (shot)
        speed = math.sqrt(carrier.velocity_x ** 2 + carrier.velocity_y ** 2)
        next_mf = frame_map.get(sorted_fids[i + 1]) if i + 1 < len(sorted_fids) else None
        if next_mf is not None and next_mf.ball is not None:
            dbx = next_mf.ball.court_x - ball.court_x
            if abs(dbx) * 25 > 5.5 and (
                next_mf.ball.court_x > 35 or next_mf.ball.court_x < 5
            ):
                labels[(fid, carrier.track_id)] = "shot"
                continue

        # Check if carrier changes next frame (pass)
        if next_mf is not None:
            next_carrier = _ball_carrier(next_mf)
            if next_carrier is not None and next_carrier.track_id != carrier.track_id:
                labels[(fid, carrier.track_id)] = "pass"
                continue

        # Speed-based dribble vs hold
        if speed > 1.8:
            labels[(fid, carrier.track_id)] = "dribble"
        else:
            labels[(fid, carrier.track_id)] = "hold"

    return labels


def _ball_carrier(mf: MatchFrame) -> "PlayerFrame | None":  # noqa: F821
    for p in mf.players:
        if p.has_ball:
            return p
    return None
