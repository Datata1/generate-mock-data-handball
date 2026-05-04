"""Core dataclasses shared across the generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TeamLabel = Literal["A", "B", "unknown"]
ActionLabel = Literal["pass", "shot", "dribble", "hold"]


@dataclass
class PlayerFrame:
    track_id: int
    team: TeamLabel
    court_x: float | None
    court_y: float | None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    confidence: float = 0.95
    on_court: bool = True
    has_ball: bool = False
    pixel_foot_x: float = 0.0
    pixel_foot_y: float = 0.0
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0


@dataclass
class BallFrame:
    court_x: float
    court_y: float
    confidence: float = 0.92
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0


@dataclass
class ActionEvent:
    track_id: int
    action: ActionLabel


@dataclass
class MatchFrame:
    frame_id: int
    timestamp_s: float
    players: list[PlayerFrame]
    ball: BallFrame | None
    action_events: list[ActionEvent] = field(default_factory=list)
