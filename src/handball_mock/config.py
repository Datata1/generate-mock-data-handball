"""Generator configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    seed: int = 42
    fps: float = 25.0
    num_matches: int = 3
    match_duration_s: float = 600.0
    video_width_px: int = 1920
    video_height_px: int = 1080
    two_pivots_prob: float = 0.25
    scenario_mix: dict[str, float] = field(
        default_factory=lambda: {
            "kreuzung": 0.13,
            "rueckpass": 0.13,
            "doppelpass": 0.13,
            "parallelstos": 0.13,
            "kreislaeuferspiel": 0.20,
            "defense_60": 0.10,
            "defense_51": 0.09,
            "defense_42": 0.09,
        }
    )
    output_db: str = "handball_mock.duckdb"
    team_a_name: str = "Wels"
    team_b_name: str = "Linz"
