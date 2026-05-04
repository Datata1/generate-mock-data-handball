"""Match assembly — picks scenarios, generates frames, applies physics post-processing."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np

from .config import GeneratorConfig
from .physics import apply_pixel_coords, compute_velocities, mark_ball_carriers
from .scenarios.defense_42 import Defense42Scenario
from .scenarios.defense_51 import Defense51Scenario
from .scenarios.defense_60 import Defense60Scenario
from .scenarios.doppelpass import DoppelpassScenario
from .scenarios.kreislaeuferspiel import KreislaeuferSpielScenario
from .scenarios.kreuzung import KreuzungScenario
from .scenarios.parallelstos import ParallelstoßScenario
from .scenarios.rueckpass import RueckpassScenario
from .scenarios.transition import TransitionScenario
from .types import MatchFrame

if TYPE_CHECKING:
    from .scenarios import ScenarioBase

_SCENARIO_CLASSES = {
    "kreuzung": KreuzungScenario,
    "rueckpass": RueckpassScenario,
    "doppelpass": DoppelpassScenario,
    "parallelstos": ParallelstoßScenario,
    "kreislaeuferspiel": KreislaeuferSpielScenario,
    "defense_60": Defense60Scenario,
    "defense_51": Defense51Scenario,
    "defense_42": Defense42Scenario,
}


class MatchGenerator:
    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def match_id(self, index: int) -> str:
        return f"mock_{self.config.seed:04d}_{index:03d}"

    def generate_match(
        self, match_index: int
    ) -> tuple[str, list[MatchFrame], list[tuple[int, int, str]]]:
        """Generate one full match.

        Returns:
            match_id, frames, scenario_spans
            where scenario_spans = list of (start_frame, end_frame, scenario_name)
        """
        match_id = self.match_id(match_index)
        target = int(self.config.match_duration_s * self.config.fps)

        frames: list[MatchFrame] = []
        scenario_spans: list[tuple[int, int, str]] = []
        current_frame = 0

        while current_frame < target:
            name = self._pick_scenario()
            cls = _SCENARIO_CLASSES[name]
            sub_seed = int(self._rng.integers(0, 2**32))
            sub_rng = np.random.default_rng(sub_seed)
            scenario: ScenarioBase = cls(sub_rng, self.config.fps, self.config)

            start = current_frame
            scenario_frames = scenario.generate(current_frame)
            frames.extend(scenario_frames)
            current_frame += len(scenario_frames)
            scenario_spans.append((start, current_frame - 1, name))

            # 3.5s transition between scenarios
            if current_frame < target:
                trans_seed = int(self._rng.integers(0, 2**32))
                trans_rng = np.random.default_rng(trans_seed)
                trans = TransitionScenario(trans_rng, self.config.fps, self.config)
                t_start = current_frame
                trans_frames = trans.generate(current_frame)
                frames.extend(trans_frames)
                current_frame += len(trans_frames)
                scenario_spans.append((t_start, current_frame - 1, "transition"))

        # Trim to exact target length
        if frames:
            last_keep = target - 1
            scenario_spans = [(s, min(e, last_keep), n) for s, e, n in scenario_spans if s <= last_keep]
        frames = frames[:target]

        # Post-processing
        compute_velocities(frames, self.config.fps)
        mark_ball_carriers(frames)
        apply_pixel_coords(frames, self.config.video_width_px, self.config.video_height_px)

        return match_id, frames, scenario_spans

    def _pick_scenario(self) -> str:
        mix = self.config.scenario_mix
        names = list(mix.keys())
        weights = np.array([mix[n] for n in names], dtype=float)
        weights /= weights.sum()
        return str(self._rng.choice(names, p=weights))
