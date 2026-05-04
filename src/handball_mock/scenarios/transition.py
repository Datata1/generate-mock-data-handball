"""Transition — smoothly repositions all players between two scenarios.

Players spread across the court during possession change, naturally producing
the 'transition' formation label.
"""

from __future__ import annotations

import numpy as np

from ..physics import drift_path
from ..types import MatchFrame
from . import ATTACK_A_BASE, DEFENSE_60_BASE, ScenarioBase, build_frames

_DURATION_S = 3.5


class TransitionScenario(ScenarioBase):
    name = "transition"

    def generate(self, start_frame: int) -> list[MatchFrame]:
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        # Spread Team A across court (some sprinting back, some holding position).
        # Players at varied x creates "transition" formation label.
        spread_a: dict[int, tuple[float, float]] = {
            1: (20.0,  1.5),   # LW sprinting back from x=33
            2: (20.0, 18.5),   # RW sprinting back
            3: (18.0,  6.0),   # LB
            4: (20.0, 10.0),   # CB (midfield)
            5: (18.0, 14.0),   # RB
            6: (25.0, 10.0),   # Pivot still somewhat forward
        }

        def _lerp_path(
            start_pos: tuple[float, float],
            end_pos: tuple[float, float],
        ) -> list[tuple[float, float]]:
            result = []
            for i in range(n):
                t = i / max(1, n - 1)
                mu = (1.0 - np.cos(t * np.pi)) / 2.0
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * mu
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * mu
                x += float(rng.normal(0, 0.04))
                y += float(rng.normal(0, 0.04))
                result.append((float(x), float(y)))
            return result

        player_paths: dict = {7: [(None, None)] * n, 14: [(None, None)] * n}

        for tid, end_pos in spread_a.items():
            start_pos = ATTACK_A_BASE.get(tid, end_pos)
            player_paths[tid] = _lerp_path(start_pos, end_pos)

        for tid, base_pos in DEFENSE_60_BASE.items():
            player_paths[tid] = drift_path(*base_pos, n, rng, amp=0.4)

        cb_path = player_paths[4]
        ball = [
            (cb_path[i][0] + float(rng.normal(0, 0.05)),
             cb_path[i][1] + float(rng.normal(0, 0.05)))
            if rng.random() > 0.03 else None
            for i in range(n)
        ]

        action_ranges = [(0, n, 4, "hold")]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
