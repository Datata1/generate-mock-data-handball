"""Doppelpass (Wall Pass) — ≤9 frames between two passes is the ML signature."""

from __future__ import annotations

from ..physics import arc_path, extend_path, smooth_path
from . import DEFENSE_51_BASE, ScenarioBase, attack_base_paths, build_frames, defense_paths

_DURATION_S = 10.0


class DoppelpassScenario(ScenarioBase):
    name = "doppelpass"

    def generate(self, start_frame: int) -> list[MatchFrame]:  # noqa: F821
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        cb_wp = [(28.0, 10.0, 0.0), (28.5, 10.0, 0.8), (33.0, 9.0, 2.5), (33.5, 9.0, _DURATION_S)]
        cb_path = smooth_path(cb_wp, fps, max_speed=7.5, rng=rng)

        lb_wp = [(27.0, 6.0, 0.0), (27.5, 6.5, _DURATION_S)]
        lb_path = smooth_path(lb_wp, fps, rng=rng)

        t_pass1 = round(0.8 * fps)
        t_pass2 = t_pass1 + 8
        arc1 = arc_path(cb_path[t_pass1], lb_path[min(t_pass1, len(lb_path)-1)], 5, rng)
        arc2 = arc_path(lb_path[min(t_pass2, len(lb_path)-1)], cb_path[min(t_pass2+7, len(cb_path)-1)], 5, rng)

        ball: list[tuple[float, float] | None] = []
        for i in range(n):
            if rng.random() < 0.03:
                ball.append(None); continue
            if i < t_pass1:
                ball.append(cb_path[min(i, len(cb_path)-1)])
            elif i < t_pass1 + len(arc1):
                ball.append(arc1[i - t_pass1])
            elif i < t_pass2:
                ball.append(lb_path[min(i, len(lb_path)-1)])
            elif i < t_pass2 + len(arc2):
                ball.append(arc2[i - t_pass2])
            else:
                ball.append(cb_path[min(i, len(cb_path)-1)])

        overrides = {3: extend_path(lb_path, n), 4: extend_path(cb_path, n)}
        a_paths = attack_base_paths(n, rng, overrides)
        b_paths = defense_paths(DEFENSE_51_BASE, n, rng)
        player_paths = {**a_paths, 7: [(None, None)] * n, **b_paths, 14: [(None, None)] * n}
        action_ranges = [
            (0, t_pass1, 4, "hold"), (t_pass1, t_pass1+1, 4, "pass"),
            (t_pass1+5, t_pass2, 3, "hold"), (t_pass2, t_pass2+1, 3, "pass"),
            (t_pass2+5, n, 4, "dribble"),
        ]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
