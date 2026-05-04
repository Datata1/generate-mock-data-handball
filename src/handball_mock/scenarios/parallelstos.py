"""Parallelstoß (Parallel Run) — LW, CB, RW sprint simultaneously from x≈27 to x≈34."""

from __future__ import annotations

from ..physics import arc_path, extend_path, smooth_path
from . import DEFENSE_60_BASE, ScenarioBase, attack_base_paths, build_frames, defense_paths

_DURATION_S = 12.0


class ParallelstoßScenario(ScenarioBase):
    name = "parallelstos"

    def generate(self, start_frame: int) -> list[MatchFrame]:  # noqa: F821
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        t_sprint_end = round(2.0 * fps)
        t_pass = t_sprint_end + round(0.5 * fps)

        cb_wp = [(28.0, 10.0, 0.0), (34.0, 10.0, 2.0), (34.5, 10.0, _DURATION_S)]
        cb_path = smooth_path(cb_wp, fps, max_speed=7.5, rng=rng)

        lw_wp = [(27.0,  1.5, 0.0), (34.0,  1.5, 2.0), (34.5,  1.5, _DURATION_S)]
        lw_path = smooth_path(lw_wp, fps, max_speed=7.5, rng=rng)

        rw_wp = [(27.0, 18.5, 0.0), (34.0, 18.5, 2.0), (34.5, 18.5, _DURATION_S)]
        rw_path = smooth_path(rw_wp, fps, max_speed=7.5, rng=rng)

        target_wing = lw_path if rng.random() < 0.5 else rw_path
        target_id = 1 if (target_wing is lw_path) else 2
        arc1 = arc_path(cb_path[min(t_pass, len(cb_path)-1)], target_wing[min(t_pass, len(target_wing)-1)], 8, rng)

        ball: list[tuple[float, float] | None] = []
        for i in range(n):
            if rng.random() < 0.03:
                ball.append(None); continue
            if i < t_pass:
                ball.append(cb_path[min(i, len(cb_path)-1)])
            elif i < t_pass + len(arc1):
                ball.append(arc1[i - t_pass])
            else:
                ball.append(target_wing[min(i, len(target_wing)-1)])

        overrides = {1: extend_path(lw_path, n), 2: extend_path(rw_path, n), 4: extend_path(cb_path, n)}
        a_paths = attack_base_paths(n, rng, overrides)
        b_paths = defense_paths(DEFENSE_60_BASE, n, rng)
        player_paths = {**a_paths, 7: [(None, None)] * n, **b_paths, 14: [(None, None)] * n}
        action_ranges = [
            (0, t_sprint_end, 4, "dribble"), (t_pass, t_pass+1, 4, "pass"),
            (t_pass+8, n, target_id, "dribble"),
        ]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
