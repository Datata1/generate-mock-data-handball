"""Rückpass + Durchbruch — CB drives x=28→32, back-passes to RB, sprints to x=36."""

from __future__ import annotations

from ..physics import arc_path, extend_path, smooth_path
from . import DEFENSE_60_BASE, ScenarioBase, attack_base_paths, build_frames, defense_paths

_DURATION_S = 15.0


class RueckpassScenario(ScenarioBase):
    name = "rueckpass"

    def generate(self, start_frame: int) -> list[MatchFrame]:  # noqa: F821
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        cb_wp = [
            (28.0, 10.0, 0.0),
            (32.0, 10.0, 1.5),   # drives toward 9m line
            (32.0, 10.0, 2.0),   # releases back-pass
            (36.0, 10.0, 3.2),   # SPRINT through gap to near 6m line
            (36.0, 10.0, _DURATION_S),
        ]
        cb_path = smooth_path(cb_wp, fps, max_speed=7.5, rng=rng)

        rb_wp = [(27.0, 14.0, 0.0), (27.5, 13.5, 2.5), (27.0, 14.0, _DURATION_S)]
        rb_path = smooth_path(rb_wp, fps, rng=rng)

        t_back = round(2.0 * fps)
        t_return = round(4.5 * fps)
        arc_back = arc_path(cb_path[t_back], rb_path[min(t_back, len(rb_path)-1)], 10, rng)
        arc_ret = arc_path(rb_path[min(t_return, len(rb_path)-1)], cb_path[min(t_return, len(cb_path)-1)], 12, rng)

        ball: list[tuple[float, float] | None] = []
        for i in range(n):
            if rng.random() < 0.03:
                ball.append(None); continue
            if i < t_back:
                ball.append(cb_path[min(i, len(cb_path)-1)])
            elif i < t_back + len(arc_back):
                ball.append(arc_back[i - t_back])
            elif i < t_return:
                ball.append(rb_path[min(i, len(rb_path)-1)])
            elif i < t_return + len(arc_ret):
                ball.append(arc_ret[i - t_return])
            else:
                ball.append(cb_path[min(i, len(cb_path)-1)])

        overrides = {4: extend_path(cb_path, n), 5: extend_path(rb_path, n)}
        a_paths = attack_base_paths(n, rng, overrides)
        b_paths = defense_paths(DEFENSE_60_BASE, n, rng)
        player_paths = {**a_paths, 7: [(None, None)] * n, **b_paths, 14: [(None, None)] * n}
        action_ranges = [
            (0, round(1.0*fps), 4, "dribble"), (t_back, t_back+1, 4, "pass"),
            (t_back+10, t_return, 5, "hold"), (t_return, t_return+1, 5, "pass"),
            (t_return+12, n, 4, "dribble"),
        ]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
