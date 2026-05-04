"""Kreisläufer-Anspiel — CB (x=27-28) feeds pivot (x=31→33) with a 3-frame forward pass.

The pivot drifts toward the opponent's 6m line (x=34) during backcourt circulation.
ML signatures: 3-frame arc, ball moves forward ~5-6m into 6m zone, pivot stationary at x>31.
"""

from __future__ import annotations

from ..physics import arc_path, drift_path, extend_path, smooth_path
from . import DEFENSE_51_BASE, ScenarioBase, attack_base_paths, build_frames, defense_paths

_DURATION_S = 18.0


class KreislaeuferSpielScenario(ScenarioBase):
    name = "kreislaeuferspiel"

    def generate(self, start_frame: int) -> list[MatchFrame]:  # noqa: F821
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        two_pivots = rng.random() < (self.config.two_pivots_prob if self.config else 0.25)

        t_cb_rb = round(1.5 * fps)
        t_rb_cb = round(4.0 * fps)
        t_cb_pv = round(6.5 * fps)   # 3-frame forward feed into 6m zone
        t_pv_cb = round(14.0 * fps)

        cb_wp = [(28.0, 10.0, 0.0), (27.5, 10.0, 1.5), (27.5, 10.0, 4.0), (28.0, 10.0, 6.5), (28.0, 10.0, _DURATION_S)]
        cb_path = smooth_path(cb_wp, fps, rng=rng)

        rb_wp = [(27.0, 14.0, 0.0), (27.5, 13.5, 3.0), (27.0, 14.0, _DURATION_S)]
        rb_path = smooth_path(rb_wp, fps, rng=rng)

        # Pivot drifts from x=31 (9m line) toward x=33 (near 6m line, inside defense)
        pivot_wp = [(31.0, 10.0, 0.0), (31.5, 10.0, 3.0), (33.0, 10.0, 6.5), (33.0, 10.0, 14.0), (32.0, 10.0, _DURATION_S)]
        pivot_path = smooth_path(pivot_wp, fps, max_speed=3.5, rng=rng)

        arc_cb_rb = arc_path(cb_path[t_cb_rb], rb_path[min(t_cb_rb, len(rb_path)-1)], 12, rng)
        arc_rb_cb = arc_path(rb_path[min(t_rb_cb, len(rb_path)-1)], cb_path[min(t_rb_cb, len(cb_path)-1)], 12, rng)
        pv_feed_pos = pivot_path[min(t_cb_pv, len(pivot_path)-1)]
        arc_cb_pv = arc_path(cb_path[min(t_cb_pv, len(cb_path)-1)], pv_feed_pos, 3, rng)
        arc_pv_cb = arc_path(pv_feed_pos, cb_path[min(t_pv_cb, len(cb_path)-1)], 12, rng)

        ball: list[tuple[float, float] | None] = []
        for i in range(n):
            if rng.random() < 0.03:
                ball.append(None); continue
            if i < t_cb_rb:
                ball.append(cb_path[min(i, len(cb_path)-1)])
            elif i < t_cb_rb + 12:
                ball.append(arc_cb_rb[min(i-t_cb_rb, len(arc_cb_rb)-1)])
            elif i < t_rb_cb:
                ball.append(rb_path[min(i, len(rb_path)-1)])
            elif i < t_rb_cb + 12:
                ball.append(arc_rb_cb[min(i-t_rb_cb, len(arc_rb_cb)-1)])
            elif i < t_cb_pv:
                ball.append(cb_path[min(i, len(cb_path)-1)])
            elif i < t_cb_pv + 3:
                ball.append(arc_cb_pv[min(i-t_cb_pv, len(arc_cb_pv)-1)])
            elif i < t_pv_cb:
                px, py = pv_feed_pos
                ball.append((px + float(rng.normal(0, 0.06)), py + float(rng.normal(0, 0.06))))
            elif i < t_pv_cb + 12:
                ball.append(arc_pv_cb[min(i-t_pv_cb, len(arc_pv_cb)-1)])
            else:
                ball.append(cb_path[min(i, len(cb_path)-1)])

        aus_wp = [(28.0, 10.0, 0.0), (28.0, 10.0, 6.5), (26.0, 10.0, 8.0), (26.0, 10.0, 14.0), (28.0, 10.0, _DURATION_S)]
        b_paths = defense_paths(DEFENSE_51_BASE, n, rng)
        b_paths[8] = extend_path(smooth_path(aus_wp, fps, rng=rng), n)

        overrides = {4: extend_path(cb_path, n), 5: extend_path(rb_path, n), 6: extend_path(pivot_path, n)}
        if two_pivots:
            pv2_wp = [(31.0, 12.5, 0.0), (32.5, 13.0, 6.5), (33.0, 13.0, 14.0), (32.0, 12.5, _DURATION_S)]
            overrides[2] = extend_path(smooth_path(pv2_wp, fps, max_speed=3.5, rng=rng), n)

        a_paths = attack_base_paths(n, rng, overrides)
        player_paths = {**a_paths, 7: [(None, None)] * n, **b_paths, 14: [(None, None)] * n}
        action_ranges = [
            (0, t_cb_rb, 4, "hold"), (t_cb_rb, t_cb_rb+1, 4, "pass"),
            (t_cb_rb+12, t_rb_cb, 5, "dribble"), (t_rb_cb, t_rb_cb+1, 5, "pass"),
            (t_rb_cb+12, t_cb_pv, 4, "hold"), (t_cb_pv, t_cb_pv+1, 4, "pass"),
            (t_cb_pv+3, t_pv_cb, 6, "hold"), (t_pv_cb, t_pv_cb+1, 6, "pass"),
            (t_pv_cb+12, n, 4, "hold"),
        ]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
