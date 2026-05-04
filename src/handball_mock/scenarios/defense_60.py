"""6-0 defense scenario — Team B compact line at x≈32, Team A circulates ball in backcourt."""

from __future__ import annotations

from ..physics import arc_path, drift_path
from . import ATTACK_A_BASE, DEFENSE_60_BASE, ScenarioBase, build_frames, defense_paths

_DURATION_S = 10.0


class Defense60Scenario(ScenarioBase):
    name = "defense_60"

    def generate(self, start_frame: int) -> list[MatchFrame]:  # noqa: F821
        rng = self.rng
        fps = self.fps
        n = round(_DURATION_S * fps)

        # Team A: slow ball circulation among backcourt (CB/LB/RB at x=24-26)
        # Wings and pivot stay at their correct positions near the opponent's goal
        cb_path = drift_path(*ATTACK_A_BASE[4], n, rng, amp=0.9)
        lb_path = drift_path(*ATTACK_A_BASE[3], n, rng, amp=0.9)
        rb_path = drift_path(*ATTACK_A_BASE[5], n, rng, amp=0.9)
        lw_path = drift_path(*ATTACK_A_BASE[1], n, rng, amp=0.7)
        rw_path = drift_path(*ATTACK_A_BASE[2], n, rng, amp=0.7)
        pv_path = drift_path(*ATTACK_A_BASE[6], n, rng, amp=0.6)

        t1 = round(2.0 * fps)
        t2 = round(5.0 * fps)
        t3 = round(7.5 * fps)
        arc1 = arc_path(cb_path[t1], lb_path[t1], 10, rng)
        arc2 = arc_path(lb_path[t2], rb_path[t2], 10, rng)
        arc3 = arc_path(rb_path[t3], cb_path[t3], 10, rng)

        ball: list[tuple[float, float] | None] = []
        for i in range(n):
            if rng.random() < 0.03:
                ball.append(None); continue
            if i < t1:
                ball.append(cb_path[i])
            elif i < t1 + 10:
                ball.append(arc1[min(i - t1, len(arc1)-1)])
            elif i < t2:
                ball.append(lb_path[i])
            elif i < t2 + 10:
                ball.append(arc2[min(i - t2, len(arc2)-1)])
            elif i < t3:
                ball.append(rb_path[i])
            elif i < t3 + 10:
                ball.append(arc3[min(i - t3, len(arc3)-1)])
            else:
                ball.append(cb_path[i])

        b_paths = defense_paths(DEFENSE_60_BASE, n, rng, amp=0.22)
        player_paths = {
            1: lw_path, 2: rw_path, 3: lb_path, 4: cb_path, 5: rb_path, 6: pv_path,
            7: [(None, None)] * n, **b_paths, 14: [(None, None)] * n,
        }
        action_ranges = [
            (0, t1, 4, "hold"), (t1, t1+1, 4, "pass"),
            (t1+10, t2, 3, "hold"), (t2, t2+1, 3, "pass"),
            (t2+10, t3, 5, "hold"), (t3, t3+1, 5, "pass"),
            (t3+10, n, 4, "hold"),
        ]
        return build_frames(start_frame, player_paths, ball, action_ranges, fps)
