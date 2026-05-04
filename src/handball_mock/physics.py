"""Player movement interpolation, velocity computation, and ball-carrier marking."""

from __future__ import annotations

import math

import numpy as np

from .court import court_to_pixel, pixel_bbox_for_ball, pixel_bbox_for_player
from .types import BallFrame, MatchFrame, PlayerFrame


def smooth_path(
    waypoints: list[tuple[float, float, float]],
    fps: float,
    max_speed: float = 7.0,
    rng: np.random.Generator | None = None,
    jitter: float = 0.04,
) -> list[tuple[float, float]]:
    """
    Cosine-eased interpolation between (x, y, t_s) waypoints.
    Returns one (x, y) per frame at fps resolution.
    Length = round((t_last - t_first) * fps) + 1.
    """
    if not waypoints:
        return []
    if len(waypoints) == 1:
        x, y, _ = waypoints[0]
        return [(float(x), float(y))]

    result: list[tuple[float, float]] = []
    for i in range(len(waypoints) - 1):
        x0, y0, t0 = waypoints[i]
        x1, y1, t1 = waypoints[i + 1]
        n = max(1, round((t1 - t0) * fps))
        for j in range(n):
            alpha = j / n
            mu = (1.0 - math.cos(alpha * math.pi)) / 2.0
            x = x0 + (x1 - x0) * mu
            y = y0 + (y1 - y0) * mu
            if rng is not None:
                x += float(rng.normal(0.0, jitter))
                y += float(rng.normal(0.0, jitter))
            result.append((float(x), float(y)))

    xf, yf, _ = waypoints[-1]
    if rng is not None:
        xf += float(rng.normal(0.0, jitter))
        yf += float(rng.normal(0.0, jitter))
    result.append((float(xf), float(yf)))

    # Enforce speed cap — prevents teleportation even with large waypoint gaps
    for i in range(1, len(result)):
        px, py = result[i - 1]
        cx, cy = result[i]
        dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        speed = dist * fps
        if speed > max_speed:
            scale = max_speed / speed
            result[i] = (px + (cx - px) * scale, py + (cy - py) * scale)

    return result


def drift_path(
    base_x: float,
    base_y: float,
    n_frames: int,
    rng: np.random.Generator,
    amp: float = 0.35,
    spring: float = 0.15,
    noise: float = 0.07,
) -> list[tuple[float, float]]:
    """Small oscillating motion around a base position (defensive sway, etc.)."""
    x, y = base_x, base_y
    result: list[tuple[float, float]] = [(x, y)]
    for _ in range(n_frames - 1):
        x += float(rng.normal(0.0, noise)) - spring * (x - base_x)
        y += float(rng.normal(0.0, noise)) - spring * (y - base_y)
        x = float(np.clip(x, base_x - amp, base_x + amp))
        y = float(np.clip(y, base_y - amp, base_y + amp))
        result.append((x, y))
    return result


def arc_path(
    from_pos: tuple[float, float],
    to_pos: tuple[float, float],
    n_frames: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Linear ball flight with a slight lateral arc (simulates real throw)."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    dist = math.sqrt(dx * dx + dy * dy) or 1.0
    perp_x = -dy / dist
    perp_y = dx / dist

    result: list[tuple[float, float]] = []
    for i in range(max(1, n_frames)):
        t = i / max(1, n_frames - 1)
        x = from_pos[0] + dx * t + perp_x * math.sin(t * math.pi) * 0.2
        y = from_pos[1] + dy * t + perp_y * math.sin(t * math.pi) * 0.2
        if rng is not None:
            x += float(rng.normal(0.0, 0.03))
            y += float(rng.normal(0.0, 0.03))
        result.append((float(x), float(y)))
    return result


def extend_path(
    path: list[tuple[float, float]],
    n_frames: int,
) -> list[tuple[float, float]]:
    """Ensure a path is exactly n_frames long by trimming or repeating the last point."""
    if len(path) >= n_frames:
        return path[:n_frames]
    return path + [path[-1]] * (n_frames - len(path))


def compute_velocities(frames: list[MatchFrame], fps: float) -> None:
    """Compute and set velocity_x/y on all PlayerFrames in place."""
    prev: dict[int, tuple[float, float]] = {}
    for mf in frames:
        for p in mf.players:
            if p.court_x is not None and p.court_y is not None:
                if p.track_id in prev:
                    px, py = prev[p.track_id]
                    vx = (p.court_x - px) * fps
                    vy = (p.court_y - py) * fps
                    speed = math.sqrt(vx * vx + vy * vy)
                    if speed > 8.5:
                        scale = 8.5 / speed
                        vx, vy = vx * scale, vy * scale
                    p.velocity_x = float(vx)
                    p.velocity_y = float(vy)
                prev[p.track_id] = (p.court_x, p.court_y)


def mark_ball_carriers(frames: list[MatchFrame]) -> None:
    """Set has_ball=True on the on-court player nearest the ball per frame."""
    for mf in frames:
        # Reset first so re-calling is idempotent
        for p in mf.players:
            p.has_ball = False
        if mf.ball is None:
            continue
        bx, by = mf.ball.court_x, mf.ball.court_y
        candidates = [
            p for p in mf.players
            if p.on_court and p.court_x is not None and p.court_y is not None
        ]
        if not candidates:
            continue
        closest = min(
            candidates,
            key=lambda p: (p.court_x - bx) ** 2 + (p.court_y - by) ** 2,  # type: ignore[operator]
        )
        closest.has_ball = True


def apply_pixel_coords(
    frames: list[MatchFrame],
    px_w: int = 1920,
    px_h: int = 1080,
) -> None:
    """Compute and set pixel coordinates on all PlayerFrames and BallFrames in place."""
    for mf in frames:
        for p in mf.players:
            if p.court_x is not None and p.court_y is not None:
                px, py = court_to_pixel(p.court_x, p.court_y, px_w, px_h)
            else:
                # Goalkeeper — fixed pixel position near goal
                px = 24.0 if p.team == "A" else float(px_w - 24)
                py = float(px_h / 2)
            p.pixel_foot_x = float(px)
            p.pixel_foot_y = float(py)
            p.bbox_x1, p.bbox_y1, p.bbox_x2, p.bbox_y2 = pixel_bbox_for_player(px, py, px_h)
        if mf.ball is not None:
            bx, by = court_to_pixel(mf.ball.court_x, mf.ball.court_y, px_w, px_h)
            mf.ball.pixel_x = float(bx)
            mf.ball.pixel_y = float(by)
            mf.ball.bbox_x1, mf.ball.bbox_y1, mf.ball.bbox_x2, mf.ball.bbox_y2 = pixel_bbox_for_ball(bx, by, px_w)
