"""Court geometry constants and pixel-coordinate transforms."""

from __future__ import annotations

COURT_W: float = 40.0   # metres, x axis
COURT_H: float = 20.0   # metres, y axis

GOAL_LEFT_X: float = 0.0
GOAL_RIGHT_X: float = 40.0
GOAL_Y: float = 10.0          # centre of both goals
GOAL_HALF_WIDTH: float = 1.5  # half-width of goal opening
SIX_M: float = 6.0
NINE_M: float = 9.0
CENTRE_X: float = 20.0

VIDEO_W: int = 1920
VIDEO_H: int = 1080


def court_to_pixel(
    court_x: float,
    court_y: float,
    px_w: int = VIDEO_W,
    px_h: int = VIDEO_H,
) -> tuple[float, float]:
    return court_x * (px_w / COURT_W), court_y * (px_h / COURT_H)


def pixel_bbox_for_player(
    px: float,
    py: float,
    px_h: int = VIDEO_H,
) -> tuple[int, int, int, int]:
    scale_y = px_h / COURT_H
    h = int(1.8 * scale_y)
    w = int(h * 0.4)
    return int(px - w / 2), int(py - h), int(px + w / 2), int(py)


def pixel_bbox_for_ball(
    px: float,
    py: float,
    px_w: int = VIDEO_W,
) -> tuple[int, int, int, int]:
    scale_x = px_w / COURT_W
    r = max(4, int(0.095 * scale_x))
    return int(px - r), int(py - r), int(px + r), int(py + r)
