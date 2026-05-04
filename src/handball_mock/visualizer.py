"""Render a match from DuckDB to an MP4 video — top-down 2D court visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

import duckdb

# ── Court geometry constants ────────────────────────────────────────────────────

_COURT_W = 40.0
_COURT_H = 20.0
_GOAL_Y_MIN = 8.5   # inner goal post y (goal is 3m wide, centred at y=10)
_GOAL_Y_MAX = 11.5

# ── Colours ─────────────────────────────────────────────────────────────────────

_TEAM_A_COLOR = "#1565C0"   # blue
_TEAM_B_COLOR = "#C62828"   # red
_BALL_FACE    = "#FFD740"
_BALL_EDGE    = "#FF6F00"
_COURT_GREEN  = "#1B5E20"
_GOAL_GREY    = "#37474F"
_CARRIER_RING = "#FFD740"


# ── Static court drawing ─────────────────────────────────────────────────────────

def _draw_court(ax: plt.Axes) -> None:
    """Add all static court markings to ax (called once per render)."""

    # Court surface
    ax.add_patch(mpatches.Rectangle(
        (0, 0), _COURT_W, _COURT_H,
        facecolor=_COURT_GREEN, edgecolor="white", linewidth=3, zorder=0,
    ))

    # Centre line
    ax.plot([20, 20], [0, 20], color="white", linewidth=2, zorder=1)

    # --- Left side (goal at x=0) ---
    # 6m arc (goal area)
    ax.add_patch(Arc((0, 10), 12, 12, angle=0, theta1=-90, theta2=90,
                     color="white", linewidth=2, zorder=1))
    # 6m connecting line segments on the sidelines
    ax.plot([0, 0], [0, _GOAL_Y_MIN - 0.01], color="white", linewidth=2, zorder=1)
    ax.plot([0, 0], [_GOAL_Y_MAX + 0.01, 20], color="white", linewidth=2, zorder=1)

    # 9m arc (free-throw line, dashed)
    ax.add_patch(Arc((0, 10), 18, 18, angle=0, theta1=-90, theta2=90,
                     color="white", linewidth=1.5, linestyle=(0, (6, 4)), zorder=1))

    # 7m penalty mark
    ax.plot(7, 10, marker="o", ms=5, color="white", zorder=2)

    # Left goal (rectangle behind the goal line)
    ax.add_patch(mpatches.Rectangle(
        (-2, _GOAL_Y_MIN), 2, _GOAL_Y_MAX - _GOAL_Y_MIN,
        facecolor=_GOAL_GREY, edgecolor="white", linewidth=2, zorder=1,
    ))
    # Goal posts highlighted on the goal line
    ax.plot([0, 0], [_GOAL_Y_MIN, _GOAL_Y_MAX], color="white", linewidth=5, zorder=2)

    # --- Right side (goal at x=40) ---
    ax.add_patch(Arc((40, 10), 12, 12, angle=0, theta1=90, theta2=270,
                     color="white", linewidth=2, zorder=1))
    ax.plot([40, 40], [0, _GOAL_Y_MIN - 0.01], color="white", linewidth=2, zorder=1)
    ax.plot([40, 40], [_GOAL_Y_MAX + 0.01, 20], color="white", linewidth=2, zorder=1)

    ax.add_patch(Arc((40, 10), 18, 18, angle=0, theta1=90, theta2=270,
                     color="white", linewidth=1.5, linestyle=(0, (6, 4)), zorder=1))

    ax.plot(33, 10, marker="o", ms=5, color="white", zorder=2)

    ax.add_patch(mpatches.Rectangle(
        (40, _GOAL_Y_MIN), 2, _GOAL_Y_MAX - _GOAL_Y_MIN,
        facecolor=_GOAL_GREY, edgecolor="white", linewidth=2, zorder=1,
    ))
    ax.plot([40, 40], [_GOAL_Y_MIN, _GOAL_Y_MAX], color="white", linewidth=5, zorder=2)

    # Axis configuration (a bit of padding around court)
    ax.set_xlim(-3.5, 43.5)
    ax.set_ylim(-2.5, 22.5)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Dynamic frame drawing ────────────────────────────────────────────────────────

_SCENARIO_DISPLAY = {
    "kreuzung":          "Kreuzung",
    "rueckpass":         "Rückpass + Durchbruch",
    "doppelpass":        "Doppelpass",
    "parallelstos":      "Parallelstoß",
    "kreislaeuferspiel": "Kreisläufer-Anspiel",
    "defense_60":        "Abwehr 6-0",
    "defense_51":        "Abwehr 5-1",
    "defense_42":        "Abwehr 4-2",
    "transition":        "Transition",
}


def _draw_frame_elements(
    ax: plt.Axes,
    players: list[dict[str, Any]],
    ball: dict[str, Any] | None,
    timestamp_s: float,
    frame_id: int,
    formation_a: str = "?",
    formation_b: str = "?",
    scenario: str | None = None,
) -> list:
    """Add per-frame dynamic artists. Returns the list so callers can remove them."""
    added: list = []

    for p in players:
        cx, cy = p.get("court_x"), p.get("court_y")
        if cx is None or cy is None:
            continue

        team = p["team"]
        has_ball = bool(p["has_ball"])
        track_id = p["track_id"]

        face = _TEAM_A_COLOR if team == "A" else _TEAM_B_COLOR
        edge = _CARRIER_RING if has_ball else "white"
        lw = 3.5 if has_ball else 1.5

        circle = Circle((cx, cy), 0.72, facecolor=face, edgecolor=edge,
                        linewidth=lw, zorder=3)
        ax.add_patch(circle)
        added.append(circle)

        txt = ax.text(cx, cy, str(track_id), ha="center", va="center",
                      color="white", fontsize=7, fontweight="bold", zorder=4)
        added.append(txt)

        # Velocity arrow — only when player is actually moving
        vx = p.get("velocity_x", 0.0) or 0.0
        vy = p.get("velocity_y", 0.0) or 0.0
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > 1.5:
            arrow_scale = min(speed, 7.0) / 7.0 * 2.0  # max 2m arrow
            ann = ax.annotate(
                "", xytext=(cx, cy),
                xy=(cx + vx / speed * arrow_scale, cy + vy / speed * arrow_scale),
                arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.2, mutation_scale=9),
                zorder=5,
            )
            added.append(ann)

    # Ball
    if ball and ball.get("court_x") is not None:
        bx, by = ball["court_x"], ball["court_y"]
        bc = Circle((bx, by), 0.38, facecolor=_BALL_FACE, edgecolor=_BALL_EDGE,
                    linewidth=2, zorder=6)
        ax.add_patch(bc)
        added.append(bc)

    # --- Overlay text ---
    mins, secs = divmod(int(timestamp_s), 60)
    clock = f"{mins:02d}:{secs:02d}  (frame {frame_id})"

    added.append(ax.text(
        0.5, 0.015, clock, transform=ax.transAxes,
        ha="center", va="bottom", color="white", fontsize=8.5, zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000AA", edgecolor="none"),
    ))
    added.append(ax.text(
        0.01, 0.985, f"A  {formation_a}", transform=ax.transAxes,
        ha="left", va="top", color=_TEAM_A_COLOR,
        fontsize=9, fontweight="bold", zorder=10,
    ))
    added.append(ax.text(
        0.99, 0.985, f"{formation_b}  B", transform=ax.transAxes,
        ha="right", va="top", color=_TEAM_B_COLOR,
        fontsize=9, fontweight="bold", zorder=10,
    ))
    if scenario:
        label = _SCENARIO_DISPLAY.get(scenario, scenario)
        added.append(ax.text(
            0.5, 0.985, label, transform=ax.transAxes,
            ha="center", va="top", color="white",
            fontsize=9, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#1A1A2ECC", edgecolor="none"),
        ))
    return added


# ── Main render function ─────────────────────────────────────────────────────────

def render_match(
    db_path: Path,
    match_id: str,
    output_path: Path,
    *,
    max_duration_s: float = 30.0,
    frame_skip: int = 3,
    dpi: int = 120,
    fig_w: float = 12.0,   # 12 × 120 dpi = 1440px (divisible by 16)
    fig_h: float = 6.0,   #  6 × 120 dpi =  720px (divisible by 16)
    progress_cb: "Callable[[int, int], None] | None" = None,  # noqa: F821
) -> None:
    """
    Render a single match from DuckDB to an MP4 video.

    Args:
        db_path:       Path to the DuckDB file.
        match_id:      Which match to render.
        output_path:   Output .mp4 file path.
        max_duration_s: Clip the video at this many seconds.
        frame_skip:    Render every Nth frame (3 → ~8fps output from 25fps source).
        dpi:           Figure resolution.
        progress_cb:   Optional callback(current, total) for progress reporting.
    """
    conn = duckdb.connect(str(db_path), read_only=True)

    row = conn.execute(
        "SELECT fps, total_frames, team_a_name, team_b_name FROM matches WHERE match_id=?",
        [match_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"Match {match_id!r} not found.")

    source_fps, total_frames, team_a_name, team_b_name = row
    max_frame = min(total_frames - 1, int(max_duration_s * source_fps))

    # Pre-load formation labels for the clip
    formation_rows = conn.execute(
        "SELECT frame_id, team, formation FROM formations "
        "WHERE match_id=? AND frame_id <= ?",
        [match_id, max_frame],
    ).fetchall()
    formations: dict[tuple[int, str], str] = {(fid, t): f for fid, t, f in formation_rows}

    def _get_formation(frame_id: int, team: str) -> str:
        sampled = (frame_id // 5) * 5
        return formations.get((sampled, team), "—")

    # Pre-load scenario spans for the clip
    scenario_rows = conn.execute(
        "SELECT start_frame, end_frame, scenario FROM scenario_labels "
        "WHERE match_id=? AND start_frame <= ? ORDER BY start_frame",
        [match_id, max_frame],
    ).fetchall()
    # Build a sorted list for range lookup
    _scenario_spans = [(s, e, n) for s, e, n in scenario_rows]

    def _get_scenario(frame_id: int) -> str | None:
        for s, e, n in _scenario_spans:
            if s <= frame_id <= e:
                return n
        return None

    # --- Matplotlib setup ---
    plt.rcParams.update({"figure.facecolor": "#0D1B2A"})
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0D1B2A")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.04)

    _draw_court(ax)

    # Title bar
    fig.text(0.5, 0.97, f"{team_a_name or 'Team A'}  vs  {team_b_name or 'Team B'}",
             ha="center", va="top", color="white", fontsize=11, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=_TEAM_A_COLOR, edgecolor="white",
                       label=f"Team A — {team_a_name or ''}"),
        mpatches.Patch(facecolor=_TEAM_B_COLOR, edgecolor="white",
                       label=f"Team B — {team_b_name or ''}"),
        mpatches.Patch(facecolor=_BALL_FACE, edgecolor=_BALL_EDGE, label="Ball"),
        mpatches.Patch(facecolor="none", edgecolor=_CARRIER_RING, linewidth=2,
                       label="Ball carrier"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.07), ncol=4,
              framealpha=0.25, labelcolor="white", facecolor="#0D1B2A",
              fontsize=8, handlelength=1.5)

    # Render pixels for canvas size once
    fig.canvas.draw()
    canvas_w = int(fig_w * dpi)
    canvas_h = int(fig_h * dpi)

    output_fps = source_fps / frame_skip
    frame_ids = list(range(0, max_frame + 1, frame_skip))
    total_out = len(frame_ids)

    with imageio.get_writer(
        str(output_path),
        fps=output_fps,
        codec="libx264",
        pixelformat="yuv420p",   # broad player compatibility
        quality=8,
        macro_block_size=16,
    ) as writer:

        for idx, frame_id in enumerate(frame_ids):
            # Fetch players
            players = conn.execute(
                """SELECT track_id, team, court_x, court_y,
                          velocity_x, velocity_y, has_ball
                   FROM players
                   WHERE match_id=? AND frame_id=?
                   ORDER BY track_id""",
                [match_id, frame_id],
            ).df().to_dict("records")

            # Fetch ball
            ball_row = conn.execute(
                "SELECT court_x, court_y FROM ball WHERE match_id=? AND frame_id=?",
                [match_id, frame_id],
            ).fetchone()
            ball = {"court_x": ball_row[0], "court_y": ball_row[1]} if ball_row else None

            ts = frame_id / source_fps
            formation_a = _get_formation(frame_id, "A")
            formation_b = _get_formation(frame_id, "B")
            scenario = _get_scenario(frame_id)

            # Draw dynamic elements
            dynamic = _draw_frame_elements(
                ax, players, ball, ts, frame_id, formation_a, formation_b, scenario
            )
            fig.canvas.draw()

            # Capture frame as numpy array
            img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(canvas_h, canvas_w, 4)[:, :, :3]  # RGBA → RGB

            writer.append_data(img)

            # Remove dynamic elements for next frame
            for artist in dynamic:
                try:
                    artist.remove()
                except (NotImplementedError, ValueError):
                    pass

            if progress_cb is not None:
                progress_cb(idx + 1, total_out)

    conn.close()
    plt.close(fig)
