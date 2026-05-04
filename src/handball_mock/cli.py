"""CLI entry point for the handball mock data generator."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import GeneratorConfig
from .generator import MatchGenerator
from .writer import MockDataWriter

app = typer.Typer(
    name="handball-mock",
    help="Generate realistic synthetic handball match data for ML exploration.",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    output: Annotated[Path, typer.Argument(help="Output DuckDB path")],
    matches: Annotated[int, typer.Option("--matches", "-n", help="Number of matches")] = 3,
    duration: Annotated[float, typer.Option("--duration", "-d", help="Seconds per match")] = 600.0,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
    fps: Annotated[float, typer.Option("--fps", help="Frames per second")] = 25.0,
    team_a: Annotated[str, typer.Option("--team-a")] = "Wels",
    team_b: Annotated[str, typer.Option("--team-b")] = "Linz",
    two_pivots_prob: Annotated[float, typer.Option("--two-pivots-prob")] = 0.25,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate synthetic handball match data and write to OUTPUT (DuckDB)."""
    config = GeneratorConfig(
        seed=seed,
        fps=fps,
        num_matches=matches,
        match_duration_s=duration,
        two_pivots_prob=two_pivots_prob,
        output_db=str(output),
        team_a_name=team_a,
        team_b_name=team_b,
    )

    gen = MatchGenerator(config)
    writer = MockDataWriter(output)

    expected_frames = int(duration * fps)
    expected_players = expected_frames * 14

    console.print(f"\n[bold]Handball Mock Data Generator[/bold]")
    console.print(f"  Output:  [cyan]{output}[/cyan]")
    console.print(f"  Matches: {matches}  ·  Duration: {duration:.0f}s each  ·  FPS: {fps:.0f}")
    console.print(f"  Teams:   {team_a} vs {team_b}  ·  Seed: {seed}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating matches...", total=matches)

        for i in range(matches):
            progress.update(task, description=f"Match {i + 1}/{matches} — generating frames...")
            match_id, frames, spans = gen.generate_match(i)

            if verbose:
                scenario_counts = {}
                for _, _, name in spans:
                    scenario_counts[name] = scenario_counts.get(name, 0) + 1
                summary = ", ".join(f"{k}×{v}" for k, v in sorted(scenario_counts.items()))
                progress.print(f"  [{match_id}] {len(frames)} frames — {summary}")

            progress.update(task, description=f"Match {i + 1}/{matches} — writing to DuckDB...")
            writer.write_match(match_id, frames, fps, team_a, team_b, scenario_spans=spans)
            progress.advance(task)

    writer.close()

    # Summary table
    table = Table(title="Generated Matches", show_header=True)
    table.add_column("Match ID", style="cyan")
    table.add_column("Frames", justify="right")
    table.add_column("Players", justify="right")

    for i in range(matches):
        mid = gen.match_id(i)
        table.add_row(mid, str(expected_frames), str(expected_players))

    console.print(table)
    console.print(f"\n[green]✓[/green] Written to [bold]{output}[/bold]\n")


@app.command()
def verify(
    db: Annotated[Path, typer.Argument(help="DuckDB file to verify")],
) -> None:
    """Run consistency checks on a generated DuckDB file."""
    import duckdb

    if not db.exists():
        console.print(f"[red]Error:[/red] {db} does not exist")
        raise typer.Exit(1)

    conn = duckdb.connect(str(db), read_only=True)
    failures = []

    checks = [
        (
            "Ball carrier uniqueness",
            "SELECT COUNT(*) FROM (SELECT match_id, frame_id, COUNT(*) n FROM players WHERE has_ball=TRUE GROUP BY 1,2 HAVING n>1)",
        ),
        (
            "Valid formation labels",
            "SELECT COUNT(*) FROM formations WHERE formation NOT IN ('6-0','5-1','4-2','attack','transition','unknown')",
        ),
        (
            "Softmax sums to 1",
            "SELECT COUNT(*) FROM action_predictions WHERE abs(pass_prob+shot_prob+dribble_prob+hold_prob-1.0)>0.02",
        ),
        (
            "Positive possession duration",
            "SELECT COUNT(*) FROM possession_phases WHERE duration_s<=0",
        ),
        (
            "Players have pixel coords",
            "SELECT COUNT(*) FROM players WHERE pixel_foot_x=0 AND pixel_foot_y=0",
        ),
    ]

    console.print(f"\n[bold]Verifying[/bold] {db}\n")
    all_ok = True
    for name, sql in checks:
        count = conn.execute(sql).fetchone()[0]
        if count == 0:
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [red]✗[/red] {name} — {count} violations")
            failures.append(name)
            all_ok = False

    # Speed check (no player moves > 8.5 m/s between consecutive frames)
    speed_sql = """
        SELECT COUNT(*) FROM (
            SELECT p1.match_id, p1.track_id, p1.frame_id,
                   sqrt(power(p1.velocity_x,2)+power(p1.velocity_y,2)) AS speed
            FROM players p1
            WHERE p1.court_x IS NOT NULL
              AND sqrt(power(p1.velocity_x,2)+power(p1.velocity_y,2)) > 9.0
        )
    """
    count = conn.execute(speed_sql).fetchone()[0]
    if count == 0:
        console.print("  [green]✓[/green] No teleportation (speed < 9 m/s)")
    else:
        console.print(f"  [yellow]⚠[/yellow]  Speed outliers: {count} frames >9 m/s (may be OK)")

    # Stats
    stats = conn.execute("""
        SELECT
            (SELECT COUNT(*) FROM matches) as matches,
            (SELECT COUNT(*) FROM frames) as frames,
            (SELECT COUNT(*) FROM players) as players,
            (SELECT COUNT(*) FROM ball) as ball_rows,
            (SELECT COUNT(*) FROM action_labels) as labels,
            (SELECT COUNT(*) FROM formations) as formations,
            (SELECT COUNT(*) FROM possession_phases) as phases
    """).fetchone()

    console.print()
    console.print(f"  Matches: {stats[0]}  Frames: {stats[1]}  Players: {stats[2]}")
    console.print(f"  Ball:    {stats[3]}  Labels: {stats[4]}  Formations: {stats[5]}  Phases: {stats[6]}")

    conn.close()
    if not all_ok:
        raise typer.Exit(1)
    console.print("\n[green]All checks passed.[/green]\n")


@app.command()
def scenarios() -> None:
    """List all available scenario types."""
    table = Table(title="Available Scenarios")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Duration", justify="right")

    rows = [
        ("kreuzung", "Kreuzung — two players cross and exchange ball", "~12 s"),
        ("rueckpass", "Rückpass + Durchbruch — give-and-go with sprint", "~15 s"),
        ("doppelpass", "Doppelpass — wall pass, sub-9-frame turnaround", "~10 s"),
        ("parallelstos", "Parallelstoß — three parallel sprints", "~12 s"),
        ("kreislaeuferspiel", "Kreisläufer-Anspiel — pivot entry with 3-frame feed", "~18 s"),
        ("defense_60", "6-0 compact defensive line", "~10 s"),
        ("defense_51", "5-1 with Ausputzer forward", "~10 s"),
        ("defense_42", "4-2 wide zone defence", "~10 s"),
    ]
    for name, desc, dur in rows:
        table.add_row(name, desc, dur)

    console.print(table)


@app.command()
def render(
    db: Annotated[Path, typer.Argument(help="DuckDB file to read from")],
    match_id: Annotated[str, typer.Argument(help="Match ID to render (e.g. mock_0042_000)")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output MP4 path")] = Path("preview.mp4"),
    duration: Annotated[float, typer.Option("--duration", "-d", help="Clip length in seconds")] = 30.0,
    frame_skip: Annotated[int, typer.Option("--frame-skip", "-s", help="Render every Nth frame (higher = faster)")] = 3,
    dpi: Annotated[int, typer.Option("--dpi", help="Figure resolution (higher = sharper)")] = 120,
) -> None:
    """Render a match to a top-down 2D video (MP4)."""
    from .visualizer import render_match

    if not db.exists():
        console.print(f"[red]Error:[/red] {db} does not exist")
        raise typer.Exit(1)

    output_fps = 25.0 / frame_skip
    console.print(
        f"\nRendering [cyan]{match_id}[/cyan] → [bold]{output}[/bold]  "
        f"({duration:.0f}s clip, {output_fps:.1f} fps output)\n"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rendering frames...", total=int(duration * 25 / frame_skip))

        def on_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        render_match(
            db_path=db,
            match_id=match_id,
            output_path=output,
            max_duration_s=duration,
            frame_skip=frame_skip,
            dpi=dpi,
            progress_cb=on_progress,
        )

    size_mb = output.stat().st_size / 1_048_576
    console.print(f"\n[green]✓[/green] Saved to [bold]{output}[/bold] ({size_mb:.1f} MB)\n")
