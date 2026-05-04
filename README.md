# Handball Mock Data Generator

Generates realistic synthetic handball match data — a **DuckDB file** that is
schema-compatible with the [wels-monorepo](../wels-monorepo) analysis pipeline.
Use this to explore ML approaches (clustering, GCN training, formation detection)
**without needing real video footage**.

---

## Quick Start

```bash
# 1 – Install
git clone <this-repo> && cd mock-data-handball
uv sync

# 2 – Generate 3 × 10-minute matches (~15 seconds)
just generate

# 3 – Verify the output
just verify

# 4 – Watch a 30-second video clip
just visualize

# 5 – Explore the database interactively
just explore
```

Everything is driven by the `justfile`. Run `just` (no arguments) to see all available commands.

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.13+ | Runtime | [python.org](https://python.org) |
| [uv](https://docs.astral.sh/uv/) | Package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [just](https://just.systems) | Task runner | `cargo install just` or `apt install just` or `brew install just` |
| [harlequin](https://harlequin.sh) | DuckDB TUI explorer | `uv tool install harlequin` *(installed automatically by `just install`)* |

---

## Installation

```bash
cd mock-data-handball
uv sync          # installs all Python dependencies into .venv
```

Dependencies installed: `duckdb`, `numpy`, `pandas`, `matplotlib`, `imageio[ffmpeg]`, `typer`, `rich`.

---

## All Commands

### Using `just` (recommended)

```bash
just                          # show this list
just install                  # uv sync + install harlequin

just generate                 # 3 matches × 10 min → handball_mock.duckdb (default)
just generate n=5 d=300       # 5 matches × 5 min
just generate seed=99         # different random seed (fully reproducible)

just verify                   # run consistency checks on handball_mock.duckdb
just all                      # generate + verify in one step

just visualize                # render a 30-second video → preview.mp4
just visualize dur=120 skip=5 # 2-minute clip at 5fps output
just visualize match=mock_0042_001 out=match2.mp4

just explore                  # open harlequin TUI on handball_mock.duckdb
just explore file=other.duckdb

just scenarios                # list all available scenario types
just smoke                    # quick 2-minute smoke test
just clean                    # delete *.duckdb files
```

### Using the CLI directly

```bash
uv run handball-mock generate output.duckdb \
  --matches 3 --duration 600 --seed 42 \
  --team-a "Wels" --team-b "Stuttgart" \
  --verbose

uv run handball-mock verify output.duckdb

uv run handball-mock render output.duckdb mock_0042_000 \
  --output clip.mp4 --duration 60 --frame-skip 3

uv run handball-mock scenarios
```

---

## What Gets Generated

### Database schema

Nine tables are written to the DuckDB file:

| Table | Contents | Rows per 10-min match |
|-------|----------|-----------------------|
| `matches` | Match metadata (FPS, team names, frame count) | 1 |
| `frames` | Per-frame metadata at 25 fps | ~15,000 |
| `players` | Position, velocity, team, `has_ball` per player per frame | ~210,000 |
| `ball` | Ball court position (~97% detection rate) | ~14,550 |
| `action_labels` | Ball-carrier action: `pass / shot / dribble / hold` | ~14,550 |
| `action_predictions` | Mock softmax probabilities (noisy but class-consistent) | ~14,550 |
| `formations` | Team formation label sampled every 5 frames | ~6,000 |
| `possession_phases` | Continuous ball-possession sequences | ~100 |
| **`scenario_labels`** | **Ground-truth play-pattern per segment — key for supervised ML** | ~35 |

The first eight tables are identical in structure to what the real CV pipeline
produces. `scenario_labels` is additional ground truth only available here.

### Key columns in `players`

| Column | Type | Description |
|--------|------|-------------|
| `court_x` | DOUBLE (nullable) | X position in metres (0 = left goal, 40 = right goal). NULL for goalkeepers. |
| `court_y` | DOUBLE (nullable) | Y position in metres (0–20). |
| `velocity_x/y` | DOUBLE | m/s, computed as Δposition × fps |
| `has_ball` | BOOLEAN | TRUE for exactly one player per frame (nearest to ball) |
| `team` | TEXT | `"A"` or `"B"` |

### Player layout (fixed track IDs)

Team A always attacks right (toward x=40). Team B defends the right goal.

| Track | Role | Team | Typical position |
|-------|------|------|-----------------|
| 1 | Left Wing | A | x≈33, y≈1.5 |
| 2 | Right Wing | A | x≈33, y≈18.5 |
| 3 | Left Back | A | x≈27, y≈6 |
| 4 | Center Back | A | x≈28, y≈10 — primary ball handler |
| 5 | Right Back | A | x≈27, y≈14 |
| 6 | Pivot (Kreisläufer) | A | x≈31–33, y≈10 — near opponent 6m line |
| 7 | Goalkeeper | A | court x/y = NULL |
| 8–13 | Defenders | B | formation-dependent (see below) |
| 14 | Goalkeeper | B | court x/y = NULL |

---

## Tactical Content

### The handball court

```
y=0  ┌──────────────────────────────────────────┐
     │ GK-A │6m│9m│        │9m│6m│ GK-B        │
     │  x=0      x=6  x=9  x=20  x=31 x=34  x=40│
y=20 └──────────────────────────────────────────┘
```

- Team A attacks right → opponent goal at x=40
- 6m line (goal area): x=34 from right goal
- 9m line (free-throw): x=31 from right goal
- Pivot plays between these lines (x=31–34)

### The 5 Spielzüge (attack patterns)

Each scenario runs for 10–18 seconds. The `scenario` column in `scenario_labels`
identifies which play is active at any frame.

| `scenario` value | Name | What happens | ML signature |
|---|---|---|---|
| `kreuzung` | Kreuzung | CB and LB cross paths + exchange ball | intersecting trajectories in y, ball changes hands twice |
| `rueckpass` | Rückpass + Durchbruch | CB drives forward, back-passes, sprints to x=36 | velocity burst ~7 m/s immediately after pass event |
| `doppelpass` | Doppelpass | CB passes to wall player, curved sprint, one-touch return | ≤9 frames (0.36s) between two consecutive passes |
| `parallelstos` | Parallelstoß | LW + CB + RW sprint simultaneously in parallel | correlated `velocity_x` across three players in the same window |
| `kreislaeuferspiel` | Kreisläufer-Anspiel | Backcourt circulation → quick 3-frame feed to pivot at x≈33 | 3-frame arc, ball Δx ≈ +5m forward into goal zone, pivot stationary at x>31 |

### The 3 defensive formations (Team B)

Team B's defenders form a curved arc in front of their goal. The shape matches
the thresholds in `ml/analysis/formation.py`.

| `formation` value | Name | Shape |
|---|---|---|
| `6-0` | Compact line | All 6 defenders in a curved arc at x≈32–34; wings closer to goal, center furthest out |
| `5-1` | Ausputzer | 5-line at x≈33–34.5 (curved) + 1 Ausputzer stepped out to x≈28 toward ball handler |
| `4-2` | Zone | Wide y-spread (>10m), two staggered depth levels |

### Scenario mix (default weights)

| Scenario | Weight |
|----------|--------|
| Kreisläufer-Anspiel | 20% |
| Kreuzung, Rückpass, Doppelpass, Parallelstoß | 13% each |
| 6-0 Defense | 10% |
| 5-1 Defense, 4-2 Defense | 9% each |

---

## Exploring the Data

### In the terminal (harlequin TUI)

```bash
just explore
```

Opens a full SQL editor against the DuckDB file with schema browser and tab completion.

### Key queries

```sql
-- What scenarios appear in a match, and when?
SELECT scenario, start_frame, ROUND(start_frame/25.0, 1) start_s,
       ROUND(duration_s, 1) dur_s
FROM scenario_labels
WHERE match_id = 'mock_0042_000'
ORDER BY start_frame;

-- Formation distribution per team
SELECT team, formation, COUNT(*) n_frames
FROM formations
GROUP BY team, formation
ORDER BY team, n_frames DESC;

-- Find all Kreisläufer-Anspiel segments and their frame ranges
SELECT sl.match_id, sl.start_frame, sl.end_frame, ROUND(sl.duration_s,1) dur_s
FROM scenario_labels sl
WHERE sl.scenario = 'kreislaeuferspiel';

-- Find Kreisläufer moments: player 6 near the 6m line for >1 second
SELECT match_id, track_id,
       MIN(timestamp_s) enter_s, MAX(timestamp_s) exit_s,
       COUNT(*) frames_near_6m
FROM players p
JOIN frames f USING (match_id, frame_id)
WHERE team='A' AND track_id=6
  AND court_x > 31.0
  AND sqrt(velocity_x*velocity_x + velocity_y*velocity_y) < 2.0
GROUP BY match_id, track_id
HAVING frames_near_6m > 25;

-- Possession phase summary
SELECT team, COUNT(*) n_phases,
       ROUND(AVG(duration_s),1) avg_s,
       ROUND(MAX(duration_s),1) max_s
FROM possession_phases
GROUP BY team;
```

### In Python

```python
import duckdb

conn = duckdb.connect("handball_mock.duckdb", read_only=True)

# Load all player frames for one scenario type (e.g. for clustering)
df = conn.execute("""
    SELECT p.match_id, p.frame_id, p.track_id, p.team,
           p.court_x, p.court_y, p.velocity_x, p.velocity_y,
           p.has_ball, sl.scenario
    FROM players p
    JOIN frames f USING (match_id, frame_id)
    JOIN scenario_labels sl
        ON p.match_id = sl.match_id
       AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
    WHERE p.court_x IS NOT NULL
    ORDER BY p.match_id, p.frame_id, p.track_id
""").fetchdf()

# Load labeled windows for supervised sequence classification
# scenario_labels is the ground truth class
windows = conn.execute("""
    SELECT sl.scenario,
           p.frame_id, p.track_id,
           p.court_x, p.court_y, p.velocity_x, p.velocity_y, p.has_ball
    FROM scenario_labels sl
    JOIN players p
        ON p.match_id = sl.match_id
       AND p.frame_id BETWEEN sl.start_frame AND sl.end_frame
    WHERE p.court_x IS NOT NULL
      AND sl.scenario != 'transition'
    ORDER BY sl.match_id, sl.start_frame, p.frame_id, p.track_id
""").fetchdf()
```

---

## Watching the Video

```bash
just visualize                        # 30s clip, ~8fps, from first match
just visualize dur=120 skip=3         # 2-minute clip, smoother
just visualize match=mock_0042_001    # different match
```

The video shows:
- **Blue circles**: Team A (Wels) — numbered by track ID
- **Red circles**: Team B (Linz) — curved defensive arc visible in 6-0
- **Gold ring** around a player: current ball carrier
- **Yellow circle**: ball
- **Velocity arrows**: on players moving >1.5 m/s
- **Top left**: Team A formation (`attack` / `transition`)
- **Top center**: current **Spielzug name** (e.g. "Kreisläufer-Anspiel")
- **Top right**: Team B formation (`6-0` / `5-1` / `4-2`)
- **Bottom center**: match clock

---

## ML Exploration Directions

This data is designed to support three types of ML experiments:

### 1. Unsupervised clustering (no labels needed)

Cluster 1-second windows of position/velocity data and check whether clusters
align with `scenario_labels`. This tests whether raw tracking data carries
enough structure to separate Spielzüge without supervision.

```python
# Build feature vectors: 12 players × 4 features (x, y, vx, vy) = 48-dim per frame
# Aggregate over a window (mean, std) → cluster with K-Means, DBSCAN, etc.
```

### 2. Supervised sequence classification

Use `scenario_labels.scenario` as class label. Train a sequence model (LSTM,
Transformer, or GCN+LSTM) on 25-frame windows to classify Spielzüge.
`action_labels` provides additional per-frame granularity.

```python
# 8 classes (5 Spielzüge + 3 defensive formations + transition)
# Per-window label from scenario_labels
```

### 3. Formation detection

`formations` provides rule-based labels every 5 frames. Use it to:
- Evaluate whether learned embeddings separate 6-0 / 5-1 / 4-2
- Fine-tune the rule-based thresholds to real data later

### Connecting to wels-monorepo

```bash
cd ../wels-monorepo
uv run wels-score mock_0042_000 \
  --db-path /path/to/mock-data-handball/handball_mock.duckdb
```

The real ML scoring pipeline (`wels-score`) reads directly from the generated
file — `formations` and `possession_phases` will be recomputed, `action_predictions`
will be populated if a trained checkpoint is available.

---

## Configuration Options

| CLI option | `just` parameter | Default | Description |
|---|---|---|---|
| `--matches` / `-n` | `n=` | 3 | Number of matches |
| `--duration` / `-d` | `d=` | 600 | Seconds per match |
| `--seed` | `seed=` | 42 | Random seed (fully reproducible) |
| `--fps` | — | 25 | Frames per second |
| `--team-a` | — | "Wels" | Team A name |
| `--team-b` | — | "Linz" | Team B name |
| `--two-pivots-prob` | — | 0.25 | Probability of 2-pivot variant in Kreisläufer scenarios |

---

## How the Generator Works

A match is assembled by picking **scenarios** from the configured mix and
stitching them together with 3.5-second transition segments.

```
Match = Scenario A + Transition + Scenario B + Transition + Scenario C + ...
```

Each scenario:
1. **Scripts waypoints** for the relevant players (e.g. CB drives forward, LB crosses)
2. **Interpolates positions** using cosine easing between waypoints (no teleportation)
3. **Non-scripted players** drift around their base position with spring dynamics
4. **Ball trajectory** is built explicitly: carry → arc → carry → arc
5. **Action events** are annotated per-frame (pass, dribble, hold)

After all frames are assembled:
- `compute_velocities()`: `vx = Δx × fps`, capped at 8.5 m/s
- `mark_ball_carriers()`: single nearest-to-ball player gets `has_ball=True`
- `apply_pixel_coords()`: court (m) → pixel for bbox columns

Then written to DuckDB: frames → players → ball → action_labels → action_predictions
→ formations → possession_phases → **scenario_labels**.

---

## Project Structure

```
mock-data-handball/
├── justfile                      ← Task runner (start here)
├── pyproject.toml
├── main.py
└── src/handball_mock/
    ├── config.py                 ← GeneratorConfig (all tunable parameters)
    ├── types.py                  ← Core dataclasses (PlayerFrame, MatchFrame, …)
    ├── court.py                  ← Court geometry constants + pixel transform
    ├── physics.py                ← smooth_path, arc_path, drift_path, velocities
    ├── labeler.py                ← Auto action-label assignment
    ├── generator.py              ← MatchGenerator: assembles scenarios into a match
    ├── writer.py                 ← DuckDB writer (all 9 tables + schema DDL)
    ├── visualizer.py             ← Renders match frames to MP4
    ├── cli.py                    ← typer CLI (generate / verify / render / scenarios)
    └── scenarios/
        ├── kreuzung.py           ← Kreuzung (Cross)
        ├── rueckpass.py          ← Rückpass + Durchbruch (Give-and-Go)
        ├── doppelpass.py         ← Doppelpass (Wall Pass)
        ├── parallelstos.py       ← Parallelstoß (Parallel Run)
        ├── kreislaeuferspiel.py  ← Kreisläufer-Anspiel (Pivot Entry)
        ├── defense_60.py         ← 6-0 defensive formation
        ├── defense_51.py         ← 5-1 defensive formation
        ├── defense_42.py         ← 4-2 defensive formation
        └── transition.py         ← Positional bridge between scenarios
```
