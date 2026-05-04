# handball-mock — task runner
# Install just: cargo install just  OR  apt install just  OR  brew install just
# Usage: just <recipe>

set dotenv-load 

db := "handball_mock.duckdb"

# List available recipes (default)
help:
    @just --list

# Install Python dependencies and harlequin DuckDB explorer
install:
    uv sync
    uv tool install harlequin 2>/dev/null || true

# Generate mock data  (override: just generate n=5 d=300 seed=99)
generate n="3" d="600" seed="42" out=db:
    uv run handball-mock generate {{out}} --matches {{n}} --duration {{d}} --seed {{seed}} --verbose

# Verify consistency of a generated DuckDB file
verify file=db:
    uv run handball-mock verify {{file}}

# Open the DuckDB TUI explorer (harlequin)
explore file=db:
    harlequin {{file}}

# Open the DuckDB web UI in the browser (DuckDB built-in)
explore-web file=db:
    duckdb --ui {{file}}

# Render a 30-second video of the first match in handball_mock.duckdb
# Override: just visualize match=mock_0042_001 out=clip.mp4 dur=60
visualize match="mock_0042_000" out="preview.mp4" dur="30" skip="3":
    uv run handball-mock render {{db}} {{match}} --output {{out}} --duration {{dur}} --frame-skip {{skip}}
    @echo "→ Open {{out}} in your video player"

# Generate + verify in one step
all n="3" d="600" seed="42" out=db: (generate n d seed out) (verify out)

# Remove generated DuckDB files
clean:
    rm -f *.duckdb

# Show available scenario types
scenarios:
    uv run handball-mock scenarios

# Regenerate the default DB and render a fresh preview clip
regenerate n="3" d="600" seed="42": (all n d seed) (visualize)

# Quick smoke-test: 1 match, 2 minutes
smoke:
    uv run handball-mock generate /tmp/handball_smoke.duckdb -n 1 -d 120 --seed 1
    uv run handball-mock verify /tmp/handball_smoke.duckdb
