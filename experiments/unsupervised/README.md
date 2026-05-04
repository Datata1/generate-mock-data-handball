# Unsupervised ML Experiments

Tests whether **raw position/velocity data alone** can cluster handball plays
into their correct tactical categories — without using the `scenario_labels`
ground truth during training, only for final evaluation.

This is the first ML validation step: if the data clusters meaningfully, the
supervised path (GCN+LSTM in wels-monorepo) is worth pursuing.

---

## Setup

```bash
# Install ML experiment dependencies (first time only)
just install-experiments
```

Adds: `scikit-learn`, `umap-learn`, `seaborn`, `tslearn`.
These are optional extras — the base `handball-mock` package stays lightweight.

---

## Running experiments

```bash
just experiment-cluster     # Tier 1: K-Means + GMM (~30 seconds)
just experiment-viz         # Tier 2: UMAP + HDBSCAN + DTW (1–2 minutes)
just experiment             # All tiers (Tier 3 skipped if torch not installed)

# Or directly:
uv run python experiments/run_all.py --tier 1
uv run python experiments/run_all.py --db other.duckdb --tier all
```

---

## What each tier tests

### Tier 1 — Segment statistics + K-Means / GMM

**Runtime:** ~15–30 seconds

**Idea:** Represent each scenario segment as an ~81-dimensional feature vector
of per-player position and velocity statistics. Cluster with K-Means and GMM.

**Features per segment:**
- 12 field players (tracks 1–6, 8–13) × (mean\_x, std\_x, mean\_y, std\_y, mean\_speed, max\_speed)
- Team A/B centroids, Team B x-variance (key 6-0/5-1/4-2 discriminator)
- Track 6 mean\_x (Kreisläufer position — distinctive for `kreislaeuferspiel`)
- Ball mean position, possession rate

**GMM note:** requires PCA to 15 dimensions first — 81 features with 9 components
on 223 samples gives singular covariance matrices with `covariance_type='full'`.

**Outputs:** `results/pca_true_scenarios.png`, `results/kmeans_elbow.png`,
`results/confusion_kmeans_best.png`

---

### Tier 2 — UMAP + HDBSCAN + DTW K-Medoids

**Runtime:** ~1–2 minutes

**2a. UMAP + HDBSCAN**

Best for visual validation. UMAP reduces the 81-dimensional feature space to 2D
while preserving neighbourhood structure. HDBSCAN then finds density clusters
without needing to specify k in advance.

Pipeline: `StandardScaler → PCA (95% variance) → UMAP(2D) → HDBSCAN`

The PCA step before UMAP is mandatory: without it, UMAP on 81 dimensions with
N=3,700 windows takes ~10 minutes vs. ~30 seconds after PCA to 20 dims.

**Outputs:** `results/umap_true_scenarios.png`, `results/umap_hdbscan_clusters.png`,
`results/confusion_umap.png`

**2b. DTW K-Medoids on ball trajectory**

Time-series aware. Each scenario segment has a ball (x, y) time series of
variable length. DTW measures similarity between trajectories accounting for
temporal distortion — a doppelpass with a tight 8-frame arc looks different
to a kreislaeuferspiel with a single 3-frame spike even if they're the same length.

Why this matters:
| Scenario | Ball trajectory shape |
|---|---|
| doppelpass | tight zig-zag, fast turnaround |
| kreislaeuferspiel | slow circulation then spike forward 5m |
| parallelstos | smooth straight-forward movement |
| defense plays | slow back-and-forth at x=26–28 |

**Outputs:** `results/dtw_pca_true_scenarios.png`, `results/dtw_clusters.png`

---

### Tier 3 — GCN frame embeddings (optional, requires torch)

**Runtime:** ~2–5 minutes if torch installed

Uses the wels-monorepo ActionPredictor's GCN encoder to extract 128-dimensional
frame embeddings, mean-pooled over a scenario segment.

**Important:** model weights are **random (untrained)**. Expected ARI ≈ 0.
This tier documents the infrastructure — run it again once `wels-train`
produces a checkpoint and results will be meaningful.

To enable: install `torch` and `torch-geometric` (either directly or via
`cd ../wels-monorepo && uv sync`).

---

## Interpreting results

After running, open **`experiments/results/FINDINGS.md`** — it is generated
automatically and contains a plain-language explanation of every plot with
specific observations about what the algorithm found and what it means for
unsupervised usefulness.

All outputs are saved to `experiments/results/` (gitignored). Summary metrics
are also written to `experiments/results/metrics.json`.

### ARI (Adjusted Rand Index) — primary metric

| ARI | Interpretation |
|-----|---------------|
| > 0.60 | Excellent — scenario structure is clearly encoded in position data |
| 0.30–0.60 | Meaningful — some scenarios separable, others confused |
| < 0.30 | Near-random — features insufficient or scenarios too similar |

### What to expect

**Easy to separate** (should be clean clusters):
- `kreislaeuferspiel`: pivot (track 6) at x≈33, stationary for 7+ seconds — very distinctive
- `parallelstos`: tracks 1, 2, 4 all show mean\_speed > 5 m/s simultaneously
- Defense vs. attack: team B x-variance is 70× different between 6-0 and 4-2

**Harder to separate** (expect some confusion):
- `kreuzung` vs `rueckpass` vs `doppelpass`: all backcourt plays at similar x positions;
  distinguishing features are velocity dynamics (std\_speed, max\_speed) not means
- `defense_60` vs `defense_51` vs `defense_42`: team B mean\_x is similar;
  team B x-variance is the key discriminator
- `transition`: will scatter across other clusters — it's a positional bridge segment

### NMI (Normalized Mutual Information)

Companion to ARI — both should move together. If ARI and NMI diverge significantly,
the clustering has highly imbalanced cluster sizes.

### Silhouette score

Internal metric (no ground truth needed). Values 0–1; higher is better.
Useful for tuning k even before looking at ARI.

---

## Output files

```
experiments/results/
├── pca_true_scenarios.png        — PCA 2D, coloured by true scenario
├── kmeans_elbow.png              — K-Means inertia for k=5..11
├── kmeans_ari_by_k.png           — ARI vs k bar chart
├── confusion_kmeans_best.png     — confusion heatmap, best K-Means
├── umap_true_scenarios.png       — UMAP 2D, coloured by true scenario
├── umap_hdbscan_clusters.png     — UMAP 2D, coloured by HDBSCAN cluster
├── confusion_umap.png            — confusion heatmap, UMAP+HDBSCAN
├── dtw_pca_true_scenarios.png    — DTW features PCA, true scenarios
├── dtw_clusters.png              — DTW K-Medoids cluster assignments
├── gcn_pca_true_scenarios.png    — GCN embedding PCA (if torch available)
└── metrics.json                  — all metrics as JSON for comparison
```

---

## Module structure

```
experiments/
├── features.py       — segment + ball trajectory feature extraction from DuckDB
├── evaluate.py       — ARI, NMI, silhouette, confusion matrix, ClusterResult
├── plot.py           — scatter plots, confusion heatmaps, elbow curve
├── baseline.py       — Tier 1: K-Means, GMM, PCA scatter
├── umap_cluster.py   — Tier 2: UMAP+HDBSCAN, DTW K-Medoids
├── gcn_cluster.py    — Tier 3: GCN embeddings (optional)
└── run_all.py        — CLI orchestrator
```

---

## Extending the experiments

**Add more features:** edit `features.py::build_segment_features`. Ideas:
- Crossing angle between track 3 and track 4 trajectories (Kreuzung signature)
- Velocity correlation between tracks 1, 2, 4 (Parallelstoß signature)
- Pass event count per segment (from `action_labels`)

**Add a method:** create a new function in `umap_cluster.py` or a new module,
import it in `run_all.py`, append results to `all_results`.

**Use window-level features:** call `build_window_features(conn)` in `run_all.py`
instead of `build_segment_features` — produces ~3,700 samples for richer UMAP.
