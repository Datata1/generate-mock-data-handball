# Supervised Learning — Training Guide

This guide explains how to train, evaluate, and interpret supervised models
for handball scenario classification. Read this before running anything.

---

## When is supervised learning worth it?

Supervised methods require **labelled data** — someone must annotate which
frames belong to which Spielzug. That annotation effort is only justified if
you get something unsupervised methods can't provide.

From the unsupervised experiments (`experiments/results/FINDINGS.md`):

| What you want | Unsupervised can do it? | Supervised needed? |
|---|---|---|
| Separate attack from defense | Yes (K-Means k=5, ARI=0.86) | No |
| Isolate kreuzung / kreislaeuferspiel | Yes (ARI>0.99) | No |
| Distinguish doppelpass vs kreislaeuferspiel | Partially (they merge in K-Means k=9) | Yes |
| Distinguish defense_60 vs rueckpass | Partially | Yes |
| Robust to real tracking noise | Unknown | Likely yes |

**Bottom line:** for the mock data, DTW achieves ARI=0.985 without labels.
Supervised adds value for the hard pairs and for real data where tracking
noise degrades unsupervised performance.

---

## The critical design constraint: split by match

**Never split by frame or segment.** If you do, training frames will be
adjacent to test frames within the same tactical sequence — the model sees
near-identical positions in both sets and the accuracy is artificially inflated.

**Correct approach: leave-one-match-out cross-validation**

```
Fold 0:  train on matches 1+2  →  test on match 0
Fold 1:  train on matches 0+2  →  test on match 1
Fold 2:  train on matches 0+1  →  test on match 2
```

With 3 matches (~74 segments each), each test fold has ~74 samples —
small enough that results have high variance. **Generate more matches first:**

```bash
just generate n=10 d=600 seed=42   # 10 matches → ~740 segments, much stabler
```

---

## Implemented methods (Tier 1 — Random Forest)

### Setup

No additional dependencies. scikit-learn is already installed.

```bash
just supervised-train                 # all 9 classes including transition
just supervised-train-clean           # 8 classes (transition excluded)
```

### What it does

1. Loads the same 81-d feature vectors used in unsupervised experiments
2. Runs leave-one-match-out cross-validation
3. Trains Random Forest (200 trees) and Gradient Boosting (150 trees)
4. Prints per-fold accuracy + F1, and mean across folds
5. Saves confusion matrix → `results/confusion_random_forest.png`
6. Saves metrics → `results/rf_metrics.json`

### Results on 3-match mock dataset

| Model | Mean Accuracy | F1 Macro | Std |
|---|---|---|---|
| **Random Forest** | **99.6%** | **99.7%** | ±0.6% |
| Gradient Boosting | 98.7% | 98.0% | ±1.1% |

vs. unsupervised:

| Method | ARI | Labels needed |
|---|---|---|
| DTW K-Means | 0.985 | None |
| RF (supervised) | 0.996 | Yes — `scenario_labels` |

**What this means:** on noise-free mock data, RF adds ~1% over DTW, for the
cost of needing labels. The confusion matrix is nearly diagonal — the hard pairs
that tripped up K-Means (doppelpass/kreislaeuferspiel, defense_60/rueckpass)
are now correctly separated.

**Important caveat:** mock data is scripted and nearly noise-free. On real
match data with CV tracking errors, expect accuracy to drop significantly
(perhaps to 75–90%). The relative ranking of methods should hold.

---

## Tier 2 — LSTM on raw frame sequences

### Why LSTM over RF

The Random Forest works on segment-level *statistics* (mean/std of positions).
It doesn't see the temporal order — the `doppelpass` 8-frame arc is detected
because the std of ball speed is high, not because the model sees the arc shape.

An LSTM processes the raw sequence frame by frame. It can learn:
- The characteristic velocity burst in `rueckpass` (happens at frame ~50/300)
- The 3-frame spike in `kreislaeuferspiel` (distinctive temporal shape)
- The crossing moment in `kreuzung`

Expected improvement on real noisy data: +5–10% accuracy over RF.

### Setup

Install PyTorch first:

```bash
# CPU only (sufficient for this dataset size):
pip install torch

# With GPU (faster training):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or if using wels-monorepo (already has torch):
cd ../wels-monorepo && uv sync && cd -
```

Verify:
```bash
uv run python -c "import torch; print(torch.__version__)"
```

### Run

```bash
just supervised-train-lstm            # 40 epochs (default)
just supervised-train-lstm epochs=80  # more epochs if accuracy plateaus
```

### Architecture

```
Input: (batch, 50 frames, 48 features)
  ↓  12 players × 4 features (court_x, court_y, velocity_x, velocity_y)
LSTM: hidden=128, num_layers=2, dropout=0.3
  ↓  last hidden state (batch, 128)
Linear: 128 → n_classes
  ↓  logits → CrossEntropyLoss
```

Each training sample: 50 frames (2 seconds) sampled uniformly from the segment.
Segments shorter than 50 frames are right-padded with the last frame.

**Hyperparameters to tune:**

| Parameter | Default | Try |
|---|---|---|
| `--epochs` | 40 | 80–120 if underfitting |
| `--lr` | 1e-3 | 5e-4 if loss oscillates |
| `--batch-size` | 16 | 32 with ≥10 matches |
| Window frames | 50 (hardcoded) | 25 (1s) or 75 (3s) |

### Checkpoints

One checkpoint per fold is saved to `results/lstm_fold{N}.pt`.
Each checkpoint contains the model weights and label encoder.

---

## Tier 3 — GCN + LSTM (`train_gcn.py`)

Implemented and ready to run. **No extra dependencies** — uses plain PyTorch,
not torch-geometric. The GCN is implemented with matrix multiplication.

### Architecture

```
Per-frame:
  12 players × 7 node features (court_x, court_y, vx, vy, has_ball, team_A, team_B)
  → k-NN graph (k=5, by court distance)
  → GCN layer (7 → 64) → GCN layer (64 → 128)
  → mean-pool over 12 players → 128-d frame embedding

Temporal:
  (50 frames, 128-d) → LSTM (hidden=128, layers=2) → last hidden state
  → Linear(128 → n_classes) → CrossEntropyLoss
```

### Why graph structure matters

The LSTM (Tier 2) gets a flat 48-d vector — 12 players concatenated.
It has no knowledge that player 6 is physically adjacent to players 11 and 12.

The GCN explicitly models spatial relationships. Message-passing aggregates
information from k nearest neighbours, so each player's embedding reflects
both its own features AND its local neighbourhood:

- **kreislaeuferspiel**: pivot (#6) surrounded by 3 defenders + near 6m line
  → GCN sees this as a unique local graph pattern
- **defense_60 vs 5-1**: graph connectivity differs (one player disconnected
  from the 5-line in 5-1 → isolated node pattern)
- **kreuzung**: tracks 3 and 4 exchange adjacency roles during crossing

### Run

```bash
just supervised-train-gcn              # 50 epochs (default)

# Custom epochs or hyperparameters:
PYTHONPATH=. uv run python experiments/supervised/train_gcn.py --epochs 100
PYTHONPATH=. uv run python experiments/supervised/train_gcn.py --epochs 100 --lr 1e-4 --no-transition
```

### Expected results and why GCN is slow to converge

| Epochs | Expected accuracy | Notes |
|--------|-----------------|-------|
| 20 | ~60% | Underfitting — predicts dominant class (transition) |
| 50 | ~75–85% | Learning scenario-specific graph patterns |
| 100 | ~85–95% | Near-RF performance on clean mock data |

**Why the GCN trains more slowly than RF:**

- RF uses 150 pre-computed statistics → fits in milliseconds, no gradient descent
- GCN has 274K parameters and learns from raw graphs → needs many gradient steps
- With only ~150 training segments per fold, the GCN risks overfitting

**Why GCN should win on REAL tracking data:**

Real CV data has position noise (±0.5–1m), occasional ID switches, and dropped
detections. The RF uses absolute mean positions — a 1m shift in court_x corrupts
the feature. The GCN uses k-NN graph topology: if a player is adjacent to 3
defenders, that fact is robust even if individual positions are shifted.

### Checkpoints

Saved to `results/gcn_fold{N}.pt` — weights + label encoder classes.

### Hyperparameters to tune

| Parameter | Default | Try |
|---|---|---|
| epochs | 50 | 100 if still improving at epoch 50 |
| lr | 5e-4 | 1e-3 if converging slowly, 1e-4 if loss oscillates |
| batch-size | 8 | 16 with ≥10 matches |
| k_neighbors | 5 | 3 (tighter) or 7 (looser) graph |

---

## Understanding the results

### What each output file shows

**`results/confusion_random_forest.png`**

A heatmap where rows = true scenario, columns = predicted scenario.
Values are percentages of the true scenario that were predicted as each class.

- **Dark red on the diagonal** = that scenario is correctly classified.
- **Dark red off the diagonal** = those two scenarios are being confused.

With mock data: nearly all diagonal. With real data: expect off-diagonal
errors between kreuzung/doppelpass and defense_60/rueckpass.

**`results/rf_metrics.json`**

```json
{
  "folds": [
    {"fold": 0, "test_match": "mock_0042_000", "accuracy": 1.0, "f1_macro": 1.0},
    ...
  ],
  "mean_by_model": {
    "Random Forest": {"mean_accuracy": 0.9963, "std_accuracy": 0.006}
  }
}
```

### Interpreting accuracy

| Accuracy | What it means for the coaching tool |
|---|---|
| > 95% | Reliable automatic tagging — coaches can trust it |
| 85–95% | Useful with manual review of low-confidence predictions |
| 70–85% | Exploratory use only — needs labelling |
| < 70% | More labelled data or better features needed |

### The variance problem

With 3 matches and ~74 test segments per fold, one mislabelled segment
changes accuracy by 1.3%. A ±0.6% std is unreliably small. The results
will be dramatically different for fold 1 vs fold 2.

**Before drawing conclusions, generate more data:**

```bash
just clean
just generate n=10 d=600 seed=42   # ~740 segments
just supervised-train
```

With 10 matches (7 train / 3 test per fold via k-fold), the std should
drop below ±2%.

---

## Step-by-step: training your first model

```bash
# Step 1: Make sure you have data
just verify

# Step 2 (optional but recommended): generate more matches for stable results
just generate n=10 d=600 seed=42

# Step 3: Train Random Forest baseline
just supervised-train

# Step 4: Read the confusion matrix
# open experiments/supervised/results/confusion_random_forest.png
# Look for off-diagonal red cells — those are the confused pairs

# Step 5: If accuracy is already >95%, you're done for mock data
# If accuracy is <85%, investigate which pairs are confused and why

# Step 6: Try without transition segments (cleaner problem)
just supervised-train-clean

# Step 7 (if torch installed): train LSTM
just supervised-train-lstm

# Step 8: Compare RF vs LSTM on the same folds
# Check if LSTM improves on the confused pairs from Step 4
```

---

## Roadmap: from mock data to real match data

When the CV pipeline produces real match data and writes it to the DuckDB:

1. **Re-run `just supervised-train`** — the code reads directly from DuckDB,
   no changes needed. Match IDs will be real video IDs instead of `mock_*`.

2. **Expect accuracy to drop** — real tracking has noise. Start by checking
   which scenarios are hardest to classify.

3. **Add more labelled matches** — each additional labelled match adds ~74
   training segments. Target: 10+ labelled matches for stable results.

4. **Consider semi-supervised**: label only 2–3 matches, use DTW to
   pre-segment the rest, then verify with the RF classifier.

5. **Feature engineering**: if accuracy is below 80%, add features that
   capture temporal dynamics:
   - Velocity variance (std over the segment, not just mean)
   - Ball arc length (how far does the ball travel per second)
   - Crossing angle (when two players exchange lanes in kreuzung)

---

## File structure

```
experiments/supervised/
├── README.md          ← this file
├── dataset.py         ← leave-one-match-out data loader
├── train_rf.py        ← Tier 1: Random Forest + Gradient Boosting
├── train_lstm.py      ← Tier 2: LSTM (requires torch)
└── results/           ← gitignored
    ├── confusion_random_forest.png
    ├── rf_metrics.json
    ├── lstm_fold0.pt   ← (after Tier 2)
    └── lstm_metrics.json
```
