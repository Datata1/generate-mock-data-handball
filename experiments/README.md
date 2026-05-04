# Experiments

ML experiments using the generated handball mock data.

## Structure

```
experiments/
├── features.py        ← shared: segment + window feature extraction from DuckDB
├── evaluate.py        ← shared: ARI, NMI, silhouette, confusion matrix helpers
├── plot.py            ← shared: scatter plots, heatmaps, colour palette
│
├── unsupervised/      ← clustering without labels (K-Means, UMAP, DTW)
│   ├── README.md      ← method guide + interpretation
│   ├── baseline.py    ← Tier 1: K-Means + GMM
│   ├── umap_cluster.py← Tier 2: UMAP+HDBSCAN + DTW K-Means
│   ├── gcn_cluster.py ← Tier 3: GCN embeddings (optional, needs torch)
│   ├── run_all.py     ← orchestrator
│   └── results/       ← gitignored: plots + metrics.json + FINDINGS.md
│
└── supervised/        ← classification with scenario_labels as ground truth
    ├── README.md      ← training guide (start here)
    ├── dataset.py     ← leave-one-match-out data loader
    ├── train_rf.py    ← Tier 1: Random Forest + Gradient Boosting
    ├── train_lstm.py  ← Tier 2: LSTM (requires torch)
    └── results/       ← gitignored: confusion matrix + metrics.json + FINDINGS.md
```

## Quick commands

```bash
# Unsupervised
just experiment               # all unsupervised tiers
just experiment-cluster       # K-Means only (~30s)
just experiment-viz           # UMAP + DTW

# Supervised
just supervised-train         # Random Forest baseline (no torch needed)
just supervised-train-lstm    # LSTM (requires torch)
```

## After running — read the findings

Both experiments generate a `results/FINDINGS.md` explaining what each plot
shows, what the numbers mean, and what to do next. Open it after each run.
