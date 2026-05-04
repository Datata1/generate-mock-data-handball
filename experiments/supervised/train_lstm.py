"""Tier 2 supervised: LSTM on fixed-length frame sequences.

Requires PyTorch. Install with:
    pip install torch
or, if using the wels-monorepo:
    cd ../wels-monorepo && uv sync && cd -

This file is a ready-to-run implementation. It will print a clear error
if torch is not installed and explain what to do.

Architecture:
  - Input: (batch, T=50, 48) — 50 frames × 12 players × 4 features
  - LSTM: hidden=128, num_layers=2, dropout=0.3
  - Classifier: Linear(128 → n_classes)

Why LSTM over Random Forest:
  - Captures temporal dynamics (e.g. the velocity burst in rueckpass)
  - The doppelpass 8-frame arc is a TIME pattern, not a position pattern
  - Expected improvement: +5–10% accuracy on the hard pairs

Usage:
    PYTHONPATH=. uv run python experiments/supervised/train_lstm.py
    PYTHONPATH=. uv run python experiments/supervised/train_lstm.py --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

RESULTS_DIR = Path("experiments/supervised/results")

# ── Torch availability check ──────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Minimal stubs so class definitions (Dataset, nn.Module) don't fail at
    # import time. All runtime code is guarded by _TORCH_AVAILABLE checks.
    class _Stub:  # type: ignore[misc]
        pass
    class _StubNN:  # type: ignore[misc]
        Module = _Stub
    nn = _StubNN()  # type: ignore[assignment]
    Dataset = _Stub  # type: ignore[assignment,misc]
    DataLoader = None  # type: ignore[assignment]


def _torch_not_available() -> None:
    print("""
┌─────────────────────────────────────────────────────────────────┐
│  PyTorch is not installed.                                      │
│                                                                 │
│  Install it with:                                               │
│    pip install torch                                            │
│  or:                                                            │
│    uv add torch                                                 │
│    uv sync                                                      │
│                                                                 │
│  For CUDA (GPU training):                                       │
│    pip install torch --index-url https://download.pytorch.org/whl/cu121
│                                                                 │
│  Then re-run:                                                   │
│    just supervised-train-lstm                                   │
└─────────────────────────────────────────────────────────────────┘
""")
    sys.exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────────

# Window of N frames per segment, sampled uniformly
WINDOW_FRAMES = 50      # 2 seconds at 25 fps
FIELD_TRACKS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]  # 12 players, no GKs
FEATURES_PER_PLAYER = 4  # court_x, court_y, velocity_x, velocity_y
INPUT_DIM = len(FIELD_TRACKS) * FEATURES_PER_PLAYER  # 48


class ScenarioWindowDataset(Dataset):
    """Samples WINDOW_FRAMES uniformly from each scenario segment.

    Each item: (tensor(WINDOW_FRAMES, 48), label_int)

    Why sample uniformly rather than taking the full segment?
    - Segment lengths vary (87 to 450 frames) — LSTM needs fixed length
    - Uniform sampling preserves temporal order and avoids padding artefacts
    - At inference time, you'd slide a 50-frame window over the match

    Returns nan-filled positions as 0.0 (GK positions are always nan).
    """

    def __init__(
        self,
        conn,
        match_ids: list[str],
        label_encoder,
        window_frames: int = WINDOW_FRAMES,
        exclude_transition: bool = False,
        rng_seed: int = 42,
    ):
        import numpy as np

        self.window_frames = window_frames
        self.label_encoder = label_encoder
        self.items: list[tuple[np.ndarray, int]] = []
        rng = np.random.default_rng(rng_seed)

        # Load scenario spans for the selected matches
        placeholders = ",".join(["?"] * len(match_ids))
        spans = conn.execute(
            f"SELECT match_id, start_frame, end_frame, scenario FROM scenario_labels "
            f"WHERE match_id IN ({placeholders})",
            match_ids,
        ).fetchall()

        for match_id, start_frame, end_frame, scenario in spans:
            if exclude_transition and scenario == "transition":
                continue
            if scenario not in label_encoder.classes_:
                continue

            label = int(label_encoder.transform([scenario])[0])
            n_frames = end_frame - start_frame + 1

            # Sample uniformly from the segment
            if n_frames >= window_frames:
                start_idx = rng.integers(0, n_frames - window_frames + 1)
                frame_ids = list(range(start_frame + start_idx, start_frame + start_idx + window_frames))
            else:
                # Segment shorter than window — repeat last frame to pad
                frame_ids = list(range(start_frame, end_frame + 1))
                while len(frame_ids) < window_frames:
                    frame_ids.append(frame_ids[-1])

            # Query player features for these frames
            ph = ",".join(["?"] * len(frame_ids))
            rows = conn.execute(
                f"""SELECT frame_id, track_id, court_x, court_y, velocity_x, velocity_y
                    FROM players
                    WHERE match_id=? AND frame_id IN ({ph}) AND track_id IN ({",".join(map(str, FIELD_TRACKS))})
                    ORDER BY frame_id, track_id""",
                [match_id] + frame_ids,
            ).fetchall()

            # Build tensor: (window_frames, 48)
            tensor = np.zeros((window_frames, INPUT_DIM), dtype=np.float32)
            frame_map = {f: i for i, f in enumerate(frame_ids)}
            track_map = {t: i for i, t in enumerate(FIELD_TRACKS)}

            for frame_id, track_id, cx, cy, vx, vy in rows:
                if frame_id not in frame_map or track_id not in track_map:
                    continue
                fi = frame_map[frame_id]
                ti = track_map[track_id]
                base = ti * FEATURES_PER_PLAYER
                tensor[fi, base]     = cx if cx is not None else 0.0
                tensor[fi, base + 1] = cy if cy is not None else 0.0
                tensor[fi, base + 2] = vx if vx is not None else 0.0
                tensor[fi, base + 3] = vy if vy is not None else 0.0

            self.items.append((tensor, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        tensor, label = self.items[idx]
        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class ScenarioLSTM(nn.Module):
    """LSTM classifier for handball scenario sequences.

    Input:  (batch, T, 48) — T frames × 12-player feature vectors
    Output: (batch, n_classes) logits

    The LSTM processes the sequence left-to-right and we take the last
    hidden state as the sequence representation. A 2-layer LSTM with
    dropout between layers helps regularise on the small dataset.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 9,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, T, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        return self.classifier(self.dropout(last_hidden))


# ── Training loop ──────────────────────────────────────────────────────────────

def run(
    db_path: str,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 16,
    exclude_transition: bool = False,
) -> None:
    import numpy as np
    import duckdb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from rich.console import Console
    from rich.rule import Rule

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Rule("[bold]LSTM Scenario Classifier[/bold]"))
    console.print(f"  Device: [cyan]{device}[/cyan]  |  Epochs: {epochs}  |  LR: {lr}")

    conn = duckdb.connect(db_path, read_only=True)
    match_ids = [r[0] for r in conn.execute("SELECT DISTINCT match_id FROM matches").fetchall()]

    # Fit label encoder on all scenarios (important: same across all folds)
    all_scenarios = conn.execute(
        "SELECT DISTINCT scenario FROM scenario_labels"
    ).fetchdf()["scenario"].tolist()
    if exclude_transition:
        all_scenarios = [s for s in all_scenarios if s != "transition"]
    all_scenarios.sort()
    le = LabelEncoder()
    le.fit(all_scenarios)
    n_classes = len(le.classes_)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_id, test_match in enumerate(match_ids):
        train_matches = [m for m in match_ids if m != test_match]
        console.print(Rule(f"Fold {fold_id} — test: {test_match}"))

        train_ds = ScenarioWindowDataset(conn, train_matches, le, exclude_transition=exclude_transition)
        test_ds  = ScenarioWindowDataset(conn, [test_match],  le, exclude_transition=exclude_transition)

        if len(train_ds) == 0 or len(test_ds) == 0:
            console.print("  [red]Empty fold, skipping[/red]")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        model = ScenarioLSTM(n_classes=n_classes).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimiser.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                train_loss += loss.item()
            scheduler.step()

            # Validation (on test fold — only for early stopping, not model selection)
            model.eval()
            y_true_v, y_pred_v = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    logits = model(X_batch.to(device))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    y_pred_v.extend(preds)
                    y_true_v.extend(y_batch.numpy())

            val_acc = accuracy_score(y_true_v, y_pred_v)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                console.print(
                    f"  Epoch {epoch+1:3d}/{epochs}  loss={train_loss/len(train_loader):.3f}"
                    f"  val_acc={val_acc:.3f}"
                )

        # Final evaluation with best weights
        model.load_state_dict(best_state)
        model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch.to(device))
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred_all.extend(preds)
                y_true_all.extend(y_batch.numpy())

        y_true_names = le.inverse_transform(y_true_all)
        y_pred_names = le.inverse_transform(y_pred_all)
        acc = accuracy_score(y_true_names, y_pred_names)
        f1 = f1_score(y_true_names, y_pred_names, average="macro", zero_division=0)

        from sklearn.metrics import classification_report as _cr
        report_dict = _cr(y_true_names, y_pred_names, output_dict=True, zero_division=0)
        per_class_f1 = {
            k: round(v["f1-score"], 4)
            for k, v in report_dict.items()
            if isinstance(v, dict) and "f1-score" in v
        }

        console.print(f"\n  [bold]Best val acc={best_val_acc:.3f}[/bold]  Test acc={acc:.3f}  F1={f1:.3f}")
        console.print(classification_report(y_true_names, y_pred_names, zero_division=0))

        fold_results.append({"fold": fold_id, "test_match": test_match,
                              "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
                              "per_class_f1": per_class_f1})

        # Save checkpoint
        ckpt_path = RESULTS_DIR / f"lstm_fold{fold_id}.pt"
        torch.save({"state_dict": best_state, "label_encoder": le.classes_.tolist()}, ckpt_path)
        console.print(f"  Checkpoint → {ckpt_path}")

    conn.close()

    if fold_results:
        console.print(Rule("[bold]LSTM Cross-validation Summary[/bold]"))
        accs = [r["accuracy"] for r in fold_results]
        f1s  = [r["f1_macro"] for r in fold_results]
        console.print(f"  Mean accuracy: [green]{np.mean(accs):.3f}[/green]  ±{np.std(accs):.3f}")
        console.print(f"  Mean F1 macro: {np.mean(f1s):.3f}")

        (RESULTS_DIR / "lstm_metrics.json").write_text(json.dumps({
            "folds": fold_results,
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy":  round(float(np.std(accs)),  4),
            "mean_f1_macro": round(float(np.mean(f1s)),  4),
        }, indent=2))


if __name__ == "__main__":
    if not _TORCH_AVAILABLE:
        _torch_not_available()

    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="handball_mock.duckdb")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-transition", action="store_true")
    args = parser.parse_args()
    run(args.db, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        exclude_transition=args.no_transition)
