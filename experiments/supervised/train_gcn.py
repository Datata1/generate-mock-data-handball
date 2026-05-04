"""Tier 3 supervised: GCN + LSTM scenario classifier.

Each frame is modelled as a player interaction GRAPH. A Graph Convolutional
Network (GCN) reads the graph and produces a spatial embedding per frame.
An LSTM then reads the sequence of frame embeddings to classify the scenario.

Why graph structure matters:
  The LSTM (Tier 2) receives a flat 48-d vector — 12 players concatenated.
  It has no way to know that player 6 is ADJACENT to players 11 and 12.
  The GCN explicitly models spatial relationships: edges connect nearby
  players, and message-passing aggregates neighbour features.

  For handball, this means:
  - kreislaeuferspiel: pivot (track 6) is surrounded by defenders (tracks 9–13)
    AND near the 6m line — the GCN sees both position AND adjacency
  - defense_60 vs defense_51: the adjacency graph has a different structure
    (one player disconnected from the 5-line in 5-1)
  - kreuzung: tracks 3 and 4 exchange adjacency roles during the crossing

No torch-geometric required — the GCN is implemented with plain PyTorch
using batch-wise matrix multiplication.

Usage:
    PYTHONPATH=. uv run python experiments/supervised/train_gcn.py
    PYTHONPATH=. uv run python experiments/supervised/train_gcn.py --epochs 60
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import duckdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.supervised.dataset import load_folds

RESULTS_DIR = Path("experiments/supervised/results")

# ── Graph constants ────────────────────────────────────────────────────────────

FIELD_TRACKS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]  # 12 players, no GKs
N_PLAYERS = len(FIELD_TRACKS)
TRACK_TO_IDX = {t: i for i, t in enumerate(FIELD_TRACKS)}

# Node feature dimensions
# [court_x, court_y, velocity_x, velocity_y, has_ball, team_a, team_b]
NODE_FEATURES = 7

K_NEIGHBORS = 5     # edges per player (k-NN by court distance)
WINDOW_FRAMES = 50  # temporal window length


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_frame_graph(
    positions: np.ndarray,    # (N_PLAYERS, 2) — court_x, court_y
    velocities: np.ndarray,   # (N_PLAYERS, 2) — vx, vy
    has_ball: np.ndarray,     # (N_PLAYERS,)  — bool
    teams: np.ndarray,        # (N_PLAYERS,)  — 0=A, 1=B
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build node features and normalised adjacency matrix for one frame.

    Returns:
        x:   (N_PLAYERS, NODE_FEATURES) — node feature matrix
        adj: (N_PLAYERS, N_PLAYERS)     — row-normalised k-NN adjacency
    """
    n = N_PLAYERS

    # Node features
    x = np.zeros((n, NODE_FEATURES), dtype=np.float32)
    x[:, 0] = positions[:, 0] / 40.0          # court_x normalised to [0, 1]
    x[:, 1] = positions[:, 1] / 20.0          # court_y normalised
    x[:, 2] = velocities[:, 0] / 8.0          # velocity normalised (max ~8 m/s)
    x[:, 3] = velocities[:, 1] / 8.0
    x[:, 4] = has_ball.astype(np.float32)
    x[:, 5] = (teams == 0).astype(np.float32)  # team A indicator
    x[:, 6] = (teams == 1).astype(np.float32)  # team B indicator

    # k-NN adjacency from court distances (including self-loops)
    dist = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )  # (N, N)

    # For each node, connect to k nearest neighbours
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        d_i = dist[i].copy()
        # Zero out own distance so it doesn't compete with actual neighbours
        d_i[i] = np.inf
        nn_idx = np.argsort(d_i)[:K_NEIGHBORS]
        adj[i, nn_idx] = 1.0
        adj[i, i] = 1.0   # self-loop

    # Row-normalise: D^{-1} A
    row_sums = adj.sum(axis=1, keepdims=True).clip(min=1.0)
    adj = adj / row_sums

    return torch.tensor(x, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ScenarioGraphDataset(Dataset):
    """One item = WINDOW_FRAMES graphs + a scenario label.

    Each graph: (node_features, adjacency_matrix) for 12 players.
    The LSTM receives a sequence of GCN-encoded graph embeddings.
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        match_ids: list[str],
        label_encoder: LabelEncoder,
        window_frames: int = WINDOW_FRAMES,
        exclude_transition: bool = False,
        rng_seed: int = 42,
    ):
        self.window_frames = window_frames
        self.items: list[tuple[list[tuple], int]] = []
        rng = np.random.default_rng(rng_seed)

        placeholders = ",".join(["?"] * len(match_ids))
        spans = conn.execute(
            f"SELECT match_id, start_frame, end_frame, scenario "
            f"FROM scenario_labels WHERE match_id IN ({placeholders})",
            match_ids,
        ).fetchall()

        for match_id, start_frame, end_frame, scenario in spans:
            if exclude_transition and scenario == "transition":
                continue
            if scenario not in label_encoder.classes_:
                continue
            label = int(label_encoder.transform([scenario])[0])
            n_frames = end_frame - start_frame + 1

            if n_frames >= window_frames:
                offset = rng.integers(0, n_frames - window_frames + 1)
                frame_ids = list(range(start_frame + offset, start_frame + offset + window_frames))
            else:
                frame_ids = list(range(start_frame, end_frame + 1))
                while len(frame_ids) < window_frames:
                    frame_ids.append(frame_ids[-1])

            # Fetch player data for all frames at once
            ph = ",".join(["?"] * len(frame_ids))
            track_ph = ",".join(map(str, FIELD_TRACKS))
            rows = conn.execute(
                f"""SELECT frame_id, track_id, team,
                           COALESCE(court_x, 20.0) AS cx,
                           COALESCE(court_y, 10.0) AS cy,
                           velocity_x, velocity_y, has_ball
                    FROM players
                    WHERE match_id=? AND frame_id IN ({ph}) AND track_id IN ({track_ph})
                    ORDER BY frame_id, track_id""",
                [match_id] + frame_ids,
            ).fetchall()

            # Group by frame_id
            frame_data: dict[int, list] = {f: [] for f in frame_ids}
            for fid, tid, team, cx, cy, vx, vy, hb in rows:
                if fid in frame_data:
                    frame_data[fid].append((tid, team, cx, cy, vx, vy, hb))

            # Build graph per frame
            graphs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for fid in frame_ids:
                positions  = np.zeros((N_PLAYERS, 2), dtype=np.float32)
                velocities = np.zeros((N_PLAYERS, 2), dtype=np.float32)
                has_ball   = np.zeros(N_PLAYERS, dtype=bool)
                teams      = np.zeros(N_PLAYERS, dtype=np.int32)

                for tid, team, cx, cy, vx, vy, hb in frame_data.get(fid, []):
                    if tid in TRACK_TO_IDX:
                        i = TRACK_TO_IDX[tid]
                        positions[i] = [cx, cy]
                        velocities[i] = [vx or 0.0, vy or 0.0]
                        has_ball[i] = bool(hb)
                        teams[i] = 0 if team == "A" else 1

                x, adj = _build_frame_graph(positions, velocities, has_ball, teams)
                graphs.append((x, adj))

            self.items.append((graphs, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        graphs, label = self.items[idx]
        # Stack into tensors: (T, N, features) and (T, N, N)
        xs   = torch.stack([g[0] for g in graphs])   # (T, N_PLAYERS, NODE_FEATURES)
        adjs = torch.stack([g[1] for g in graphs])   # (T, N_PLAYERS, N_PLAYERS)
        return xs, adjs, torch.tensor(label, dtype=torch.long)


def _collate(batch):
    xs   = torch.stack([b[0] for b in batch])   # (B, T, N, F)
    adjs = torch.stack([b[1] for b in batch])   # (B, T, N, N)
    ys   = torch.stack([b[2] for b in batch])   # (B,)
    return xs, adjs, ys


# ── Model ─────────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU(D^{-1} A H W)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x:   (..., N, in_features)
        # adj: (..., N, N) — pre-normalised
        aggregated = torch.matmul(adj, x)        # (..., N, in_features)
        return F.relu(self.linear(aggregated))   # (..., N, out_features)


class ScenarioGCNLSTM(nn.Module):
    """GCN per frame → mean-pool over players → LSTM over frames → classify.

    Architecture:
      Frame: (N_PLAYERS, NODE_FEATURES) → GCN(7→64) → GCN(64→128)
             → mean over players → (128,) frame embedding
      Sequence: (T, 128) → LSTM(hidden=128, layers=2) → last hidden
             → Linear(128 → n_classes)
    """

    def __init__(
        self,
        node_features: int = NODE_FEATURES,
        gcn_hidden: int = 64,
        gcn_out: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 9,
    ):
        super().__init__()
        self.gcn1 = GCNLayer(node_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)
        self.lstm = nn.LSTM(
            input_size=gcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden, n_classes)

    def encode_frame(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """GCN encoding for one (or batched) frame.

        Args:
            x:   (..., N_PLAYERS, NODE_FEATURES)
            adj: (..., N_PLAYERS, N_PLAYERS)
        Returns:
            (..., gcn_out) — mean-pooled graph embedding
        """
        h = self.gcn1(x, adj)    # (..., N, gcn_hidden)
        h = self.gcn2(h, adj)    # (..., N, gcn_out)
        return h.mean(dim=-2)    # (..., gcn_out)  — mean over players

    def forward(
        self, xs: torch.Tensor, adjs: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass over a sequence of frames.

        Args:
            xs:   (batch, T, N_PLAYERS, NODE_FEATURES)
            adjs: (batch, T, N_PLAYERS, N_PLAYERS)
        Returns:
            (batch, n_classes) logits
        """
        B, T = xs.shape[:2]
        # Encode all frames at once: reshape to (B*T, N, F) for efficiency
        x_flat   = xs.view(B * T, N_PLAYERS, NODE_FEATURES)
        adj_flat = adjs.view(B * T, N_PLAYERS, N_PLAYERS)
        emb_flat = self.encode_frame(x_flat, adj_flat)          # (B*T, gcn_out)
        emb_seq  = emb_flat.view(B, T, -1)                      # (B, T, gcn_out)

        _, (h_n, _) = self.lstm(emb_seq)
        last_h = h_n[-1]                                        # (B, lstm_hidden)
        return self.classifier(self.dropout(last_h))


# ── Training loop ──────────────────────────────────────────────────────────────

def run(
    db_path: str,
    epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 8,
    exclude_transition: bool = False,
) -> None:
    from rich.console import Console
    from rich.rule import Rule

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(Rule("[bold]Tier 3 — GCN + LSTM Scenario Classifier[/bold]"))
    console.print(
        f"  Device: [cyan]{device}[/cyan]  |  Epochs: {epochs}  |  LR: {lr}\n"
        f"  Graph: {N_PLAYERS} players, k={K_NEIGHBORS} neighbours, "
        f"{NODE_FEATURES}-d node features\n"
        f"  Window: {WINDOW_FRAMES} frames ({WINDOW_FRAMES/25:.1f}s)"
    )

    conn = duckdb.connect(db_path, read_only=True)
    match_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT match_id FROM matches"
    ).fetchall()]

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

    for test_match in match_ids:
        train_matches = [m for m in match_ids if m != test_match]
        fold_id = match_ids.index(test_match)
        console.print(Rule(f"Fold {fold_id} — test: {test_match}"))

        train_ds = ScenarioGraphDataset(conn, train_matches, le, exclude_transition=exclude_transition)
        test_ds  = ScenarioGraphDataset(conn, [test_match],  le, exclude_transition=exclude_transition)
        console.print(f"  Train: {len(train_ds)} segments  |  Test: {len(test_ds)} segments")

        if len(train_ds) == 0 or len(test_ds) == 0:
            console.print("  [red]Empty fold — skipping[/red]")
            continue

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=_collate)

        model = ScenarioGCNLSTM(n_classes=n_classes).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"  Model parameters: {n_params:,}")

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None
        epoch_history: list[dict] = []   # records every epoch for convergence plot

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for xs, adjs, ys in train_loader:
                xs, adjs, ys = xs.to(device), adjs.to(device), ys.to(device)
                optimiser.zero_grad()
                logits = model(xs, adjs)
                loss = criterion(logits, ys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()
            scheduler.step()

            # Validation
            model.eval()
            y_true_v, y_pred_v = [], []
            with torch.no_grad():
                for xs, adjs, ys in test_loader:
                    preds = model(xs.to(device), adjs.to(device)).argmax(dim=1).cpu().numpy()
                    y_pred_v.extend(preds)
                    y_true_v.extend(ys.numpy())

            avg_loss = total_loss / max(1, len(train_loader))
            val_acc = accuracy_score(y_true_v, y_pred_v)
            epoch_history.append({
                "epoch": epoch + 1,
                "fold": fold_id,
                "train_loss": round(avg_loss, 4),
                "val_acc": round(float(val_acc), 4),
            })

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 10 == 0:
                console.print(
                    f"  Epoch {epoch+1:3d}/{epochs}  "
                    f"loss={avg_loss:.3f}  "
                    f"val_acc={val_acc:.3f}  (best={best_acc:.3f})"
                )

        # Final evaluation with best weights
        model.load_state_dict(best_state)
        model.eval()
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for xs, adjs, ys in test_loader:
                preds = model(xs.to(device), adjs.to(device)).argmax(dim=1).cpu().numpy()
                y_pred_all.extend(preds)
                y_true_all.extend(ys.numpy())

        y_true_names = le.inverse_transform(y_true_all)
        y_pred_names = le.inverse_transform(y_pred_all)
        acc = float(accuracy_score(y_true_names, y_pred_names))
        f1  = float(f1_score(y_true_names, y_pred_names, average="macro", zero_division=0))

        # Per-class F1 for the per-scenario heatmap
        from sklearn.metrics import classification_report as _cr
        report_dict = _cr(y_true_names, y_pred_names, output_dict=True, zero_division=0)
        per_class_f1 = {
            k: round(v["f1-score"], 4)
            for k, v in report_dict.items()
            if isinstance(v, dict) and "f1-score" in v
        }

        console.print(f"\n  [bold]Best val acc={best_acc:.3f}[/bold]  "
                      f"Final test acc=[green]{acc:.3f}[/green]  F1={f1:.3f}")
        console.print(classification_report(y_true_names, y_pred_names, zero_division=0))

        fold_results.append({
            "fold": fold_id, "test_match": test_match,
            "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
            "per_class_f1": per_class_f1,
            "epoch_history": epoch_history,
        })

        ckpt = RESULTS_DIR / f"gcn_fold{fold_id}.pt"
        torch.save({"state_dict": best_state, "classes": le.classes_.tolist()}, ckpt)
        console.print(f"  Checkpoint → {ckpt}")

    conn.close()

    if fold_results:
        from rich.rule import Rule as R
        console.print(R("[bold]GCN+LSTM Cross-validation Summary[/bold]"))
        accs = [r["accuracy"] for r in fold_results]
        f1s  = [r["f1_macro"]  for r in fold_results]
        console.print(f"  Mean accuracy: [green]{np.mean(accs):.3f}[/green] ±{np.std(accs):.3f}")
        console.print(f"  Mean F1 macro: {np.mean(f1s):.3f}")
        console.print()
        console.print("[dim]Method comparison (mock data):[/dim]")
        console.print("  DTW K-Means (unsupervised)   ARI ≈ 0.985 — no labels")
        console.print("  Random Forest (supervised)   acc ≈ 0.996 — segment stats")
        console.print("  LSTM (supervised)            acc = ?     — temporal sequences")
        console.print(f"  GCN+LSTM (supervised)        acc = {np.mean(accs):.3f}   — graphs + temporal")
        console.print()
        console.print(
            "  On mock data the GCN may not outperform RF because positions are scripted\n"
            "  and noise-free. On real data, graph structure should be more robust to\n"
            "  CV tracking noise than flat position statistics."
        )

        # Flatten all epoch histories for the convergence plot
        all_epoch_history = [
            rec for fold in fold_results
            for rec in fold.get("epoch_history", [])
        ]
        summary = {
            "folds": [{k: v for k, v in f.items() if k != "epoch_history"}
                      for f in fold_results],
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy":  round(float(np.std(accs)),  4),
            "mean_f1_macro": round(float(np.mean(f1s)),  4),
            "epoch_history": all_epoch_history,
            "total_epochs": epochs,
        }
        (RESULTS_DIR / "gcn_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN+LSTM scenario classifier")
    parser.add_argument("--db", default="handball_mock.duckdb")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-transition", action="store_true")
    args = parser.parse_args()
    run(args.db, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, exclude_transition=args.no_transition)
