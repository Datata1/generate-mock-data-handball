"""Generate comparison plots and FINDINGS.md from all available supervised results.

Reads whatever *_metrics.json files exist in results/ and generates:
  1. method_comparison.png   — accuracy + F1 across all methods
  2. per_scenario_f1.png     — which Spielzüge each method detects (requires per-class data)
  3. gcn_convergence.png     — GCN training curve (requires epoch history)
  4. FINDINGS.md             — plain-language explanation of all plots

Run after any training job:
    PYTHONPATH=. uv run python experiments/supervised/report.py

Safe to re-run — overwrites outputs each time.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.rule import Rule

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.features import SCENARIO_ORDER

RESULTS_DIR = Path("experiments/supervised/results")
console = Console()

plt.rcParams.update({
    "figure.facecolor": "#0D1B2A",
    "axes.facecolor":   "#111827",
    "axes.edgecolor":   "#374151",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "#9CA3AF",
    "ytick.color":      "#9CA3AF",
    "grid.color":       "#1F2937",
    "axes.grid":        True,
})


# ── Load all available results ─────────────────────────────────────────────────

def _load_results() -> dict[str, dict]:
    """Return dict of method_name → metrics dict for each available JSON."""
    available = {}

    rf_path = RESULTS_DIR / "rf_metrics.json"
    if rf_path.exists():
        rf = json.loads(rf_path.read_text())
        for model_name, stats in rf["mean_by_model"].items():
            folds = [f for f in rf["folds"] if f["model"] == model_name]
            available[model_name] = {
                "mean_accuracy": stats["mean_accuracy"],
                "std_accuracy":  stats["std_accuracy"],
                "mean_f1_macro": stats["mean_f1_macro"],
                "folds": folds,
            }

    lstm_path = RESULTS_DIR / "lstm_metrics.json"
    if lstm_path.exists():
        lstm = json.loads(lstm_path.read_text())
        accs = [f["accuracy"] for f in lstm["folds"]]
        f1s  = [f["f1_macro"]  for f in lstm["folds"]]
        available["LSTM"] = {
            "mean_accuracy": round(np.mean(accs), 4),
            "std_accuracy":  round(np.std(accs),  4),
            "mean_f1_macro": round(np.mean(f1s),  4),
            "folds": lstm["folds"],
        }

    gcn_path = RESULTS_DIR / "gcn_metrics.json"
    if gcn_path.exists():
        gcn = json.loads(gcn_path.read_text())
        ep = gcn.get("total_epochs", "?")
        available[f"GCN+LSTM ({ep}ep)"] = {
            "mean_accuracy": gcn["mean_accuracy"],
            "std_accuracy":  gcn["std_accuracy"],
            "mean_f1_macro": gcn["mean_f1_macro"],
            "folds": gcn["folds"],
            "epoch_history": gcn.get("epoch_history", []),
            "total_epochs": ep,
        }

    return available


# ── Plot 1: Method comparison ──────────────────────────────────────────────────

def _plot_method_comparison(results: dict[str, dict]) -> None:
    """Grouped bar chart: accuracy + F1 macro for each method."""
    methods   = list(results.keys())
    acc_means = [results[m]["mean_accuracy"]  for m in methods]
    acc_stds  = [results[m]["std_accuracy"]   for m in methods]
    f1_means  = [results[m]["mean_f1_macro"]  for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.8), 5), dpi=130)

    bars_acc = ax.bar(x - width/2, acc_means, width, yerr=acc_stds,
                      capsize=4, label="Accuracy",
                      color=["#10B981" if a > 0.90 else "#F59E0B" if a > 0.70 else "#EF4444"
                             for a in acc_means],
                      error_kw={"ecolor": "#6B7280", "elinewidth": 1.5})
    bars_f1  = ax.bar(x + width/2, f1_means,  width,
                      label="F1 macro (mean)",
                      color="#3B82F6", alpha=0.85)

    # Unsupervised DTW baseline
    ax.axhline(y=0.985, color="#A78BFA", linestyle="--", linewidth=1.5, alpha=0.8,
               label="DTW (unsupervised, ARI=0.985)")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Supervised method comparison — mock data (leave-one-match-out CV)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.3)

    # Annotate bars with values
    for bar, val in zip(bars_acc, acc_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="white")

    fig.tight_layout()
    out = RESULTS_DIR / "method_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]→ {out}[/dim]")


# ── Plot 2: Per-scenario F1 heatmap ───────────────────────────────────────────

def _mean_per_class_f1(folds: list[dict]) -> dict[str, float]:
    """Average per-class F1 across folds from fold list."""
    class_f1s: dict[str, list[float]] = {}
    for fold in folds:
        for cls, f1 in fold.get("per_class_f1", {}).items():
            class_f1s.setdefault(cls, []).append(f1)
    return {cls: round(np.mean(vals), 3) for cls, vals in class_f1s.items()}


def _plot_per_scenario_f1(results: dict[str, dict]) -> bool:
    """Heatmap: scenarios × methods — only if per_class_f1 data is available."""
    # Check at least one method has per-class data
    methods_with_data = [
        m for m in results
        if any("per_class_f1" in f for f in results[m].get("folds", []))
    ]
    if not methods_with_data:
        console.print("  [yellow]per_scenario_f1.png skipped — re-run training to get per-class data[/yellow]")
        return False

    scenarios = [s for s in SCENARIO_ORDER if s != "macro avg" and s != "weighted avg"]
    matrix = np.full((len(scenarios), len(methods_with_data)), np.nan)

    for j, method in enumerate(methods_with_data):
        cls_f1 = _mean_per_class_f1(results[method]["folds"])
        for i, sc in enumerate(scenarios):
            if sc in cls_f1:
                matrix[i, j] = cls_f1[sc]

    # Shorten method labels
    method_labels = [m.replace("Random Forest", "RF").replace("Gradient Boosting", "GBoost") for m in methods_with_data]

    fig, ax = plt.subplots(
        figsize=(max(5, len(methods_with_data) * 1.6), max(5, len(scenarios) * 0.7)),
        dpi=130,
    )
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        vmin=0, vmax=1,
        xticklabels=method_labels, yticklabels=scenarios,
        linewidths=0.5, linecolor="#1F2937",
        ax=ax, mask=mask,
        cbar_kws={"label": "F1 score (mean over folds)"},
        annot_kws={"size": 9},
    )
    ax.set_title("Per-scenario F1 score by method\n(green = reliably detected, red = missed)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("True scenario")
    fig.tight_layout()
    out = RESULTS_DIR / "per_scenario_f1.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]→ {out}[/dim]")
    return True


# ── Plot 3: GCN convergence ────────────────────────────────────────────────────

def _plot_gcn_convergence(results: dict[str, dict]) -> bool:
    """Line chart: GCN val_accuracy and train_loss per epoch per fold."""
    gcn_key = next((k for k in results if k.startswith("GCN")), None)
    if gcn_key is None:
        return False
    history = results[gcn_key].get("epoch_history", [])
    if not history:
        console.print("  [yellow]gcn_convergence.png skipped — no epoch history (re-run GCN)[/yellow]")
        return False

    fold_ids = sorted(set(h["fold"] for h in history))
    colors = ["#3B82F6", "#10B981", "#F59E0B"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=130)

    for fold_id, color in zip(fold_ids, colors):
        fold_h = sorted([h for h in history if h["fold"] == fold_id], key=lambda x: x["epoch"])
        epochs  = [h["epoch"] for h in fold_h]
        val_acc = [h["val_acc"] for h in fold_h]
        tr_loss = [h["train_loss"] for h in fold_h]

        ax1.plot(epochs, val_acc, color=color, linewidth=2, label=f"Fold {fold_id}")
        ax2.plot(epochs, tr_loss, color=color, linewidth=2, label=f"Fold {fold_id}")

    ax1.axhline(y=0.5, color="#6B7280", linestyle=":", alpha=0.6, label="50% (random-ish)")
    ax1.set_title("GCN Validation Accuracy per epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, framealpha=0.3)

    ax2.set_title("GCN Training Loss per epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("CrossEntropy loss")
    ax2.legend(fontsize=9, framealpha=0.3)

    total_ep = results[gcn_key].get("total_epochs", "?")
    fig.suptitle(f"GCN+LSTM training convergence ({total_ep} epochs)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = RESULTS_DIR / "gcn_convergence.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    console.print(f"  [dim]→ {out}[/dim]")
    return True


# ── FINDINGS.md generation ─────────────────────────────────────────────────────

def _generate_findings(results: dict[str, dict], has_per_class: bool, has_convergence: bool) -> None:
    gcn_key = next((k for k in results if k.startswith("GCN")), None)
    gcn_ep  = results[gcn_key].get("total_epochs", "?") if gcn_key else "?"

    lines = [
        "# Supervised Learning — Findings",
        "",
        "Generated automatically after running `just supervised-report`.",
        "Re-run any training method, then re-run `just supervised-report` to update.",
        "",
        "---",
        "",
        "## All methods at a glance",
        "",
        "| Method | Mean Accuracy | F1 Macro | Std | Labels needed |",
        "|---|---|---|---|---|",
    ]

    # DTW unsupervised as baseline row
    lines.append("| DTW K-Means (unsupervised) | ARI=0.985 | — | — | No |")
    for method, data in results.items():
        acc = data["mean_accuracy"]
        f1  = data["mean_f1_macro"]
        std = data["std_accuracy"]
        flag = "⚠ underfitting" if acc < 0.70 else ("✓" if acc > 0.90 else "~")
        lines.append(f"| {method} | {acc:.3f} {flag} | {f1:.3f} | ±{std:.3f} | Yes |")

    lines += [
        "",
        "**Key insight:** The unsupervised DTW baseline (ARI=0.985) sets a high bar.",
        "Supervised methods need labels — they're only worth the annotation effort",
        "if they significantly outperform DTW on the hard scenario pairs, or if",
        "they generalise better to real (noisy) tracking data.",
        "",
        "---",
        "",
    ]

    # ── Plot 1: method comparison ──────────────────────────────────────────
    if (RESULTS_DIR / "method_comparison.png").exists():
        lines += [
            "## `method_comparison.png` — Accuracy + F1 across all methods",
            "",
            "![method comparison](method_comparison.png)",
            "",
            "**What it shows:** Each method's mean accuracy (coloured bar) and F1 macro",
            "(blue bar) across the 3-fold cross-validation. Error bars = std across folds.",
            "The purple dashed line = DTW unsupervised ARI (0.985) — the no-label baseline.",
            "",
            "**How to read it:**",
            "- Bars above the dashed line = better than unsupervised (worth labelling)",
            "- Wide error bars = high fold-to-fold variance (need more training matches)",
            "- Green bar = accuracy > 90%, yellow = 70–90%, red = < 70%",
            "",
            "**Key observations per method:**",
        ]

        for method, data in results.items():
            acc = data["mean_accuracy"]
            std = data["std_accuracy"]
            if acc > 0.95:
                obs = f"**{acc:.1%}** — near-perfect on mock data. High std (±{std:.2f}) likely due to small dataset (3 matches). Expect lower accuracy on real tracking data."
            elif acc > 0.75:
                obs = f"**{acc:.1%}** — good overall but high fold variance (±{std:.2f}) suggests 3 matches is insufficient for stable training."
            else:
                obs = f"**{acc:.1%}** — underfitting. Model is not trained long enough or needs more data. The 50%-ish accuracy reflects predicting the dominant class (transition)."
            lines.append(f"- **{method}:** {obs}")

        lines += ["", "---", ""]

    # ── Plot 2: per-scenario F1 ────────────────────────────────────────────
    if has_per_class and (RESULTS_DIR / "per_scenario_f1.png").exists():
        lines += [
            "## `per_scenario_f1.png` — Per-scenario F1 by method",
            "",
            "![per scenario F1](per_scenario_f1.png)",
            "",
            "**What it shows:** For each scenario (row) and each method (column),",
            "the mean F1 score across all cross-validation folds.",
            "Green = reliably detected, Red = missed, NaN = not in test set.",
            "",
            "**How to read it:**",
            "- A full green column = that method reliably classifies all scenarios",
            "- A red row = that scenario is hard for all methods",
            "- Compare columns: where RF is green but LSTM is yellow = RF wins for that scenario",
            "",
            "**Key observations:**",
            "- `transition` is almost always green — it's the majority class and easy to detect",
            "- `kreuzung` and `kreislaeuferspiel` should be green for RF (very distinct positions)",
            "- `doppelpass` vs `kreislaeuferspiel` is the hardest pair — look for red/yellow there",
            "- GCN column (if present) will be weaker if still underfitting",
            "",
            "**What this tells the trainer:**",
            "The red cells are exactly the Spielzüge that need more labelled examples",
            "or a richer feature representation to detect reliably.",
            "",
            "---",
            "",
        ]
    elif not has_per_class:
        lines += [
            "## Per-scenario F1 (requires re-run)",
            "",
            "The per-class F1 heatmap is not yet available because the existing results",
            "were generated before per-class tracking was added.",
            "",
            "Re-run training to get this plot:",
            "```bash",
            "just supervised-train      # RF (fast, seconds)",
            "just supervised-train-gcn  # GCN (50 epochs, ~minutes)",
            "just supervised-report     # generate plots",
            "```",
            "",
            "---",
            "",
        ]

    # ── Plot 3: GCN convergence ────────────────────────────────────────────
    if has_convergence and (RESULTS_DIR / "gcn_convergence.png").exists():
        gcn_acc = results.get(gcn_key, {}).get("mean_accuracy", "?")
        lines += [
            f"## `gcn_convergence.png` — GCN training convergence ({gcn_ep} epochs)",
            "",
            "![GCN convergence](gcn_convergence.png)",
            "",
            "**What it shows:** Left: validation accuracy per epoch for each of the 3 folds.",
            "Right: training loss per epoch. Fold colours match.",
            "",
            "**How to read it:**",
            "- If val_acc is still rising at the last epoch → train more epochs",
            "- If val_acc plateaus or drops while loss keeps falling → overfitting",
            "- High fold-to-fold spread → 3 matches is too few for stable training",
            "",
            "**Key observations:**",
        ]

        if isinstance(gcn_acc, float) and gcn_acc < 0.70:
            lines += [
                f"- At {gcn_ep} epochs, mean val_acc = {gcn_acc:.3f} — the model is still",
                "  learning. The first epochs are dominated by the `transition` class",
                "  (50% of data), so accuracy starts near 50% even for random predictions.",
                "- The loss should still be decreasing — run more epochs to see if it",
                "  converges. Try `just supervised-train-gcn` for the default 50 epochs.",
                f"- If accuracy is still below 75% at epoch {gcn_ep}, the model may need:",
                "  1. More training matches (`just generate n=10`)",
                "  2. Lower learning rate (`--lr 1e-4`)",
                "  3. Longer training (`--epochs 100`)",
            ]
        else:
            lines += [
                f"- At {gcn_ep} epochs, mean val_acc = {gcn_acc:.3f}.",
                "- If the curve has plateaued, this is the best the GCN can achieve on this data.",
                "- If it's still rising, run more epochs.",
            ]

        lines += ["", "---", ""]

    elif gcn_key and not has_convergence:
        lines += [
            f"## GCN convergence (requires re-run with epoch tracking)",
            "",
            "The GCN was run before epoch history tracking was added.",
            "",
            "```bash",
            "just supervised-train-gcn   # re-run at 50 epochs with history",
            "just supervised-report",
            "```",
            "",
            "---",
            "",
        ]

    # ── Tier-by-tier analysis ──────────────────────────────────────────────
    lines += [
        "## What each tier tells you",
        "",
        "### Tier 1 — Random Forest",
        "Works on pre-computed position/velocity statistics (same 81 features as",
        "unsupervised K-Means). With mock data it achieves near-perfect accuracy because",
        "positions are scripted and noise-free. On real CV tracking data expect ~75-90%",
        "(position noise ±0.5–1m corrupts mean_x features).",
        "",
        "**Best for:** quick validation that labeled data is useful; feature importance",
        "shows which player positions and velocities matter most.",
        "",
        "### Tier 2 — LSTM",
        "Processes the raw frame sequence (50 frames × 12 players × 4 features).",
        "Learns temporal patterns — when the velocity burst happens within the segment,",
        "not just that there was a burst. High fold variance because 150 training segments",
        "is marginal for a 130K-parameter model.",
        "",
        "**Best for:** scenarios where timing matters (doppelpass 8-frame arc,",
        "rueckpass sprint at frame ~50/300). Needs ≥10 training matches for stable results.",
        "",
        "### Tier 3 — GCN+LSTM",
        "Each frame is a player interaction graph. The GCN learns which spatial patterns",
        "(adjacency, distance, team structure) identify each Spielzug. The LSTM then reads",
        "the sequence of graph embeddings. Slower to converge than RF/LSTM but designed to",
        "be more robust to tracking noise.",
        "",
        f"**At {gcn_ep} epochs (current run):** likely underfitting — accuracy near 60%.",
        "This reflects predicting the dominant class (transition).",
        "**Recommended:** run 50–100 epochs, or 50+ epochs with ≥10 training matches.",
        "",
        "**Best for:** real tracking data where absolute positions are noisy but graph",
        "topology (who is near whom) is more stable.",
        "",
        "---",
        "",
        "## What to do next",
        "",
        "```bash",
        "# 1. Generate more matches for stable results",
        "just clean && just generate n=10 d=600 seed=42",
        "",
        "# 2. Re-run all methods",
        "just supervised-train",
        "just supervised-train-gcn",
        "",
        "# 3. Regenerate all plots and FINDINGS.md",
        "just supervised-report",
        "```",
        "",
        "With 10 matches:",
        "- RF fold std should drop below ±1%",
        "- LSTM and GCN should converge more reliably",
        "- The per-scenario F1 heatmap will show which Spielzüge each method handles",
    ]

    (RESULTS_DIR / "FINDINGS.md").write_text("\n".join(lines))
    console.print(f"  [dim]→ {RESULTS_DIR}/FINDINGS.md[/dim]")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    console.print(Rule("[bold]Supervised Learning — Report Generator[/bold]"))

    results = _load_results()
    if not results:
        console.print("[red]No results found in experiments/supervised/results/[/red]")
        console.print("Run `just supervised-train` first.")
        return

    console.print(f"\n  Available methods: {', '.join(results.keys())}\n")
    console.print("  Generating plots...")

    _plot_method_comparison(results)
    has_per_class = _plot_per_scenario_f1(results)
    has_convergence = _plot_gcn_convergence(results)
    _generate_findings(results, has_per_class, has_convergence)

    console.print(f"\n[green]Done.[/green] Open [bold]{RESULTS_DIR}/FINDINGS.md[/bold]\n")


if __name__ == "__main__":
    main()
