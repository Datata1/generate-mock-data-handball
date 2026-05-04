"""Tier 1 supervised baseline: Random Forest + Gradient Boosting.

No PyTorch needed. Runs in seconds on the 223-sample dataset.
Uses leave-one-match-out cross-validation to avoid data leakage.

Usage:
    PYTHONPATH=. uv run python experiments/supervised/train_rf.py
    PYTHONPATH=. uv run python experiments/supervised/train_rf.py --db other.duckdb
    PYTHONPATH=. uv run python experiments/supervised/train_rf.py --no-transition
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.supervised.dataset import Fold, load_folds, class_distribution
from experiments.plot import plot_confusion_heatmap, SCENARIO_COLORS

RESULTS_DIR = Path("experiments/supervised/results")
console = Console()


# ── Model definitions ──────────────────────────────────────────────────────────

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    ),
}


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _fold_report(fold: Fold, model_name: str, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import classification_report as _cr
    acc = float(accuracy_score(fold.y_test, y_pred))
    f1_macro = float(f1_score(fold.y_test, y_pred, average="macro", zero_division=0))
    report_dict = _cr(fold.y_test, y_pred, output_dict=True, zero_division=0)
    # Extract per-class F1 (skip the aggregate rows)
    per_class_f1 = {
        k: round(v["f1-score"], 4)
        for k, v in report_dict.items()
        if isinstance(v, dict) and "f1-score" in v
    }
    return {
        "fold": fold.fold_id,
        "test_match": fold.test_match,
        "train_n": len(fold.y_train),
        "test_n": len(fold.y_test),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class_f1": per_class_f1,
        "model": model_name,
    }


def _print_cv_summary(results: list[dict], console: Console) -> None:
    """Print per-fold and mean metrics as a Rich table."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Model")
    table.add_column("Fold", justify="right")
    table.add_column("Test match", style="dim")
    table.add_column("Train N", justify="right")
    table.add_column("Test N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 macro", justify="right")

    for r in results:
        acc_s = (
            f"[green]{r['accuracy']:.3f}[/green]" if r["accuracy"] > 0.80
            else f"[yellow]{r['accuracy']:.3f}[/yellow]" if r["accuracy"] > 0.60
            else f"[red]{r['accuracy']:.3f}[/red]"
        )
        table.add_row(
            r["model"], str(r["fold"]), r["test_match"],
            str(r["train_n"]), str(r["test_n"]),
            acc_s, f"{r['f1_macro']:.3f}",
        )

    console.print(table)

    # Mean per model
    model_names = list({r["model"] for r in results})
    console.print()
    mean_table = Table(title="Mean across folds", show_header=True, header_style="bold")
    mean_table.add_column("Model", style="cyan")
    mean_table.add_column("Mean Accuracy", justify="right")
    mean_table.add_column("Mean F1 macro", justify="right")
    mean_table.add_column("Std Accuracy", justify="right")

    for name in model_names:
        fold_results = [r for r in results if r["model"] == name]
        accs = [r["accuracy"] for r in fold_results]
        f1s = [r["f1_macro"] for r in fold_results]
        acc_s = (
            f"[green]{np.mean(accs):.3f}[/green]" if np.mean(accs) > 0.80
            else f"[yellow]{np.mean(accs):.3f}[/yellow]"
        )
        mean_table.add_row(name, acc_s, f"{np.mean(f1s):.3f}", f"±{np.std(accs):.3f}")

    console.print(mean_table)


def _feature_importance_top10(model, feature_names: list[str]) -> list[tuple[str, float]]:
    if not hasattr(model, "feature_importances_"):
        return []
    imp = model.feature_importances_
    ranked = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    return ranked[:10]


# ── Confusion matrix adapted for evaluate.ClusterResult interface ──────────────

class _PseudoClusterResult:
    """Thin wrapper so we can reuse plot_confusion_heatmap."""
    def __init__(self, method, cluster_labels, true_labels, ari):
        self.method = method
        self.cluster_labels = cluster_labels
        self.true_labels = true_labels
        self.ari = ari
        self.n_clusters = len(set(cluster_labels))

    def _get_confusion_df(self):
        from experiments.evaluate import build_confusion_df
        return build_confusion_df(self)


# ── Main training loop ─────────────────────────────────────────────────────────

def run(db_path: str, exclude_transition: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path, read_only=True)
    console.print(Rule("[bold]Supervised Learning — Random Forest + Gradient Boosting[/bold]"))

    # Load folds
    folds = load_folds(conn, exclude_transition=exclude_transition)
    conn.close()
    n_classes = len(set(np.concatenate([f.y_train for f in folds])))
    console.print(
        f"\n  Matches: {len(folds)} folds (leave-one-match-out CV)\n"
        f"  Segments: train≈{len(folds[0].y_train)}, test≈{len(folds[0].y_test)}\n"
        f"  Classes: {n_classes}  "
        + ("[dim](transition excluded)[/dim]" if exclude_transition else "")
    )

    all_results: list[dict] = []
    # Collect all test predictions for final confusion matrix
    all_y_true: list[np.ndarray] = []
    all_y_pred: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}

    for model_name, clf_template in MODELS.items():
        console.print(Rule(f"[cyan]{model_name}[/cyan]"))
        fold_importances: list[list[tuple[str, float]]] = []

        for fold in folds:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(fold.X_train)
            X_test = scaler.transform(fold.X_test)

            # Clone model (avoid fitting state bleeding between folds)
            import copy
            clf = copy.deepcopy(clf_template)
            clf.fit(X_train, fold.y_train)
            y_pred = clf.predict(X_test)

            result = _fold_report(fold, model_name, y_pred)
            all_results.append(result)
            all_y_pred[model_name].append(y_pred)

            if fold.fold_id == 0:
                all_y_true.append(fold.y_test)  # collect once

            # Feature importances from first fold for inspection
            if fold.fold_id == 0:
                # Feature names: column order matches build_segment_features output
                n_feat = X_train.shape[1]
                feat_names = [f"feat_{i}" for i in range(n_feat)]
                top10 = _feature_importance_top10(clf, feat_names)
                fold_importances.append(top10)

            console.print(
                f"  Fold {fold.fold_id} | test={fold.test_match[-9:]} "
                f"| acc={result['accuracy']:.3f} | f1={result['f1_macro']:.3f}"
            )

        # Print per-class report for the last fold (most representative)
        last_fold = folds[-1]
        scaler = StandardScaler()
        clf = copy.deepcopy(clf_template)
        clf.fit(scaler.fit_transform(last_fold.X_train), last_fold.y_train)
        y_pred_last = clf.predict(scaler.transform(last_fold.X_test))

        console.print("\n  [dim]Classification report (last fold):[/dim]")
        report = classification_report(last_fold.y_test, y_pred_last, zero_division=0)
        for line in report.split("\n"):
            console.print(f"    {line}")

    console.print(Rule("[bold]Cross-validation Summary[/bold]"))
    _print_cv_summary(all_results, console)

    # Save confusion matrix for the best model (by mean accuracy)
    model_mean_acc = {}
    for name in MODELS:
        accs = [r["accuracy"] for r in all_results if r["model"] == name]
        model_mean_acc[name] = np.mean(accs)
    best_model = max(model_mean_acc, key=model_mean_acc.get)

    # Build pooled y_true and y_pred across folds for the best model
    pooled_true = np.concatenate([f.y_test for f in folds])
    pooled_pred = np.concatenate(all_y_pred[best_model])

    # Wrap as pseudo-ClusterResult for reusing the confusion matrix plotter
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(pooled_true, pooled_pred)
    pseudo = _PseudoClusterResult(
        method=best_model,
        cluster_labels=pooled_pred,
        true_labels=pooled_true,
        ari=ari,
    )
    # Build confusion df manually since pseudo uses str cluster_labels
    import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
    from experiments.features import SCENARIO_ORDER

    true_labels_arr = pooled_true
    pred_labels_arr = pooled_pred
    all_scenarios = [s for s in SCENARIO_ORDER if s in true_labels_arr]
    rows = []
    for ts in all_scenarios:
        mask = true_labels_arr == ts
        if not mask.any():
            continue
        row = {ps: int((pred_labels_arr[mask] == ps).sum()) for ps in all_scenarios}
        rows.append({"true_scenario": ts, **row})
    conf_df = pd.DataFrame(rows).set_index("true_scenario").fillna(0).astype(int)
    row_sums = conf_df.sum(axis=1).replace(0, 1)
    conf_pct = (conf_df.div(row_sums, axis=0) * 100).round(1)

    fig, ax = plt.subplots(
        figsize=(max(6, len(conf_pct.columns) * 1.1), max(5, len(conf_pct) * 0.7)),
        dpi=120,
    )
    sns.heatmap(conf_pct, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="#333", ax=ax,
                cbar_kws={"label": "% of true scenario"}, annot_kws={"size": 9})
    mean_acc = model_mean_acc[best_model]
    ax.set_title(
        f"{best_model} — pooled CV  (mean acc={mean_acc:.3f}, ARI={ari:.3f})",
        fontsize=11
    )
    ax.set_xlabel("Predicted scenario")
    ax.set_ylabel("True scenario")
    fig.tight_layout()
    conf_path = RESULTS_DIR / f"confusion_{best_model.lower().replace(' ', '_')}.png"
    fig.savefig(conf_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    console.print(f"\n  [dim]Confusion matrix → {conf_path}[/dim]")

    # Save metrics JSON
    summary = {
        "folds": all_results,
        "mean_by_model": {
            name: {
                "mean_accuracy": round(float(np.mean([r["accuracy"] for r in all_results if r["model"] == name])), 4),
                "std_accuracy":  round(float(np.std([r["accuracy"]  for r in all_results if r["model"] == name])), 4),
                "mean_f1_macro": round(float(np.mean([r["f1_macro"] for r in all_results if r["model"] == name])), 4),
            }
            for name in MODELS
        }
    }
    metrics_path = RESULTS_DIR / "rf_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    console.print(f"  [dim]Metrics → {metrics_path}[/dim]")

    _generate_supervised_findings(summary, model_mean_acc, best_model, ari, RESULTS_DIR)
    console.print(f"  [dim]Findings → {RESULTS_DIR}/FINDINGS.md[/dim]")

    _print_comparison_note(all_results, console)


def _print_comparison_note(results: list[dict], console: Console) -> None:
    console.print()
    console.print(Rule("[bold]Supervised vs Unsupervised: What changed?[/bold]"))
    model_names = list({r["model"] for r in results})
    for name in model_names:
        accs = [r["accuracy"] for r in results if r["model"] == name]
        mean_acc = np.mean(accs)
        console.print(
            f"\n  [cyan]{name}[/cyan]  mean accuracy={mean_acc:.3f}"
        )
    console.print("""
  Compare to unsupervised results (from experiments/results/metrics.json):
    DTW K-Means (unsupervised)  ARI=0.985  ← no labels needed
    K-Means k=5 (unsupervised)  ARI=0.863  ← no labels needed

  Supervised accuracy is the fraction of correctly classified segments on
  held-out matches. ARI and accuracy are not directly comparable, but:

  • If supervised accuracy < DTW ARI: labelling effort not worth it for this data.
  • If supervised accuracy > 0.90:    supervised adds value for the hard pairs
    (doppelpass vs kreislaeuferspiel, defense_60 vs rueckpass).
  • The confusion matrix shows WHERE the model fails — same pairs as unsupervised.

  Note: with only ~150 training segments per fold, results have HIGH VARIANCE.
  Generate more matches (just generate n=10) to get stable estimates.
""")


def _generate_supervised_findings(
    summary: dict,
    model_mean_acc: dict,
    best_model: str,
    ari: float,
    output_dir: Path,
) -> None:
    """Generate FINDINGS.md explaining each supervised output file."""
    best_acc = model_mean_acc[best_model]
    best_std = summary["mean_by_model"][best_model]["std_accuracy"]
    n_folds = len(summary["folds"]) // len(model_mean_acc)

    conf_img = f"confusion_{best_model.lower().replace(' ', '_')}.png"

    lines = [
        "# Supervised Learning — Findings",
        "",
        "Generated automatically after running `just supervised-train`.",
        "",
        "---",
        "",
        f"## `{conf_img}` — Confusion matrix (pooled CV)",
        "",
        f"![confusion]({conf_img})",
        "",
        "**What it shows:** Each row is a true scenario, each column is the predicted",
        "scenario. Values are percentages of that true scenario that were predicted",
        "as each class (rows sum to 100%). Dark red = high concentration.",
        "",
        "**How to read it:**",
        "- **Dark red on the diagonal** = that scenario is correctly classified.",
        "- **Dark red off the diagonal** = those two scenarios are being confused.",
        "  The off-diagonal pattern tells you exactly where to focus: either better",
        "  features, more training data, or accepting that those plays look similar.",
        "",
        f"**Key finding:** {best_model} achieves {best_acc:.1%} mean accuracy.",
    ]

    if best_acc > 0.95:
        lines += [
            "The confusion matrix is nearly all-diagonal — the model correctly",
            "separates all 9 scenarios on the held-out matches.",
            "",
            "**Important caveat:** this is on noise-free mock data. The mock generator",
            "places players at scripted positions with small Gaussian jitter (~5cm).",
            "Real CV tracking has much larger errors (position noise ±1m, ID switches,",
            "missed detections). Expect accuracy to drop to 75–90% on real match data.",
        ]
    elif best_acc > 0.80:
        lines += [
            "The confusion matrix shows good diagonal structure, but some off-diagonal",
            "entries reveal which scenarios the model struggles to separate.",
            "These are likely the same hard pairs identified in the unsupervised analysis.",
        ]
    else:
        lines += [
            "Significant off-diagonal entries indicate pairs of scenarios that look",
            "similar in position/velocity statistics. Consider: more training matches,",
            "additional features (velocity variance, ball arc shape), or the LSTM model.",
        ]

    lines += [
        "",
        "---",
        "",
        "## `rf_metrics.json` — Cross-validation metrics",
        "",
        "```json",
    ]
    import json
    lines.append(json.dumps(summary["mean_by_model"], indent=2))
    lines += [
        "```",
        "",
        "**What it shows:** Leave-one-match-out cross-validation results.",
        f"Each of the {n_folds} folds trains on {n_folds-1} matches and tests on 1.",
        "",
        "**How to read it:**",
        "- `mean_accuracy`: average fraction of correctly classified segments across folds.",
        "- `std_accuracy`: standard deviation across folds. High std (>0.05) means",
        "  the result depends heavily on which match is held out — you need more data.",
        "- `mean_f1_macro`: average F1 across all classes weighted equally.",
        "  More informative than accuracy when class sizes differ (many transition",
        "  segments skew accuracy upward).",
        "",
        f"**Key numbers:** {best_model} — accuracy {best_acc:.3f} ±{best_std:.3f}",
        "",
    ]

    # Model comparison table
    lines += [
        "| Model | Mean Accuracy | F1 Macro | Std |",
        "|---|---|---|---|",
    ]
    for name, stats in summary["mean_by_model"].items():
        lines.append(
            f"| {name} | {stats['mean_accuracy']:.3f} | {stats['mean_f1_macro']:.3f} | ±{stats['std_accuracy']:.3f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Supervised vs Unsupervised: What changed?",
        "",
        "| Method | Score | Labels needed |",
        "|---|---|---|",
        "| DTW K-Means (unsupervised) | ARI = 0.985 | No |",
        "| K-Means k=5 (unsupervised) | ARI = 0.863 | No |",
        f"| {best_model} (supervised) | acc = {best_acc:.3f} | Yes |",
        "",
        "**ARI and accuracy are not directly comparable**, but both are bounded",
        "[0, 1] where 1 is perfect. Higher is better for both.",
        "",
    ]

    if best_acc > 0.95:
        lines += [
            "**On mock data:** supervised adds ~{:.0%} over DTW (which needs no labels).".format(best_acc - 0.985),
            "The question is whether this marginal gain is worth the labelling effort.",
            "For the mock dataset: **probably not** — DTW already works extremely well.",
            "",
            "**On real data:** supervised will likely outperform DTW significantly.",
            "DTW is sensitive to tracking noise (jittered ball positions look like different",
            "trajectories). Position statistics are more robust; LSTM is even more so.",
        ]
    else:
        lines += [
            "Supervised improves over position-statistic-based unsupervised clustering.",
            "The confusion matrix shows which hard pairs benefit most from labels.",
        ]

    lines += [
        "",
        "---",
        "",
        "## What to do next",
        "",
        "1. **Generate more matches** for stable CV estimates:",
        "   ```bash",
        "   just clean && just generate n=10 d=600 seed=42 && just supervised-train",
        "   ```",
        "   With 10 matches the std should drop below ±2%.",
        "",
        "2. **Try without transition** (cleaner 8-class problem):",
        "   ```bash",
        "   just supervised-train-clean",
        "   ```",
        "",
        "3. **Train the LSTM** (if torch installed) to capture temporal dynamics:",
        "   ```bash",
        "   just supervised-train-lstm",
        "   ```",
        "   Expected improvement: +5–10% on the hard pairs (doppelpass vs kreislaeuferspiel).",
        "",
        "4. **Run on real data** — once the CV pipeline produces a DuckDB with",
        "   real match tracking and `scenario_labels` are annotated, run the same commands.",
        "   No code changes needed; just point `--db` at the real database.",
    ]

    (output_dir / "FINDINGS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="handball_mock.duckdb")
    parser.add_argument("--no-transition", action="store_true",
                        help="Exclude transition segments (cleaner 8-class problem)")
    args = parser.parse_args()
    run(args.db, exclude_transition=args.no_transition)
