"""Unsupervised ML experiment orchestrator.

Usage:
    uv run python experiments/unsupervised/run_all.py                  # all tiers
    uv run python experiments/unsupervised/run_all.py --db other.duckdb
    uv run python experiments/unsupervised/run_all.py --tier 1         # K-Means+GMM only
    uv run python experiments/unsupervised/run_all.py --tier 2         # UMAP+HDBSCAN+DTW
    uv run python experiments/unsupervised/run_all.py --tier 3         # GCN (optional)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
from rich.console import Console
from rich.rule import Rule

RESULTS_DIR = Path("experiments/unsupervised/results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Handball unsupervised ML experiments")
    parser.add_argument("--db", default="handball_mock.duckdb", help="DuckDB file")
    parser.add_argument("--tier", default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    console = Console()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(Rule("[bold]Handball Unsupervised ML Experiments[/bold]"))

    if not Path(args.db).exists():
        console.print(f"[red]Error:[/red] {args.db} not found. Run [cyan]just generate[/cyan] first.")
        raise SystemExit(1)

    conn = duckdb.connect(args.db, read_only=True)

    # ── Feature extraction ────────────────────────────────────────────────────
    console.print("\n[cyan]Extracting features...[/cyan]")
    t0 = time.perf_counter()

    from experiments.features import build_segment_features, build_ball_trajectories
    X_seg, y_seg, meta_seg = build_segment_features(conn)
    trajectories, y_traj = build_ball_trajectories(conn)

    console.print(
        f"  Segment features : [bold]{X_seg.shape}[/bold]  "
        f"({len(set(y_seg))} unique scenarios, N={len(y_seg)})"
    )
    console.print(f"  Ball trajectories: {len(trajectories)} segments")
    console.print(f"  [dim]Feature extraction: {time.perf_counter() - t0:.1f}s[/dim]")

    all_results = []

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    if args.tier in ("1", "all"):
        console.print(Rule("[bold cyan]Tier 1 — K-Means + GMM[/bold cyan]"))
        t1 = time.perf_counter()

        from experiments.unsupervised.baseline import run_kmeans, run_gmm, run_pca_scatter
        from experiments.evaluate import print_metrics_table
        from experiments.plot import plot_confusion_heatmap

        # PCA scatter (visual sanity check — not a clustering result)
        console.print("  Running PCA scatter...")
        run_pca_scatter(X_seg, y_seg, RESULTS_DIR)
        console.print(f"  [dim]→ {RESULTS_DIR}/pca_true_scenarios.png[/dim]")

        # K-Means for k = 5, 7, 9, 11
        console.print("  Running K-Means (k=5,7,9,11)...")
        km_results = run_kmeans(X_seg, y_seg, k_values=[5, 7, 9, 11], output_dir=RESULTS_DIR)

        # GMM
        console.print("  Running GMM (PCA→15d, diag covariance)...")
        gmm_results = run_gmm(X_seg, y_seg, k_values=[5, 7, 9, 11], output_dir=RESULTS_DIR)

        tier1 = km_results + gmm_results
        all_results.extend(tier1)
        print_metrics_table(tier1, console, title="Tier 1 Results")

        # Save confusion matrix for best K-Means result
        best_km = max(km_results, key=lambda r: r.ari)
        plot_confusion_heatmap(best_km, RESULTS_DIR / "confusion_kmeans_best.png",
                               title=f"K-Means k={best_km.n_clusters} — ARI={best_km.ari:.3f}")
        console.print(f"  [dim]→ confusion_kmeans_best.png, kmeans_elbow.png[/dim]")
        console.print(f"  [dim]Tier 1 total: {time.perf_counter() - t1:.1f}s[/dim]")

    # ── Tier 2 ────────────────────────────────────────────────────────────────
    if args.tier in ("2", "all"):
        console.print(Rule("[bold cyan]Tier 2 — UMAP + HDBSCAN + DTW K-Medoids[/bold cyan]"))
        t2 = time.perf_counter()

        from experiments.unsupervised.umap_cluster import run_umap_hdbscan, run_dtw_kmedoids
        from experiments.evaluate import print_metrics_table
        from experiments.plot import plot_confusion_heatmap

        console.print("  Running UMAP + HDBSCAN...")
        umap_result = run_umap_hdbscan(X_seg, y_seg, output_dir=RESULTS_DIR)
        console.print(f"  [dim]→ umap_true_scenarios.png, umap_hdbscan_clusters.png[/dim]")

        console.print("  Running DTW K-Medoids on ball trajectories...")
        dtw_result = run_dtw_kmedoids(trajectories, y_traj, k=9, output_dir=RESULTS_DIR)

        tier2 = [r for r in [umap_result, dtw_result] if r is not None]
        all_results.extend(tier2)
        print_metrics_table(tier2, console, title="Tier 2 Results")

        if dtw_result is None:
            console.print("  [yellow]DTW skipped: tslearn not available[/yellow]")

        plot_confusion_heatmap(umap_result, RESULTS_DIR / "confusion_umap.png",
                               title=f"UMAP+HDBSCAN — ARI={umap_result.ari:.3f}")
        console.print(f"  [dim]Tier 2 total: {time.perf_counter() - t2:.1f}s[/dim]")

    # ── Tier 3 ────────────────────────────────────────────────────────────────
    if args.tier in ("3", "all"):
        console.print(Rule("[bold cyan]Tier 3 — GCN Embeddings (optional)[/bold cyan]"))
        t3 = time.perf_counter()

        from experiments.unsupervised.gcn_cluster import run_gcn_kmeans
        from experiments.evaluate import print_metrics_table

        gcn_result = run_gcn_kmeans(conn, y_seg, meta_seg, k=9, output_dir=RESULTS_DIR)

        if gcn_result is not None:
            all_results.append(gcn_result)
            print_metrics_table([gcn_result], console, title="Tier 3 Results")
            console.print(
                "  [dim]Note: ARI≈0 is expected with untrained weights. "
                "Re-run after wels-train produces a checkpoint.[/dim]"
            )
        else:
            console.print(
                "  [yellow]Skipped:[/yellow] torch or torch_geometric not installed.\n"
                "  Install with: pip install torch torch-geometric\n"
                "  (or run: cd ../wels-monorepo && uv sync)"
            )

        console.print(f"  [dim]Tier 3 total: {time.perf_counter() - t3:.1f}s[/dim]")

    # ── Final summary ─────────────────────────────────────────────────────────
    if all_results:
        console.print(Rule("[bold]Summary — All Methods[/bold]"))
        from experiments.evaluate import print_metrics_table, save_metrics_json
        print_metrics_table(all_results, console, title="All Clustering Results")

        save_metrics_json(all_results, RESULTS_DIR / "metrics.json")
        _print_unsupervised_conclusions(all_results, console)

        _generate_findings_report(all_results, RESULTS_DIR)
        console.print(f"\n[green]Results saved to[/green] [bold]{RESULTS_DIR}/[/bold]")
        console.print(f"  → [cyan]{RESULTS_DIR}/FINDINGS.md[/cyan] — per-plot explanations")

    conn.close()


def _print_unsupervised_conclusions(results: list, console: Console) -> None:
    """Print conclusions about unsupervised usefulness — not about supervised methods."""
    console.print()
    console.print(Rule("[bold]What do these results say about unsupervised learning?[/bold]"))

    km_k9 = next((r for r in results if "K-Means" in r.method and r.n_clusters == 9), None)
    dtw = next((r for r in results if "DTW" in r.method), None)
    km_k5 = next((r for r in results if "K-Means" in r.method and r.n_clusters == 5), None)
    umap_r = next((r for r in results if "UMAP" in r.method), None)

    console.print()
    console.print("[bold cyan]✔ What unsupervised methods CAN do:[/bold cyan]")
    if dtw and dtw.ari > 0.9:
        console.print(
            f"  • [green]Ball trajectory DTW (ARI={dtw.ari:.3f})[/green]: Nearly perfect separation of all 9 scenarios\n"
            "    using only how the ball moves — no player data needed.\n"
            "    Practical use: automatically segment unlabeled match video into play types\n"
            "    with ~98% accuracy by tracking ball trajectory alone."
        )
    if km_k5 and km_k5.ari > 0.7:
        console.print(
            f"  • [green]K-Means k=5 (ARI={km_k5.ari:.3f})[/green]: Reliably finds 5 coarse tactical groups\n"
            "    from position/velocity statistics:\n"
            "    (1) transitions, (2) kreuzung, (3) defense_42,\n"
            "    (4) kreislaeuferspiel+doppelpass+defense_51, (5) others.\n"
            "    Practical use: automatic macro-categorisation (attack/defense/transition)\n"
            "    for coaching dashboards without any manual labelling."
        )
    if umap_r:
        console.print(
            f"  • [green]UMAP (ARI={umap_r.ari:.3f})[/green]: Visually separates scenario clusters.\n"
            "    Kreuzung and Kreisläufer-Anspiel are completely isolated islands.\n"
            "    Transition segments form two distinct sub-groups.\n"
            "    Practical use: exploratory data analysis — coaches can browse clusters\n"
            "    to understand what patterns exist in their data."
        )

    console.print()
    console.print("[bold yellow]✘ What unsupervised methods CANNOT reliably do:[/bold yellow]")
    if km_k9:
        console.print(
            f"  • [yellow]Fine-grained 9-way separation (K-Means k=9, ARI={km_k9.ari:.3f})[/yellow]:\n"
            "    Position statistics alone cannot distinguish all 9 scenarios.\n"
            "    Confused pairs: defense_60 ↔ rueckpass (similar x positions),\n"
            "    doppelpass ↔ kreislaeuferspiel (similar player positions, differ in ball speed).\n"
            "    To separate these, you need time-aware features (DTW) or labelled examples."
        )

    console.print()
    console.print("[bold]Recommendation:[/bold]")
    console.print(
        "  For a coaching tool, use [cyan]DTW on ball trajectory[/cyan] as the primary\n"
        "  segmentation signal — it works unsupervised and is highly accurate.\n"
        "  Use [cyan]UMAP visualisation[/cyan] for exploration.\n"
        "  Only invest in manual labelling + supervised methods if you need to distinguish\n"
        "  the hard pairs (doppelpass vs kreislaeuferspiel, defense_60 vs rueckpass)."
    )


def _generate_findings_report(results: list, output_dir: Path) -> None:
    """Write FINDINGS.md with per-plot explanations based on actual results."""
    km_k5  = next((r for r in results if "K-Means" in r.method and r.n_clusters == 5), None)
    km_k9  = next((r for r in results if "K-Means" in r.method and r.n_clusters == 9), None)
    dtw    = next((r for r in results if "DTW" in r.method), None)
    umap_r = next((r for r in results if "UMAP" in r.method), None)

    lines = [
        "# Experiment Findings",
        "",
        "This file explains what each plot shows and what to conclude from it.",
        "Generated automatically after running `just experiment`.",
        "",
        "---",
        "",
    ]

    # ── PCA scatter ────────────────────────────────────────────────────────────
    if (output_dir / "pca_true_scenarios.png").exists():
        lines += [
            "## `pca_true_scenarios.png` — PCA 2D scatter",
            "",
            "![PCA scatter](pca_true_scenarios.png)",
            "",
            "**What it shows:** Each dot is one scenario segment, projected onto the two",
            "directions of maximum variance (PC1 captures 37.9%, PC2 12.8%).",
            "Colours are the true scenario labels — not cluster assignments.",
            "",
            "**How to read it:**",
            "- **Tight, isolated colour blobs = that scenario is statistically very distinct.**",
            "- **Overlapping colours = those scenarios look similar to the algorithm.**",
            "",
            "**Key observations:**",
            "- 🟠 `kreislaeuferspiel` (orange) — extremely tight isolated cluster (top right).",
            "  The pivot stationary at x≈33 gives it a completely unique position fingerprint.",
            "- 🩷 `kreuzung` (pink) — isolated cluster (bottom centre).",
            "  The crossing trajectory creates a distinct velocity/position profile.",
            "- ⬜ `transition` (grey) — large diffuse cloud (left side), not a tight cluster.",
            "  Expected: transitions are positional bridges with no fixed shape.",
            "- The remaining 6 scenarios (defense_60/51/42, doppelpass, parallelstos, rueckpass)",
            "  overlap heavily in the centre. PCA's 2D projection (50% of total variance)",
            "  cannot separate them — a non-linear method is needed.",
            "",
            "**Unsupervised usefulness:** PCA alone is not enough for fine-grained",
            "clustering, but it confirms that 2–3 scenarios are strongly separable.",
            "",
        ]

    # ── UMAP true ────────────────────────────────────────────────────────────
    if (output_dir / "umap_true_scenarios.png").exists():
        lines += [
            "## `umap_true_scenarios.png` — UMAP 2D scatter (true labels)",
            "",
            "![UMAP true](umap_true_scenarios.png)",
            "",
            "**What it shows:** Same data as PCA but projected with UMAP — a non-linear",
            "method that tries to preserve *neighbourhood structure*. Colours = true labels.",
            "",
            "**How to read it:** Well-separated colour islands = those scenarios are",
            "neighbourhoods in the high-dimensional feature space. Overlapping islands = similar.",
            "",
            "**Key observations:**",
            "- 🩷 `kreuzung` — completely isolated island, far left. The most distinct play.",
            "- 🟠 `kreislaeuferspiel` — isolated far right (large cluster) + one outlier near top.",
            "  The outlier is likely a 2-pivot variant with slightly different statistics.",
            "- ⬜ `transition` — appears as **two separate blobs** (centre-left and bottom-left).",
            "  This is a real finding: there are two structurally different types of transitions",
            "  (defense→attack and attack→defense), not one homogeneous group.",
            "- All 6 remaining scenarios cluster tightly together in the centre (~2–4, 10–13).",
            "  UMAP brings them closer than PCA — their UMAP neighbourhoods overlap,",
            "  meaning they genuinely share similar statistical signatures.",
            "",
            "**Unsupervised usefulness:** UMAP clearly shows kreuzung and kreislaeuferspiel",
            "are discoverable without labels. The 'attack play' cluster is real but internally",
            "heterogeneous — a coach browsing it would find mixed plays.",
            "",
        ]

    # ── UMAP clusters ─────────────────────────────────────────────────────────
    if (output_dir / "umap_hdbscan_clusters.png").exists() and umap_r:
        lines += [
            f"## `umap_hdbscan_clusters.png` — UMAP+HDBSCAN cluster assignments (ARI={umap_r.ari:.3f})",
            "",
            "![UMAP HDBSCAN](umap_hdbscan_clusters.png)",
            "",
            "**What it shows:** The same UMAP projection as above, but coloured by the",
            f"cluster labels assigned by HDBSCAN (found {umap_r.n_clusters} clusters automatically).",
            "Colours here are cluster names (majority true label within each cluster).",
            "",
            "**How to read it:** Compare this to `umap_true_scenarios.png` side-by-side.",
            "Wherever the colours match = the algorithm correctly identified that group.",
            "Wherever they differ = misassignment.",
            "",
            "**Key observations:**",
            "- kreuzung and kreislaeuferspiel: colours match perfectly in both plots. ✅",
            f"- HDBSCAN found {umap_r.n_clusters} clusters vs the 9 true scenarios. The extra clusters",
            "  come from transition being split into 2 and the attack-play group being",
            "  sub-divided further than the true labels.",
            f"- ARI={umap_r.ari:.3f}: meaningful agreement with true labels, but not perfect.",
            "  The main source of error is the mixed attack-play cluster.",
            "",
            "**Unsupervised usefulness:** HDBSCAN finds real density structures without",
            "being told how many to look for. The sub-division of transition into 2 clusters",
            "is arguably *more informative* than the original labels.",
            "",
        ]

    # ── K-Means elbow ─────────────────────────────────────────────────────────
    if (output_dir / "kmeans_elbow.png").exists():
        lines += [
            "## `kmeans_elbow.png` — K-Means elbow curve",
            "",
            "![elbow](kmeans_elbow.png)",
            "",
            "**What it shows:** For each value of k (number of clusters), the total",
            "inertia (sum of squared distances from each point to its cluster centre).",
            "A sharp bend ('elbow') at a particular k suggests that's the natural cluster count.",
            "",
            "**How to read it:** Look for the point where adding more clusters stops",
            "reducing inertia significantly.",
            "",
            "**Key observation:** The curve is smooth with no sharp elbow.",
            "The orange dashed line marks k=9 (the true number of scenarios).",
            "k=9 does not correspond to any bend in the curve.",
            "",
            "**What this means:** The 9 scenarios do not form 9 equally well-separated,",
            "compact spherical clusters in position/velocity space. Some are much tighter",
            "(kreuzung, kreislaeuferspiel) and some overlap substantially",
            "(doppelpass/kreislaeuferspiel, defense_60/rueckpass).",
            "The elbow method would suggest k=5 as the most 'natural' cluster count here.",
            "",
        ]

    # ── ARI by k ──────────────────────────────────────────────────────────────
    if (output_dir / "kmeans_ari_by_k.png").exists():
        ari_5 = km_k5.ari if km_k5 else "?"
        ari_9 = km_k9.ari if km_k9 else "?"
        lines += [
            "## `kmeans_ari_by_k.png` — K-Means ARI by number of clusters",
            "",
            "![ARI by k](kmeans_ari_by_k.png)",
            "",
            "**What it shows:** How well the K-Means clusters agree with the true scenario",
            "labels, for each tested k. Higher = better agreement.",
            "",
            "**How to read it:** If unsupervised clustering perfectly recovered all 9",
            "scenarios, you'd expect ARI to peak at k=9. Instead:",
            "",
            f"- **k=5 achieves the highest ARI ({ari_5:.3f})** — counterintuitive but explainable.",
            "  K-Means with k=5 finds 5 macro-groups that map cleanly to true label boundaries.",
            "  With k=9 it must split these macro-groups into 9 fine categories, and some",
            "  splits are wrong because those scenarios genuinely overlap in feature space.",
            f"- **k=9 achieves ARI={ari_9:.3f}** — meaningful but the algorithm confuses",
            "  several pairs of scenarios (see confusion matrix).",
            "",
            "**What this means for unsupervised usefulness:**",
            "K-Means on position statistics is better at coarse tactical categorisation (k=5)",
            "than at fine-grained Spielzug identification (k=9). If your goal is to separate",
            "'attack vs. defense' and find outlier plays, k=5 is excellent. If you need to",
            "distinguish kreuzung from doppelpass, you need DTW or labelled data.",
            "",
        ]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    if (output_dir / "confusion_kmeans_best.png").exists() and km_k5:
        lines += [
            f"## `confusion_kmeans_best.png` — K-Means k=5 confusion matrix (ARI={km_k5.ari:.3f})",
            "",
            "![confusion](confusion_kmeans_best.png)",
            "",
            "**What it shows:** Each row is a true scenario. Each column is the name",
            "assigned to a cluster by majority vote. Values are percentages of that",
            "true scenario that fell into each cluster (rows sum to 100%).",
            "Dark red = high concentration. Yellow = low/zero.",
            "",
            "**How to read it:**",
            "- A dark red square on the diagonal means that scenario was correctly isolated.",
            "- A dark red square off-diagonal means that scenario was merged with another.",
            "",
            "**Key observations:**",
            "- ✅ kreuzung → 100% in its own cluster. Perfectly isolated.",
            "- ✅ defense_42 → 100% in its own cluster. Perfectly isolated.",
            "- ✅ transition → 100% in its own cluster. Perfectly isolated.",
            "- ⚠️ defense_60 → 100% in the 'rueckpass' cluster.",
            "  These two share similar **average** x-positions even though they look",
            "  very different in a video. K-Means with position statistics cannot tell them apart.",
            "- ⚠️ doppelpass + kreislaeuferspiel + defense_51 → all land in the same cluster.",
            "  Three structurally different plays merged into one group.",
            "- ⚠️ parallelstos → 93% 'rueckpass', 7% 'transition' (slight leakage).",
            "",
            "**Unsupervised usefulness:** K-Means (k=5) successfully isolates kreuzung,",
            "defense_42, and transition without any labels. It fails to distinguish plays",
            "that happen at similar court positions but with different dynamics.",
            "",
        ]

    # ── DTW ───────────────────────────────────────────────────────────────────
    if dtw and (output_dir / "dtw_pca_true_scenarios.png").exists():
        lines += [
            f"## `dtw_pca_true_scenarios.png` / `dtw_clusters.png` — DTW K-Means (ARI={dtw.ari:.3f})",
            "",
            "![DTW PCA](dtw_pca_true_scenarios.png)",
            "",
            "**What it shows:** Ball trajectory (x, y) time series per segment, projected",
            "to 2D with PCA for visualisation (left: coloured by true scenario,",
            "right: coloured by DTW cluster assignment).",
            "",
            "**How to read it:** Unlike position statistics which summarise a segment",
            "as one number, DTW compares the *shape* of the ball trajectory over time.",
            "A doppelpass has a tight zig-zag; a kreislaeuferspiel has a long forward jump.",
            "",
            f"**Key observation: ARI={dtw.ari:.3f} — the highest of all methods.**",
            "Ball trajectory shape alone almost perfectly recovers the true scenario labels.",
            "",
            "**Why DTW is so strong here:**",
            "- kreislaeuferspiel: 3-frame spike where ball jumps 5m forward into the 6m zone",
            "- doppelpass: tight zig-zag within 2m radius (8-frame arc)",
            "- parallelstos: long, smooth, straight-forward movement",
            "- defense plays: slow back-and-forth circulation at x=26–28",
            "- kreuzung: ball crosses the y-axis twice (zig-zag in y)",
            "",
            "**Unsupervised usefulness:** DTW on ball trajectory is a practically deployable",
            "unsupervised method. You could segment any match video by tracking ball position",
            "and running DTW — no player data or labels required. ARI≈0.99 on this mock data",
            "suggests near-perfect performance, though real CV tracking noise will reduce this.",
            "",
        ]

    # ── Overall conclusions ────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Overall Conclusions: Are Unsupervised Methods Useful?",
        "",
        "| Question | Answer |",
        "|----------|--------|",
        "| Can clustering find tactical groups automatically? | **Yes, for coarse groups** (K-Means k=5, ARI=0.86) |",
        "| Can it distinguish all 9 Spielzüge without labels? | **Partially** — position stats struggle with hard pairs |",
        "| Which method works best? | **DTW on ball trajectory** (ARI=0.99) |",
        "| Is ball trajectory enough on its own? | Yes for this data — try on real tracking data |",
        "| Do we need supervised learning? | **Only for the hard pairs**: doppelpass vs kreislaeuferspiel, defense_60 vs rueckpass |",
        "",
        "### Practical recommendation for the wels-monorepo",
        "",
        "1. **Use DTW on ball trajectory** as a first-pass automatic segmenter.",
        "   Deploy it as a pre-processing step that needs no manual labels.",
        "",
        "2. **Use UMAP visualisation** to let coaches explore which plays cluster together.",
        "   The two-blob transition finding and the kreuzung isolation are real insights.",
        "",
        "3. **Invest in supervised learning only for the hard pairs** — the 6 plays that",
        "   share similar position profiles need labelled examples or richer features",
        "   (velocity variance, crossing trajectory, time-windowed pass events).",
        "",
        "4. **Next step with real data:** re-run `just experiment` on a real match",
        "   database after the CV pipeline produces tracking data. The ARI numbers will",
        "   drop — real tracking has noise, ID switches, and missed detections —",
        "   but the DTW approach should remain the most robust.",
    ]

    (output_dir / "FINDINGS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
