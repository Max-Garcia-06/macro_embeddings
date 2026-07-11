"""Persist and visualize the statistics behind Source A's existing EDA scripts.

`visualize_source_a.py`, `analyze_source_a_similarity.py`, and
`analyze_source_a_clusters.py` have already been run once each, but the
numbers that matter most (PCA explained variance, the full k-sweep,
silhouette score, Mantel r/p) are only ever logged to stdout -- never
persisted to a file. This script re-invokes those scripts' existing,
side-effect-free functions with their original seeds/constants so the
numbers are provably identical to what was already trusted, writes them to
`analysis-output/stats.json`, and renders three static summary figures from
that same JSON.

Output: `analysis-output/stats.json`, `analysis-output/figures/*.png`,
`analysis-output/figures/source-a-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from analyze_source_a_clusters import (
    K_CANDIDATES,
    N_PERMUTATIONS,
    RANDOM_SEED,
    SILHOUETTE_SAMPLE_SIZE,
    TOP_REPRESENTATIVE_COUNT,
    filter_to_fifty_states,
    mantel_test,
    summarize_clusters,
)
from analyze_source_a_similarity import (
    MIN_CONTENT_LENGTH,
    build_pairwise_table,
    drop_stub_counties,
    haversine_distance_matrix,
)
from sklearn.metrics import silhouette_score
from visualize_source_a import (
    CENTROIDS_CACHE_PATH,
    EMBEDDINGS_PARQUET_PATH,
    compute_pc1,
    fetch_county_centroids,
    load_embeddings,
)

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "stats.json"

# Number of highest/lowest-PC1 counties to label on the histogram (matches
# EXTREME_COUNTY_LABEL_COUNT in visualize_source_a.py).
PC1_LABEL_COUNT: int = 3

# Dataviz reference palette (matches the existing EDA scripts' chart chrome).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
GRIDLINE_COLOR: str = "#e1e0d9"
PRIMARY_FILL_COLOR: str = "#2a78d6"
ACCENT_FILL_COLOR: str = "#e34948"
BASELINE_COLOR: str = "#52514e"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def choose_k_with_scores(
    embeddings: np.ndarray, k_candidates: range, sample_size: int
) -> tuple[int, float, dict[int, float]]:
    """Re-run the silhouette k-sweep, keeping every k's score (not just the winner).

    Mirrors analyze_source_a_clusters.choose_k exactly (same seed, same
    n_init, same sampling), but returns the full sweep for the
    stats-appendix k-sensitivity note.

    Args:
        embeddings: (n, d) L2-normalized embedding matrix.
        k_candidates: Candidate values of k to try.
        sample_size: Number of points to subsample when scoring each k.

    Returns:
        Tuple of (best k, best silhouette score, {k: silhouette score}).
    """
    scores: dict[int, float] = {}
    for k in k_candidates:
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(
            embeddings
        )
        scores[k] = float(
            silhouette_score(
                embeddings,
                labels,
                sample_size=min(sample_size, len(embeddings)),
                random_state=RANDOM_SEED,
            )
        )
        logger.info("k=%d silhouette=%.4f", k, scores[k])
    best_k = max(scores, key=scores.get)
    logger.info("Selected k=%d (silhouette=%.4f)", best_k, scores[best_k])
    return best_k, scores[best_k], scores


def render_similarity_vs_distance_figure(
    pairs_df: pd.DataFrame, mantel_r: float, mantel_p: float, n_permutations: int
) -> None:
    """Render the main figure: similarity-vs-distance hexbin with Mantel annotation.

    Args:
        pairs_df: Output of build_pairwise_table.
        mantel_r: Observed Mantel correlation.
        mantel_p: Mantel permutation p-value.
        n_permutations: Number of permutations used for the Mantel test.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    hb = ax.hexbin(
        pairs_df["distance_km"],
        pairs_df["similarity"],
        gridsize=60,
        cmap="Blues",
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax, label="Pair count")

    slope, intercept = np.polyfit(pairs_df["distance_km"], pairs_df["similarity"], 1)
    trend_x = np.array([pairs_df["distance_km"].min(), pairs_df["distance_km"].max()])
    ax.plot(trend_x, slope * trend_x + intercept, color=BASELINE_COLOR, linestyle="--", linewidth=2)

    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("Embedding cosine similarity")
    ax.set_title("Source A: county-pair similarity vs. geographic distance")
    ax.text(
        0.02,
        0.02,
        f"Mantel r = {mantel_r:.4f}, p = {mantel_p:.4f} ({n_permutations} permutations)",
        transform=ax.transAxes,
        fontsize=10,
        color=PRIMARY_INK_COLOR,
        verticalalignment="bottom",
        bbox=dict(facecolor=SURFACE_COLOR, edgecolor=GRIDLINE_COLOR, boxstyle="round,pad=0.4"),
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure-01-similarity-vs-distance.png", dpi=200)
    plt.close(fig)


def render_pc1_distribution_figure(
    df: pd.DataFrame, explained_variance_ratio: float
) -> None:
    """Render the supporting figure: PC1 histogram with extreme counties labeled.

    Args:
        df: DataFrame with `county_name` and `pc1` columns.
        explained_variance_ratio: Fraction of embedding variance PC1 captures.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.hist(df["pc1"], bins=60, color=PRIMARY_FILL_COLOR, edgecolor=SURFACE_COLOR)
    ax.set_xlabel("PC1 value")
    ax.set_ylabel("County count")
    ax.set_title(f"Source A: PC1 distribution ({explained_variance_ratio:.1%} of variance explained)")

    # Extreme counties cluster tightly in x at each tail, so label them with
    # a text box per tail (ranked list) instead of individual in-plot labels,
    # which overlap illegibly when the underlying pc1 values are close.
    top_counties = df.nlargest(PC1_LABEL_COUNT, "pc1")
    bottom_counties = df.nsmallest(PC1_LABEL_COUNT, "pc1")
    top_text = "Highest PC1:\n" + "\n".join(top_counties["county_name"])
    bottom_text = "Lowest PC1:\n" + "\n".join(bottom_counties["county_name"])
    box_style = dict(facecolor=SURFACE_COLOR, edgecolor=GRIDLINE_COLOR, boxstyle="round,pad=0.4")
    ax.text(
        0.98, 0.97, top_text, transform=ax.transAxes, fontsize=8, color=PRIMARY_INK_COLOR,
        ha="right", va="top", bbox=box_style,
    )
    ax.text(
        0.02, 0.97, bottom_text, transform=ax.transAxes, fontsize=8, color=PRIMARY_INK_COLOR,
        ha="left", va="top", bbox=box_style,
    )
    for pc1_value in pd.concat([top_counties["pc1"], bottom_counties["pc1"]]):
        ax.axvline(pc1_value, color=GRIDLINE_COLOR, linewidth=1, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure-02-pc1-distribution.png", dpi=200)
    plt.close(fig)


def render_cluster_coherence_figure(cluster_summary_df: pd.DataFrame) -> None:
    """Render the supporting figure: per-cluster geographic coherence vs. baseline.

    Args:
        cluster_summary_df: Per-cluster rows from summarize_clusters (one row
            per representative county; `cluster`, `cluster_size`,
            `mean_intra_cluster_km`, `corpus_mean_pairwise_km` are repeated
            per cluster).
    """
    per_cluster = cluster_summary_df.drop_duplicates("cluster").sort_values("cluster")
    corpus_mean = float(per_cluster["corpus_mean_pairwise_km"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    labels = [f"Cluster {c}\n(n={n})" for c, n in zip(per_cluster["cluster"], per_cluster["cluster_size"])]
    ax.bar(labels, per_cluster["mean_intra_cluster_km"], color=PRIMARY_FILL_COLOR)
    ax.axhline(corpus_mean, color=ACCENT_FILL_COLOR, linestyle="--", linewidth=2, label=f"Corpus mean ({corpus_mean:.0f} km)")
    ax.set_ylabel("Mean intra-cluster distance (km)")
    ax.set_title("Source A: cluster geographic coherence vs. corpus baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure-03-cluster-coherence.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as stats.json).
    """
    lines = [
        "# Source A Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Counties in embeddings parquet | {stats['pc1_n_counties']} |",
        f"| Counties used for clustering/Mantel test | {stats['clustering_n_counties']} |",
        f"| PC1 explained variance ratio | {stats['pc1_explained_variance_ratio']:.4f} |",
        f"| Selected k (silhouette) | {stats['selected_k']} |",
        f"| Silhouette score at selected k | {stats['selected_k_silhouette']:.4f} |",
        f"| Mantel r | {stats['mantel_r']:.4f} |",
        f"| Mantel p-value | {stats['mantel_p']:.4f} |",
        f"| Mantel permutations | {stats['mantel_n_permutations']} |",
        f"| Corpus mean pairwise distance (km) | {stats['corpus_mean_pairwise_km']:.0f} |",
    ]
    for cluster_id, size in stats["cluster_sizes"].items():
        mean_km = stats["cluster_mean_intra_km"][cluster_id]
        lines.append(f"| Cluster {cluster_id} size / mean intra-cluster km | {size} / {mean_km:.0f} |")

    (FIGURES_DIR / "source-a-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source A's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = embeddings_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon"]
    )

    # PC1 matches visualize_source_a.py exactly: computed on the merged-with-
    # centroids set, before the stub/50-state filtering that clustering uses.
    pc1, pc1_explained_variance_ratio = compute_pc1(merged["embedding"])
    pc1_df = merged.assign(pc1=pc1)[["county_name", "pc1"]]
    render_pc1_distribution_figure(pc1_df, pc1_explained_variance_ratio)

    # Clustering/Mantel pipeline matches analyze_source_a_clusters.py's
    # main() exactly: drop stubs, then restrict to the 50 states.
    filtered = drop_stub_counties(merged, MIN_CONTENT_LENGTH).reset_index(drop=True)
    filtered = filter_to_fifty_states(filtered)
    logger.info("Recomputing clustering + Mantel test for %d counties...", len(filtered))

    embeddings = np.vstack(filtered["embedding"].to_numpy())
    best_k, best_silhouette, silhouette_by_k = choose_k_with_scores(
        embeddings, K_CANDIDATES, SILHOUETTE_SAMPLE_SIZE
    )
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    distance_matrix = haversine_distance_matrix(
        filtered["lat"].to_numpy(), filtered["lon"].to_numpy()
    )
    cluster_summary_df = summarize_clusters(
        filtered, embeddings, labels, kmeans.cluster_centers_, distance_matrix, TOP_REPRESENTATIVE_COUNT
    )
    render_cluster_coherence_figure(cluster_summary_df)

    similarity_matrix = embeddings @ embeddings.T
    mantel_r, mantel_p = mantel_test(similarity_matrix, distance_matrix, N_PERMUTATIONS, RANDOM_SEED)
    logger.info("Mantel test: r=%.4f, p=%.4f (%d permutations)", mantel_r, mantel_p, N_PERMUTATIONS)

    pairs_df = build_pairwise_table(filtered)
    render_similarity_vs_distance_figure(pairs_df, mantel_r, mantel_p, N_PERMUTATIONS)

    per_cluster = cluster_summary_df.drop_duplicates("cluster").sort_values("cluster")
    stats = {
        "pc1_n_counties": int(len(pc1_df)),
        "pc1_explained_variance_ratio": pc1_explained_variance_ratio,
        "clustering_n_counties": int(len(filtered)),
        "silhouette_scores_by_k": {int(k): v for k, v in silhouette_by_k.items()},
        "selected_k": int(best_k),
        "selected_k_silhouette": float(best_silhouette),
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "mantel_n_permutations": N_PERMUTATIONS,
        "mantel_seed": RANDOM_SEED,
        "corpus_mean_pairwise_km": float(per_cluster["corpus_mean_pairwise_km"].iloc[0]),
        "cluster_sizes": {str(row.cluster): int(row.cluster_size) for row in per_cluster.itertuples()},
        "cluster_mean_intra_km": {
            str(row.cluster): float(row.mean_intra_cluster_km) for row in per_cluster.itertuples()
        },
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
