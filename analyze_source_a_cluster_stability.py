"""Cross-seed K-means stability and cluster-coherence permutation test for Source A.

Round-0 insights (`analysis-output/source-a-findings.md`, section 3) flagged
two open gaps in `analyze_source_a_clusters.py`'s k=2
finding: (1) the k=2 solution's stability was only checked at a single seed,
and (2) the ~10% intra-cluster distance deviation from the corpus mean was
never tested against a random-labeling null. This script closes both gaps:

1. **Cross-seed stability**: re-run K-means at the previously-selected k
   across multiple random seeds, and report the pairwise Adjusted Rand Index
   (ARI) between each pair of seeds' label assignments (1.0 = identical
   partition, ~0.0 = no better than chance agreement).
2. **Permutation test on cluster coherence**: for the k=2 clustering (fixed
   seed, same as `analyze_source_a_clusters.py`), test whether each cluster's
   mean intra-cluster geographic distance is more extreme (tighter or looser)
   than what randomly reassigning county labels (same cluster sizes) would
   produce by chance.

Output: log lines only; no new artifacts, to keep this a focused sensitivity
check rather than a second full EDA bundle.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from analyze_source_a_clusters import (
    K_CANDIDATES,
    RANDOM_SEED,
    SILHOUETTE_SAMPLE_SIZE,
    choose_k,
    configure_logging,
    filter_to_fifty_states,
)
from analyze_source_a_similarity import drop_stub_counties, haversine_distance_matrix
from visualize_source_a import (
    CENTROIDS_CACHE_PATH,
    EMBEDDINGS_PARQUET_PATH,
    fetch_county_centroids,
    load_embeddings,
)

MIN_CONTENT_LENGTH: int = 100
STABILITY_SEEDS: list[int] = [42, 7, 123, 2024, 99]
N_PERMUTATIONS: int = 999

logger = logging.getLogger(__name__)


def cross_seed_stability(
    embeddings: np.ndarray, k: int, seeds: list[int], sample_size: int
) -> tuple[list[float], list[float]]:
    """Re-run K-means at a fixed k across multiple seeds and score agreement.

    Args:
        embeddings: (n, d) L2-normalized embedding matrix.
        k: Cluster count to hold fixed across seeds.
        seeds: Random seeds to try.
        sample_size: Subsample size for silhouette scoring.

    Returns:
        Tuple of (per-seed silhouette scores, pairwise ARI scores across all
        seed pairs).
    """
    labelings: list[np.ndarray] = []
    silhouettes: list[float] = []
    for seed in seeds:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(embeddings)
        score = silhouette_score(
            embeddings, labels, sample_size=min(sample_size, len(embeddings)), random_state=seed
        )
        logger.info("seed=%d silhouette=%.4f", seed, score)
        labelings.append(labels)
        silhouettes.append(score)

    ari_scores = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i, j in combinations(range(len(labelings)), 2)
    ]
    return silhouettes, ari_scores


def permutation_test_cluster_coherence(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    """Test each cluster's mean intra-cluster distance against a random-labeling null.

    Shuffles county-to-cluster labels (preserving cluster sizes) to build a
    null distribution of mean intra-cluster distance per cluster, then reports
    two one-sided p-values per cluster: how often random labeling is at least
    as tight, and how often it is at least as loose, as the observed cluster.

    Args:
        distance_matrix: (n, n) pairwise haversine distance matrix, in km.
        labels: (n,) observed cluster assignment per county.
        n_permutations: Number of label permutations for the null distribution.
        seed: Random seed for permutations.

    Returns:
        Mapping of cluster id to a dict with `observed_km`, `p_tighter_than_chance`,
        `p_looser_than_chance`.
    """
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(seed)

    observed: dict[int, float] = {}
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        sub = distance_matrix[np.ix_(idx, idx)]
        observed[lbl] = float(sub[np.triu_indices(len(idx), k=1)].mean())

    null_distributions: dict[int, np.ndarray] = {lbl: np.empty(n_permutations) for lbl in unique_labels}
    for p in range(n_permutations):
        permuted_labels = rng.permutation(labels)
        for lbl in unique_labels:
            idx = np.where(permuted_labels == lbl)[0]
            sub = distance_matrix[np.ix_(idx, idx)]
            null_distributions[lbl][p] = sub[np.triu_indices(len(idx), k=1)].mean()

    results: dict[int, dict[str, float]] = {}
    for lbl in unique_labels:
        null = null_distributions[lbl]
        obs = observed[lbl]
        p_tighter = (np.sum(null <= obs) + 1) / (n_permutations + 1)
        p_looser = (np.sum(null >= obs) + 1) / (n_permutations + 1)
        results[int(lbl)] = {
            "observed_km": obs,
            "p_tighter_than_chance": float(p_tighter),
            "p_looser_than_chance": float(p_looser),
        }
    return results


def main() -> None:
    """Run the cross-seed stability check and cluster-coherence permutation test."""
    configure_logging()

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = embeddings_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon"]
    )
    merged = drop_stub_counties(merged, MIN_CONTENT_LENGTH).reset_index(drop=True)
    merged = filter_to_fifty_states(merged)
    logger.info("Running stability + permutation checks for %d counties...", len(merged))

    embeddings = np.vstack(merged["embedding"].to_numpy())

    best_k = choose_k(embeddings, K_CANDIDATES, SILHOUETTE_SAMPLE_SIZE)

    logger.info("--- Cross-seed stability at k=%d, seeds=%s ---", best_k, STABILITY_SEEDS)
    silhouettes, ari_scores = cross_seed_stability(
        embeddings, best_k, STABILITY_SEEDS, SILHOUETTE_SAMPLE_SIZE
    )
    logger.info(
        "Silhouette across seeds: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
        np.mean(silhouettes),
        np.std(silhouettes),
        np.min(silhouettes),
        np.max(silhouettes),
    )
    logger.info(
        "Pairwise ARI across %d seed pairs: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
        len(ari_scores),
        np.mean(ari_scores),
        np.std(ari_scores),
        np.min(ari_scores),
        np.max(ari_scores),
    )

    logger.info("--- Permutation test on cluster coherence (seed=%d) ---", RANDOM_SEED)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    distance_matrix = haversine_distance_matrix(
        merged["lat"].to_numpy(), merged["lon"].to_numpy()
    )
    corpus_mean_km = distance_matrix[np.triu_indices(len(merged), k=1)].mean()
    logger.info("Corpus-wide mean pairwise distance: %.0f km", corpus_mean_km)

    coherence_results = permutation_test_cluster_coherence(
        distance_matrix, labels, N_PERMUTATIONS, RANDOM_SEED
    )
    for cluster_id, result in sorted(coherence_results.items()):
        logger.info(
            "Cluster %d: observed=%.0f km (n_permutations=%d) | "
            "p(tighter than chance)=%.4f | p(looser than chance)=%.4f",
            cluster_id,
            result["observed_km"],
            N_PERMUTATIONS,
            result["p_tighter_than_chance"],
            result["p_looser_than_chance"],
        )


if __name__ == "__main__":
    main()
