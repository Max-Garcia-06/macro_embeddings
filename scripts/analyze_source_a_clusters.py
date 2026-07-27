"""Clustering and Mantel-test EDA for Source A county embeddings.

Two analyses:

1. K-means clustering of the 1024-dim embeddings (k chosen by silhouette
   score), mapped by county to see whether embedding-space "archetypes" line
   up with geography or cut across it. Each cluster is summarized by its
   most representative counties (nearest to the cluster centroid) and its
   geographic coherence (mean intra-cluster distance vs. the corpus-wide
   mean).
2. A Mantel permutation test that turns the descriptive distance-vs-
   similarity trend from `analyze_source_a_similarity.py` into a p-value:
   correlate the geographic-distance matrix against the embedding-similarity
   matrix, then permute county labels to build a null distribution.

Output: `source_a_clusters.html` (interactive Plotly map) and
`source_a_cluster_summary.csv` (per-cluster representative counties).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analyze_source_a_similarity import drop_stub_counties, haversine_distance_matrix
from visualize_source_a import (
    CENTROIDS_CACHE_PATH,
    EMBEDDINGS_PARQUET_PATH,
    fetch_county_centroids,
    load_embeddings,
)

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_MAP_PATH: Path = OUTPUTS_DIR / "source_a_clusters.html"
OUTPUT_SUMMARY_CSV_PATH: Path = OUTPUTS_DIR / "source_a_cluster_summary.csv"

# Counties whose de-boilerplated intro text is too short to compare; same
# threshold used in analyze_source_a_similarity.py for consistency.
MIN_CONTENT_LENGTH: int = 100

# Suffixes/names identifying non-50-state entries in county_name (Puerto Rico
# municipios and DC), per the display-name conventions in
# ingest_source_a._build_county_display_name.
NON_STATE_SUFFIXES: tuple[str, ...] = (", Puerto Rico",)
NON_STATE_NAMES: tuple[str, ...] = ("District of Columbia",)

RANDOM_SEED: int = 42
K_CANDIDATES: range = range(2, 13)
SILHOUETTE_SAMPLE_SIZE: int = 1000
TOP_REPRESENTATIVE_COUNT: int = 5
N_PERMUTATIONS: int = 499

# Qualitative palette (dataviz reference palette family, extended to 12
# categorical steps) for cluster coloring on the map.
CLUSTER_COLOR_SEQUENCE: list[str] = [
    "#2a78d6", "#e34948", "#3d9a5c", "#e0a72e", "#8558c2", "#2fa8a0",
    "#d6668f", "#7a8c3e", "#c2762e", "#5c6bc0", "#a8562e", "#3e8c7a",
]

SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def choose_k(embeddings: np.ndarray, k_candidates: range, sample_size: int) -> int:
    """Pick the cluster count with the highest silhouette score.

    Args:
        embeddings: (n, d) L2-normalized embedding matrix.
        k_candidates: Candidate values of k to try.
        sample_size: Number of points to subsample when scoring each k
            (silhouette_score is O(n^2); subsampling keeps this tractable).

    Returns:
        The k with the highest silhouette score.
    """
    best_k = k_candidates[0]
    best_score = -1.0
    for k in k_candidates:
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(
            embeddings
        )
        score = silhouette_score(
            embeddings,
            labels,
            sample_size=min(sample_size, len(embeddings)),
            random_state=RANDOM_SEED,
        )
        logger.info("k=%d silhouette=%.4f", k, score)
        if score > best_score:
            best_k, best_score = k, score
    logger.info("Selected k=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


def filter_to_fifty_states(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Puerto Rico municipios and DC, keeping only the 50 states.

    Args:
        df: DataFrame with a `county_name` column.

    Returns:
        Filtered DataFrame containing only 50-state counties.
    """
    is_non_state = df["county_name"].str.endswith(NON_STATE_SUFFIXES) | df[
        "county_name"
    ].isin(NON_STATE_NAMES)
    dropped_count = int(is_non_state.sum())
    if dropped_count:
        logger.info("Dropping %d non-50-state entries (Puerto Rico / DC)", dropped_count)
    return df.loc[~is_non_state].reset_index(drop=True)


def summarize_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    distance_matrix: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    """Summarize each cluster's representative counties and geographic coherence.

    Args:
        df: DataFrame with `county_name` (one row per county, same order as
            `embeddings`/`labels`).
        embeddings: (n, d) L2-normalized embedding matrix.
        labels: (n,) cluster assignment per county.
        centers: (k, d) K-means cluster centers.
        distance_matrix: (n, n) pairwise haversine distance matrix, in km.
        top_n: Number of nearest-to-centroid counties to list per cluster.

    Returns:
        Long-format DataFrame with one row per (cluster, representative
        county), plus cluster size and mean intra-cluster geographic
        distance.
    """
    corpus_mean_distance = distance_matrix[np.triu_indices(len(df), k=1)].mean()

    rows: list[dict] = []
    for cluster_id in np.unique(labels):
        member_idx = np.where(labels == cluster_id)[0]
        center = centers[cluster_id] / np.linalg.norm(centers[cluster_id])
        similarity_to_center = embeddings[member_idx] @ center
        ranked = member_idx[np.argsort(-similarity_to_center)][:top_n]

        if len(member_idx) > 1:
            sub_distances = distance_matrix[np.ix_(member_idx, member_idx)]
            mean_intra_distance = sub_distances[np.triu_indices(len(member_idx), k=1)].mean()
        else:
            mean_intra_distance = float("nan")

        for rank, idx in enumerate(ranked):
            rows.append(
                {
                    "cluster": cluster_id,
                    "cluster_size": len(member_idx),
                    "mean_intra_cluster_km": mean_intra_distance,
                    "corpus_mean_pairwise_km": corpus_mean_distance,
                    "rank": rank,
                    "county_name": df["county_name"].iloc[idx],
                    "similarity_to_centroid": similarity_to_center[np.argsort(-similarity_to_center)][rank],
                }
            )
    return pd.DataFrame(rows)


def build_cluster_map(df: pd.DataFrame) -> "px.Figure":
    """Build a categorical US bubble map colored by cluster assignment.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `cluster` columns.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="cluster",
        hover_name="county_name",
        scope="usa",
        color_discrete_sequence=CLUSTER_COLOR_SEQUENCE,
        title="Source A county embeddings — K-means clusters",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color=SURFACE_COLOR)))
    return fig


def mantel_test(
    similarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Permutation test for correlation between two pairwise matrices.

    Permutes county labels on the distance matrix (keeping the similarity
    matrix fixed) to build a null distribution for the correlation between
    geographic distance and embedding similarity.

    **Why this rather than `stats_utils.permutation_test_corr`**: the two
    inputs are pairwise matrices, so the n(n-1)/2 upper-triangle entries this
    flattens are not independent observations -- every county appears in n-1
    of them, and moving one county perturbs a whole row and column at once.
    Feeding those flattened vectors to an ordinary correlation test would
    treat ~4M dependent pairs (at n=2,849) as ~4M independent draws and return
    a p-value orders of magnitude too small. Permuting *county labels* --
    `perm[row_idx], perm[col_idx]`, which relabels the matrix coherently
    rather than shuffling pair rows independently -- is what preserves that
    dependence structure under the null. Use `permutation_test_corr` for flat
    per-county vectors; use this whenever both inputs are pairwise matrices.

    Caveat: the statistic is still Pearson r on the flattened triangles, so it
    measures linear association only, and Mantel tests are known to have low
    power against spatially structured alternatives.

    Args:
        similarity_matrix: (n, n) pairwise cosine similarity matrix.
        distance_matrix: (n, n) pairwise haversine distance matrix.
        n_permutations: Number of label permutations for the null distribution.
        seed: Random seed for permutations.

    Returns:
        Tuple of (observed Pearson correlation, two-sided permutation p-value).
    """
    n = similarity_matrix.shape[0]
    row_idx, col_idx = np.triu_indices(n, k=1)
    sim_flat = similarity_matrix[row_idx, col_idx]
    dist_flat = distance_matrix[row_idx, col_idx]
    observed_r = float(np.corrcoef(sim_flat, dist_flat)[0, 1])

    rng = np.random.default_rng(seed)
    permuted_rs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        permuted_dist_flat = distance_matrix[perm[row_idx], perm[col_idx]]
        permuted_rs[i] = np.corrcoef(sim_flat, permuted_dist_flat)[0, 1]

    p_value = (np.sum(np.abs(permuted_rs) >= abs(observed_r)) + 1) / (n_permutations + 1)
    return observed_r, float(p_value)


def main() -> None:
    """Run the clustering and Mantel-test EDA pipeline."""
    configure_logging()

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = embeddings_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon"]
    )
    merged = drop_stub_counties(merged, MIN_CONTENT_LENGTH).reset_index(drop=True)
    merged = filter_to_fifty_states(merged)
    logger.info("Running clustering + Mantel test for %d counties...", len(merged))

    embeddings = np.vstack(merged["embedding"].to_numpy())

    best_k = choose_k(embeddings, K_CANDIDATES, SILHOUETTE_SAMPLE_SIZE)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    merged = merged.assign(cluster=labels.astype(str))

    distance_matrix = haversine_distance_matrix(
        merged["lat"].to_numpy(), merged["lon"].to_numpy()
    )
    summary_df = summarize_clusters(
        merged, embeddings, labels, kmeans.cluster_centers_, distance_matrix, TOP_REPRESENTATIVE_COUNT
    )
    summary_df.to_csv(OUTPUT_SUMMARY_CSV_PATH, index=False)
    logger.info("Wrote cluster summary to %s", OUTPUT_SUMMARY_CSV_PATH)
    for cluster_id, group in summary_df.groupby("cluster"):
        logger.info(
            "Cluster %s (n=%d, mean intra-cluster dist=%.0f km vs. corpus mean %.0f km): %s",
            cluster_id,
            group["cluster_size"].iloc[0],
            group["mean_intra_cluster_km"].iloc[0],
            group["corpus_mean_pairwise_km"].iloc[0],
            ", ".join(group["county_name"]),
        )

    fig = build_cluster_map(merged)
    fig.write_html(OUTPUT_MAP_PATH)
    logger.info("Wrote cluster map to %s", OUTPUT_MAP_PATH)

    similarity_matrix = embeddings @ embeddings.T
    observed_r, p_value = mantel_test(
        similarity_matrix, distance_matrix, N_PERMUTATIONS, RANDOM_SEED
    )
    logger.info(
        "Mantel test: r=%.4f, p=%.4f (%d permutations) -- %s",
        observed_r,
        p_value,
        N_PERMUTATIONS,
        "geography significantly predicts embedding similarity"
        if p_value < 0.05
        else "no significant geography/similarity association",
    )


if __name__ == "__main__":
    main()
