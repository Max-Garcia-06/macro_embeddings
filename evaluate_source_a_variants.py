"""Compare Source A embedding variants against the intro-text baseline.

Usage:
    uv run evaluate_source_a_variants.py BASELINE.parquet VARIANT.parquet [VARIANT2.parquet ...]

The analysis county set is fixed once from the baseline parquet (stub filter,
50-states filter, centroid match) and reused for every variant, so metric
differences come from the embeddings, not from set composition. Results are
printed as a comparison table and persisted to analysis-output/variant-eval.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from analyze_source_a_clusters import (
    RANDOM_SEED,
    N_PERMUTATIONS,
    filter_to_fifty_states,
    mantel_test,
)
from analyze_source_a_similarity import (
    MIN_CONTENT_LENGTH,
    drop_stub_counties,
    haversine_distance_matrix,
)
from ingest_source_a import configure_logging
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

FAR_APART_PERCENTILE: float = 75.0
TOP_FAR_PAIR_COUNT: int = 5
OUTPUT_JSON_PATH: Path = (
    Path(__file__).resolve().parent / "analysis-output" / "variant-eval.json"
)

# The five far-apart-but-similar pairs from
# analysis-output/source-a-findings.md section 3.4 — the boilerplate
# mechanisms this experiment line is trying to remove.
TRACKED_BOILERPLATE_PAIRS: list[tuple[str, str]] = [
    ("Lincoln County, Kansas", "Lincoln County, Oregon"),
    ("Montgomery County, Alabama", "Stutsman County, North Dakota"),
    ("Stutsman County, North Dakota", "Williamsburg County, South Carolina"),
    ("Franklin County, Maine", "Franklin County, Nebraska"),
    ("Stutsman County, North Dakota", "Providence County, Rhode Island"),
]

REPORT_METRICS: list[str] = [
    "n_counties",
    "pairwise_similarity_mean",
    "pairwise_similarity_std",
    "mantel_r",
    "mantel_p",
    "silhouette_k2",
    "tracked_pair_mean",
]

logger = logging.getLogger(__name__)


def build_analysis_frame(baseline_df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Fix the analysis county set from the baseline parquet.

    Args:
        baseline_df: Baseline parquet contents (county_name, fips_code,
            raw_intro_text, embedding).
        centroids: Output of fetch_county_centroids (fips_code, lat, lon).

    Returns:
        DataFrame with county_name, fips_code, lat, lon for every county in
        the fixed analysis set.
    """
    df = drop_stub_counties(baseline_df, MIN_CONTENT_LENGTH)
    df = filter_to_fifty_states(df)
    df = df.merge(centroids, on="fips_code", how="inner")
    logger.info("Analysis set fixed at %d counties", len(df))
    return df[["county_name", "fips_code", "lat", "lon"]].reset_index(drop=True)


def evaluate_variant(
    variant_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> dict:
    """Compute the comparison metrics for one embedding variant.

    Args:
        variant_df: DataFrame with county_name and embedding columns.
        analysis_df: Fixed analysis frame (county_name, lat, lon) whose row
            order matches distance_matrix.
        distance_matrix: (n, n) pairwise haversine distances for analysis_df.

    Returns:
        Metrics dict (see REPORT_METRICS, plus tracked_pair_similarity and
        top_far_similar_pairs).

    Raises:
        ValueError: If the variant is missing any analysis-set county.
    """
    merged = analysis_df.merge(
        variant_df[["county_name", "embedding"]], on="county_name", how="left"
    )
    if merged["embedding"].isna().any():
        missing = int(merged["embedding"].isna().sum())
        raise ValueError(f"Variant is missing {missing} analysis-set county(ies)")

    embeddings = np.vstack(merged["embedding"].to_numpy())
    names = merged["county_name"].to_numpy()
    similarity = embeddings @ embeddings.T
    row_idx, col_idx = np.triu_indices(len(merged), k=1)
    sims_flat = similarity[row_idx, col_idx]
    dist_flat = distance_matrix[row_idx, col_idx]

    mantel_r, mantel_p = mantel_test(similarity, distance_matrix, N_PERMUTATIONS, RANDOM_SEED)
    labels = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10).fit_predict(embeddings)
    silhouette = float(silhouette_score(embeddings, labels, random_state=RANDOM_SEED))

    name_to_idx = {name: i for i, name in enumerate(names)}
    tracked: dict[str, float] = {}
    for a, b in TRACKED_BOILERPLATE_PAIRS:
        if a in name_to_idx and b in name_to_idx:
            tracked[f"{a} | {b}"] = float(similarity[name_to_idx[a], name_to_idx[b]])

    far_mask = dist_flat >= np.percentile(dist_flat, FAR_APART_PERCENTILE)
    order = np.argsort(sims_flat[far_mask])[::-1][:TOP_FAR_PAIR_COUNT]
    far_rows = row_idx[far_mask][order]
    far_cols = col_idx[far_mask][order]
    top_far_pairs = [
        {
            "pair": f"{names[i]} | {names[j]}",
            "similarity": float(similarity[i, j]),
            "distance_km": float(distance_matrix[i, j]),
        }
        for i, j in zip(far_rows, far_cols)
    ]

    return {
        "n_counties": int(len(merged)),
        "pairwise_similarity_mean": float(sims_flat.mean()),
        "pairwise_similarity_std": float(sims_flat.std()),
        "mantel_r": mantel_r,
        "mantel_p": mantel_p,
        "silhouette_k2": silhouette,
        "tracked_pair_similarity": tracked,
        "tracked_pair_mean": float(np.mean(list(tracked.values()))) if tracked else None,
        "top_far_similar_pairs": top_far_pairs,
    }


def main() -> None:
    """Evaluate every given parquet against the first (baseline) one."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquets", nargs="+", type=Path, help="baseline first, then variants")
    args = parser.parse_args()

    centroids = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    baseline_df = pd.read_parquet(args.parquets[0])
    analysis_df = build_analysis_frame(baseline_df, centroids)
    distance_matrix = haversine_distance_matrix(
        analysis_df["lat"].to_numpy(dtype=float), analysis_df["lon"].to_numpy(dtype=float)
    )

    results: dict[str, dict] = {}
    for path in args.parquets:
        logger.info("Evaluating %s ...", path.name)
        results[path.name] = evaluate_variant(pd.read_parquet(path), analysis_df, distance_matrix)

    header = f"{'metric':<28}" + "".join(f"{name:>36}" for name in results)
    logger.info("\n%s", header)
    for metric in REPORT_METRICS:
        row = f"{metric:<28}" + "".join(
            f"{(results[name][metric] if results[name][metric] is not None else float('nan')):>36.4f}"
            for name in results
        )
        logger.info("%s", row)
    for name, result in results.items():
        logger.info("Top far-similar pairs — %s:", name)
        for entry in result["top_far_similar_pairs"]:
            logger.info(
                "  %.3f @ %5.0f km  %s", entry["similarity"], entry["distance_km"], entry["pair"]
            )

    payload = {
        "generated": datetime.date.today().isoformat(),
        "baseline": args.parquets[0].name,
        "seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "results": results,
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()
