"""Full-corpus validation: LLM-cleaned (gemma2:9b) vs. baseline (regex-cleaned) embeddings.

analysis-output/source-a-findings.md sec 12 validated the LLM boilerplate
cleaning on a 128-county subset only, and explicitly gated full adoption on
re-running the same comparison at full-corpus scale once
`source_a_embeddings_llm.parquet` finished. This script runs that gate:
pairwise similarity mean/std, Mantel r (geo <-> similarity), and the 8
tracked-pair mean from sec 3.4, baseline vs. LLM-cleaned, on the same
50-state non-stub county set used throughout the rest of the analysis.

Usage:
    uv run python compare_llm_cleaning_full_corpus.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_source_a_clusters import (
    N_PERMUTATIONS,
    RANDOM_SEED,
    filter_to_fifty_states,
    mantel_test,
)
from analyze_source_a_similarity import (
    MIN_CONTENT_LENGTH,
    drop_stub_counties,
    haversine_distance_matrix,
)
from ingest_source_a import configure_logging
from visualize_source_a import (
    CENTROIDS_CACHE_PATH,
    EMBEDDINGS_PARQUET_PATH,
    fetch_county_centroids,
    load_embeddings,
)

logger = logging.getLogger(__name__)

LLM_PARQUET_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "source_a_embeddings_llm.parquet"

# The 5 top-quartile-distance "surprisingly similar" pairs from sec 3.4
# (8 unique counties -- Stutsman County, ND appears in 3 of the 5 pairs).
TRACKED_PAIRS: list[tuple[str, str]] = [
    ("Lincoln County, Kansas", "Lincoln County, Oregon"),
    ("Montgomery County, Alabama", "Stutsman County, North Dakota"),
    ("Stutsman County, North Dakota", "Williamsburg County, South Carolina"),
    ("Franklin County, Maine", "Franklin County, Nebraska"),
    ("Stutsman County, North Dakota", "Providence County, Rhode Island"),
]


def cosine_stats(embeddings: np.ndarray) -> tuple[float, float]:
    """Compute pairwise cosine-similarity mean/std for L2-normalized embeddings.

    Args:
        embeddings: (n, d) L2-normalized embedding matrix.

    Returns:
        Tuple of (pairwise similarity mean, pairwise similarity std).
    """
    similarity = embeddings @ embeddings.T
    triu = similarity[np.triu_indices(len(embeddings), k=1)]
    return float(triu.mean()), float(triu.std())


def tracked_pair_similarities(df: pd.DataFrame) -> list[float]:
    """Compute cosine similarity for each sec 3.4 tracked pair, if present.

    Args:
        df: DataFrame with `county_name` and `embedding` columns.

    Returns:
        List of per-pair cosine similarities (skips any pair missing a county).
    """
    lookup = dict(zip(df["county_name"], df["embedding"]))
    similarities = []
    for county_a, county_b in TRACKED_PAIRS:
        if county_a not in lookup or county_b not in lookup:
            logger.warning("Tracked pair missing: '%s' / '%s'", county_a, county_b)
            continue
        vector_a = np.asarray(lookup[county_a])
        vector_b = np.asarray(lookup[county_b])
        similarities.append(float(vector_a @ vector_b))
    return similarities


def main() -> None:
    """Run the full-corpus baseline-vs-LLM-cleaned comparison."""
    configure_logging()

    baseline_raw = pd.read_parquet(EMBEDDINGS_PARQUET_PATH)
    llm_raw = pd.read_parquet(LLM_PARQUET_PATH)

    baseline_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    baseline_df = filter_to_fifty_states(baseline_df)
    baseline_df = drop_stub_counties(baseline_df, MIN_CONTENT_LENGTH)

    common_names = set(baseline_df["county_name"]) & set(llm_raw["county_name"])
    logger.info(
        "Common county set: %d (baseline filtered: %d, llm total: %d)",
        len(common_names),
        len(baseline_df),
        len(llm_raw),
    )
    baseline_df = baseline_df[baseline_df["county_name"].isin(common_names)].reset_index(drop=True)
    llm_df = llm_raw[llm_raw["county_name"].isin(common_names)].reset_index(drop=True)

    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    baseline_merged = baseline_df.merge(centroids_df, on="fips_code", how="inner")
    llm_merged = llm_df.merge(centroids_df, on="fips_code", how="inner")

    common_fips = set(baseline_merged["fips_code"]) & set(llm_merged["fips_code"])
    baseline_merged = (
        baseline_merged[baseline_merged["fips_code"].isin(common_fips)]
        .sort_values("fips_code")
        .reset_index(drop=True)
    )
    llm_merged = (
        llm_merged[llm_merged["fips_code"].isin(common_fips)].sort_values("fips_code").reset_index(drop=True)
    )
    logger.info("Final aligned county set (has centroid coords): %d", len(baseline_merged))

    distance_matrix = haversine_distance_matrix(
        baseline_merged["lat"].to_numpy(dtype=float), baseline_merged["lon"].to_numpy(dtype=float)
    )

    baseline_embeddings = np.vstack(baseline_merged["embedding"].to_numpy())
    llm_embeddings = np.vstack(llm_merged["embedding"].to_numpy())

    print(f"\n{'Variant':<16}{'N':>6}{'Sim mean':>10}{'Sim std':>10}{'Mantel r':>12}{'p':>10}")
    for name, embeddings in [("baseline", baseline_embeddings), ("LLM-cleaned", llm_embeddings)]:
        sim_mean, sim_std = cosine_stats(embeddings)
        similarity_matrix = embeddings @ embeddings.T
        mantel_r, mantel_p = mantel_test(similarity_matrix, distance_matrix, N_PERMUTATIONS, RANDOM_SEED)
        print(
            f"{name:<16}{len(embeddings):>6}{sim_mean:>10.3f}{sim_std:>10.3f}{mantel_r:>12.3f}{mantel_p:>10.3f}"
        )

    baseline_pair_sims = tracked_pair_similarities(baseline_raw)
    llm_pair_sims = tracked_pair_similarities(llm_raw)
    baseline_tp_mean = float(np.mean(baseline_pair_sims))
    llm_tp_mean = float(np.mean(llm_pair_sims))
    print(
        f"\ntracked_pair_mean (sec 3.4, n={len(TRACKED_PAIRS)}): "
        f"baseline={baseline_tp_mean:.3f} llm={llm_tp_mean:.3f} "
        f"delta={llm_tp_mean - baseline_tp_mean:+.3f}"
    )
    print(f"baseline pair sims: {[f'{s:.3f}' for s in baseline_pair_sims]}")
    print(f"llm pair sims:      {[f'{s:.3f}' for s in llm_pair_sims]}")


if __name__ == "__main__":
    main()
