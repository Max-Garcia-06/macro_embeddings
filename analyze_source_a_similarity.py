"""Pairwise similarity-vs-distance EDA for Source A county embeddings.

Computes cosine similarity (embeddings are already L2-normalized, so this is
a plain dot product) and geographic (haversine) distance for every county
pair, then surfaces two things:

1. A distance-vs-similarity scatter (density background + a fitted trend
   line) showing whether nearby counties tend to read as more similar.
2. A ranked table of "surprising" pairs: county pairs in the top quartile of
   geographic distance that are nonetheless highly similar in embedding
   space -- candidates for "similar despite being far apart."

Output: `source_a_similarity.html` (interactive Plotly scatter) and
`source_a_similarity_pairs.csv` (ranked pair table).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ingest_source_a import strip_boilerplate_phrasing, strip_self_reference
from visualize_source_a import (
    CENTROIDS_CACHE_PATH,
    EMBEDDINGS_PARQUET_PATH,
    fetch_county_centroids,
    load_embeddings,
)

OUTPUT_HTML_PATH: Path = Path(__file__).resolve().parent / "source_a_similarity.html"
OUTPUT_PAIRS_CSV_PATH: Path = Path(__file__).resolve().parent / "source_a_similarity_pairs.csv"

EARTH_RADIUS_KM: float = 6371.0088
FAR_APART_PERCENTILE: float = 75.0
TOP_N_PAIRS: int = 20

# Counties whose lead text is almost entirely templated boilerplate (see
# strip_self_reference / strip_boilerplate_phrasing) leave behind only a
# handful of characters once that boilerplate is stripped -- too little
# content for embedding similarity between them to mean anything. 100 chars
# sits at a natural gap in the length distribution (~10th percentile) between
# these near-empty stubs and counties with at least a sentence of real
# narrative content.
MIN_CONTENT_LENGTH: int = 100

# Sequential blue ramp (dataviz reference palette, steps 150-650) for the
# density background; a warm red for the highlighted outlier pairs.
DENSITY_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#fcfcfb"),
    (0.2, "#b7d3f6"),
    (0.4, "#6da7ec"),
    (0.6, "#3987e5"),
    (0.8, "#1c5cab"),
    (1.0, "#104281"),
]
OUTLIER_MARKER_COLOR: str = "#e34948"
TREND_LINE_COLOR: str = "#52514e"

# Chart chrome from the dataviz reference palette (light mode).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
GRIDLINE_COLOR: str = "#e1e0d9"
AXIS_LINE_COLOR: str = "#c3c2b7"

# Number of top outlier pairs to direct-label on the chart (the rest stay
# reachable via hover only).
TOP_LABEL_COUNT: int = 3
# Cycled per label (in rank order) so tightly clustered points don't collide.
LABEL_TEXT_POSITIONS: list[str] = ["top center", "bottom center", "top center"]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def haversine_distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Compute the pairwise great-circle distance matrix in kilometers.

    Args:
        lat: Latitudes in degrees, shape (n,).
        lon: Longitudes in degrees, shape (n,).

    Returns:
        Symmetric (n, n) distance matrix in kilometers.
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def drop_stub_counties(df: pd.DataFrame, min_content_length: int) -> pd.DataFrame:
    """Drop counties whose de-boilerplated intro text is too short to compare.

    Args:
        df: DataFrame with `county_name` and `raw_intro_text` columns.
        min_content_length: Minimum character length of the stripped text
            required to keep a county.

    Returns:
        Filtered DataFrame containing only counties at or above the threshold.
    """
    content_length = df.apply(
        lambda row: len(
            strip_boilerplate_phrasing(
                strip_self_reference(row["raw_intro_text"], row["county_name"])
            )
        ),
        axis=1,
    )
    dropped = df.loc[content_length < min_content_length, "county_name"]
    if not dropped.empty:
        logger.info(
            "Dropping %d stub county(ies) with < %d chars of content: %s",
            len(dropped),
            min_content_length,
            ", ".join(dropped),
        )
    return df.loc[content_length >= min_content_length].reset_index(drop=True)


def build_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a long-format table of all unique county pairs with similarity and distance.

    Args:
        df: DataFrame with `county_name`, `embedding`, `lat`, `lon` (one row per county).

    Returns:
        DataFrame with columns `county_a`, `county_b`, `similarity`, `distance_km`,
        one row per unique unordered pair.
    """
    embeddings = np.vstack(df["embedding"].to_numpy())
    similarity_matrix = embeddings @ embeddings.T
    distance_matrix = haversine_distance_matrix(
        df["lat"].to_numpy(), df["lon"].to_numpy()
    )

    i, j = np.triu_indices(len(df), k=1)
    return pd.DataFrame(
        {
            "county_a": df["county_name"].to_numpy()[i],
            "county_b": df["county_name"].to_numpy()[j],
            "similarity": similarity_matrix[i, j],
            "distance_km": distance_matrix[i, j],
        }
    )


def rank_surprising_pairs(pairs_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Rank county pairs that are highly similar despite being geographically far apart.

    Args:
        pairs_df: Output of build_pairwise_table.
        top_n: Number of top pairs to return.

    Returns:
        Top `top_n` pairs by similarity, restricted to the top quartile of
        pairwise distance, sorted descending by similarity.
    """
    far_threshold = np.percentile(pairs_df["distance_km"], FAR_APART_PERCENTILE)
    far_pairs = pairs_df[pairs_df["distance_km"] >= far_threshold]
    return far_pairs.sort_values("similarity", ascending=False).head(top_n)


def build_scatter(pairs_df: pd.DataFrame, surprising_df: pd.DataFrame) -> go.Figure:
    """Build the distance-vs-similarity density scatter with outliers highlighted.

    Args:
        pairs_df: All county pairs.
        surprising_df: The ranked "far but similar" subset to highlight.

    Returns:
        Plotly Figure ready to export.
    """
    slope, intercept = np.polyfit(pairs_df["distance_km"], pairs_df["similarity"], 1)
    trend_x = np.array([pairs_df["distance_km"].min(), pairs_df["distance_km"].max()])
    trend_y = slope * trend_x + intercept

    fig = go.Figure()
    fig.add_trace(
        go.Histogram2d(
            x=pairs_df["distance_km"],
            y=pairs_df["similarity"],
            colorscale=DENSITY_COLORSCALE,
            nbinsx=60,
            nbinsy=60,
            colorbar=dict(title="Pair count"),
            name="All county pairs",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_x,
            y=trend_y,
            mode="lines",
            line=dict(color=TREND_LINE_COLOR, width=2, dash="dash"),
            name=f"Trend (slope={slope:.2e}/km)",
        )
    )
    pair_names = [
        f"{a} ↔ {b}" for a, b in zip(surprising_df["county_a"], surprising_df["county_b"])
    ]
    # Direct labels drop the state name to stay short enough for these tightly
    # clustered points; the full "county, state" pair is still in the hover text.
    short_pair_names = [
        f"{a.split(',')[0]} ↔ {b.split(',')[0]}"
        for a, b in zip(surprising_df["county_a"], surprising_df["county_b"])
    ]
    direct_labels = [
        name if rank < TOP_LABEL_COUNT else "" for rank, name in enumerate(short_pair_names)
    ]
    text_positions = [
        LABEL_TEXT_POSITIONS[rank % len(LABEL_TEXT_POSITIONS)] if rank < TOP_LABEL_COUNT else "top center"
        for rank in range(len(direct_labels))
    ]
    fig.add_trace(
        go.Scatter(
            x=surprising_df["distance_km"],
            y=surprising_df["similarity"],
            mode="markers+text",
            marker=dict(
                color=OUTLIER_MARKER_COLOR, size=10, line=dict(width=2, color=SURFACE_COLOR)
            ),
            text=direct_labels,
            textposition=text_positions,
            textfont=dict(size=10, color=PRIMARY_INK_COLOR),
            customdata=pair_names,
            hovertemplate="%{customdata}<br>distance=%{x:.0f} km<br>similarity=%{y:.3f}<extra></extra>",
            name="Notable far-but-similar pairs",
        )
    )
    fig.update_layout(
        title="Source A: county-pair embedding similarity vs. geographic distance",
        xaxis=dict(
            title="Geographic distance (km)",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        yaxis=dict(
            title="Embedding cosine similarity",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main() -> None:
    """Run the pairwise similarity-vs-distance EDA pipeline."""
    configure_logging()

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = embeddings_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon"]
    )
    merged = drop_stub_counties(merged, MIN_CONTENT_LENGTH)
    logger.info("Building pairwise table for %d counties...", len(merged))

    pairs_df = build_pairwise_table(merged)
    surprising_df = rank_surprising_pairs(pairs_df, TOP_N_PAIRS)

    surprising_df.to_csv(OUTPUT_PAIRS_CSV_PATH, index=False)
    logger.info("Wrote %d ranked pairs to %s", len(surprising_df), OUTPUT_PAIRS_CSV_PATH)

    fig = build_scatter(pairs_df, surprising_df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote scatter to %s", OUTPUT_HTML_PATH)

    logger.info("Top %d far-but-similar pairs:", TOP_N_PAIRS)
    for row in surprising_df.itertuples():
        logger.info(
            "  %.3f sim | %5.0f km | %s <-> %s",
            row.similarity,
            row.distance_km,
            row.county_a,
            row.county_b,
        )


if __name__ == "__main__":
    main()
