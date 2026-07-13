"""Geographic-proximity-vs-economic-divergence EDA for Source C velocity data.

Standardizes (z-score) the two velocity axes -- unemployment velocity (pp/year)
and GDP velocity ($k/year) are on wildly different scales, so a raw Euclidean
distance would be dominated entirely by GDP -- then computes, for every county
pair, that standardized "economic distance" alongside geographic (haversine)
distance.

Source A's similarity script surfaced "far apart geographically but similar
in narrative" as the interesting outlier. For velocity data the more
economically interesting story runs the other way: county pairs that are
geographic neighbors (bottom quartile of distance) but whose economies are
moving in sharply different directions -- a signal invisible in either
source's static snapshot alone.

Output: `source_c_similarity.html` (interactive Plotly scatter) and
`source_c_similarity_pairs.csv` (ranked pair table).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analyze_source_a_similarity import haversine_distance_matrix
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_c_similarity.html"
OUTPUT_PAIRS_CSV_PATH: Path = OUTPUTS_DIR / "source_c_similarity_pairs.csv"

NEARBY_PERCENTILE: float = 25.0
TOP_N_PAIRS: int = 20

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
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
GRIDLINE_COLOR: str = "#e1e0d9"
AXIS_LINE_COLOR: str = "#c3c2b7"

TOP_LABEL_COUNT: int = 3
LABEL_TEXT_POSITIONS: list[str] = ["top center", "bottom center", "top center"]

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_pairwise_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a long-format table of all unique county pairs with economic/geographic distance.

    Args:
        df: DataFrame with `county_name`, `unemployment_velocity`,
            `gdp_velocity`, `lat`, `lon` (one row per county, both
            velocities present).

    Returns:
        DataFrame with columns `county_a`, `county_b`, `economic_distance`,
        `distance_km`, one row per unique unordered pair. `economic_distance`
        is Euclidean distance in z-scored (unemployment_velocity, gdp_velocity)
        space.
    """
    velocities = df[["unemployment_velocity", "gdp_velocity"]].to_numpy()
    standardized = (velocities - velocities.mean(axis=0)) / velocities.std(axis=0)

    i, j = np.triu_indices(len(df), k=1)
    economic_distance = np.linalg.norm(standardized[i] - standardized[j], axis=1)
    geo_distance_matrix = haversine_distance_matrix(df["lat"].to_numpy(), df["lon"].to_numpy())

    return pd.DataFrame(
        {
            "county_a": df["county_name"].to_numpy()[i],
            "county_b": df["county_name"].to_numpy()[j],
            "economic_distance": economic_distance,
            "distance_km": geo_distance_matrix[i, j],
        }
    )


def rank_diverging_neighbors(pairs_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Rank geographically close county pairs with the most divergent economic velocity.

    Args:
        pairs_df: Output of build_pairwise_table.
        top_n: Number of top pairs to return.

    Returns:
        Top `top_n` pairs by economic_distance, restricted to the bottom
        quartile of pairwise geographic distance, sorted descending by
        economic_distance.
    """
    nearby_threshold = np.percentile(pairs_df["distance_km"], NEARBY_PERCENTILE)
    nearby_pairs = pairs_df[pairs_df["distance_km"] <= nearby_threshold]
    return nearby_pairs.sort_values("economic_distance", ascending=False).head(top_n)


def build_scatter(pairs_df: pd.DataFrame, diverging_df: pd.DataFrame) -> go.Figure:
    """Build the distance-vs-economic-divergence density scatter with outliers highlighted.

    Args:
        pairs_df: All county pairs.
        diverging_df: The ranked "close but diverging" subset to highlight.

    Returns:
        Plotly Figure ready to export.
    """
    slope, intercept = np.polyfit(pairs_df["distance_km"], pairs_df["economic_distance"], 1)
    trend_x = np.array([pairs_df["distance_km"].min(), pairs_df["distance_km"].max()])
    trend_y = slope * trend_x + intercept

    counts, x_edges, y_edges = np.histogram2d(
        pairs_df["distance_km"], pairs_df["economic_distance"], bins=60
    )
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=counts.T,
            colorscale=DENSITY_COLORSCALE,
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
        f"{a} ↔ {b}" for a, b in zip(diverging_df["county_a"], diverging_df["county_b"])
    ]
    short_pair_names = [
        f"{a.split(',')[0]} ↔ {b.split(',')[0]}"
        for a, b in zip(diverging_df["county_a"], diverging_df["county_b"])
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
            x=diverging_df["distance_km"],
            y=diverging_df["economic_distance"],
            mode="markers+text",
            marker=dict(
                color=OUTLIER_MARKER_COLOR, size=10, line=dict(width=2, color=SURFACE_COLOR)
            ),
            text=direct_labels,
            textposition=text_positions,
            textfont=dict(size=10, color=PRIMARY_INK_COLOR),
            customdata=pair_names,
            hovertemplate="%{customdata}<br>distance=%{x:.0f} km<br>econ. distance=%{y:.2f}<extra></extra>",
            name="Notable close-but-diverging pairs",
        )
    )
    fig.update_layout(
        title="Source C: county-pair economic divergence vs. geographic distance",
        xaxis=dict(
            title="Geographic distance (km)",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        yaxis=dict(
            title="Economic distance (standardized velocity space)",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=GRIDLINE_COLOR,
            linecolor=AXIS_LINE_COLOR,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def main() -> None:
    """Run the proximity-vs-divergence EDA pipeline."""
    configure_logging()

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = source_c_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon", "unemployment_velocity", "gdp_velocity"]
    )
    logger.info("Building pairwise table for %d counties...", len(merged))

    pairs_df = build_pairwise_table(merged)
    diverging_df = rank_diverging_neighbors(pairs_df, TOP_N_PAIRS)

    diverging_df.to_csv(OUTPUT_PAIRS_CSV_PATH, index=False)
    logger.info("Wrote %d ranked pairs to %s", len(diverging_df), OUTPUT_PAIRS_CSV_PATH)

    fig = build_scatter(pairs_df, diverging_df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote scatter to %s", OUTPUT_HTML_PATH)

    logger.info("Top %d close-but-diverging pairs:", TOP_N_PAIRS)
    for row in diverging_df.itertuples():
        logger.info(
            "  %.2f econ.dist | %5.0f km | %s <-> %s",
            row.economic_distance,
            row.distance_km,
            row.county_a,
            row.county_b,
        )


if __name__ == "__main__":
    main()
