"""Joint EDA: does Source A embedding similarity track Source C economic momentum?

`E_macro_extendedProposal.pdf`'s justification for Source A claims Wikipedia
intro-text embeddings carry distinctive economic-transition narrative. Source
A's own findings report (`analysis-output/source-a/source-a-findings.md` §8, §10)
flags this as untested: everything checked there was embedding similarity
against *geography*, not against real economic variables. Source C now
provides those variables (unemployment and GDP velocity per county), so this
script runs the same Mantel-test methodology
(`analyze_source_a_clusters.mantel_test`) against economic distance instead
of geographic distance.

GDP velocity is measured in absolute dollars, which Source C's own report
(`analysis-output/source-c/source-c-findings.md` §5) flags as confounded with economy
size rather than growth rate. To avoid that confound leaking into this
correlation test, economic distance here standardizes unemployment velocity
alongside a locally-computed `gdp_velocity_pct` (`gdp_velocity / gdp_latest`)
rather than raw `gdp_velocity`.

Output: `source_a_source_c_correlation.html` (interactive Plotly scatter) and
`source_a_source_c_correlation_pairs.csv` (ranked pair table).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analyze_source_a_clusters import filter_to_fifty_states, mantel_test
from analyze_source_a_similarity import drop_stub_counties
from visualize_source_a import EMBEDDINGS_PARQUET_PATH, load_embeddings
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_a_source_c_correlation.html"
OUTPUT_PAIRS_CSV_PATH: Path = OUTPUTS_DIR / "source_a_source_c_correlation_pairs.csv"

# Same threshold used in analyze_source_a_similarity.py / analyze_source_a_clusters.py
# for consistency: counties whose de-boilerplated intro text is too short to
# compare are dropped before joining against Source C.
MIN_CONTENT_LENGTH: int = 100

RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 499
SIMILARITY_PERCENTILE: float = 75.0
TOP_N_PAIRS: int = 20

# Sequential blue ramp (dataviz reference palette, steps 150-650) for the
# density background; a warm red for the highlighted outlier pairs -- same
# palette as analyze_source_a_similarity.py / analyze_source_c_similarity.py.
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


def build_economic_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute pairwise Euclidean distance in standardized velocity space.

    Args:
        df: DataFrame with `unemployment_velocity`, `gdp_velocity`, and
            `gdp_latest` columns.

    Returns:
        Symmetric (n, n) distance matrix over z-scored
        (unemployment_velocity, gdp_velocity_pct) space, where
        `gdp_velocity_pct = gdp_velocity / gdp_latest` sidesteps the
        economy-size confound in raw `gdp_velocity` (source-c-findings.md §5).
    """
    gdp_velocity_pct = df["gdp_velocity"] / df["gdp_latest"]
    features = np.column_stack([df["unemployment_velocity"].to_numpy(), gdp_velocity_pct.to_numpy()])
    standardized = (features - features.mean(axis=0)) / features.std(axis=0)
    diff = standardized[:, None, :] - standardized[None, :, :]
    return np.linalg.norm(diff, axis=2)


def build_pairwise_table(
    df: pd.DataFrame, similarity_matrix: np.ndarray, economic_distance_matrix: np.ndarray
) -> pd.DataFrame:
    """Build a long-format table of all unique county pairs with similarity and economic distance.

    Args:
        df: DataFrame with `county_name` (one row per county, same order as
            the two matrices).
        similarity_matrix: (n, n) pairwise embedding cosine similarity matrix.
        economic_distance_matrix: (n, n) pairwise economic distance matrix.

    Returns:
        DataFrame with columns `county_a`, `county_b`, `similarity`,
        `economic_distance`, one row per unique unordered pair.
    """
    i, j = np.triu_indices(len(df), k=1)
    return pd.DataFrame(
        {
            "county_a": df["county_name"].to_numpy()[i],
            "county_b": df["county_name"].to_numpy()[j],
            "similarity": similarity_matrix[i, j],
            "economic_distance": economic_distance_matrix[i, j],
        }
    )


def rank_similar_diverging_pairs(pairs_df: pd.DataFrame, similarity_percentile: float, top_n: int) -> pd.DataFrame:
    """Rank textually similar county pairs by how economically divergent they are.

    Args:
        pairs_df: Output of build_pairwise_table.
        similarity_percentile: Percentile threshold defining "textually similar"
            (e.g. 75.0 keeps the top quartile of embedding similarity).
        top_n: Number of top pairs to return.

    Returns:
        Top `top_n` pairs by economic_distance, restricted to the top
        quartile of embedding similarity, sorted descending by
        economic_distance.
    """
    similar_threshold = np.percentile(pairs_df["similarity"], similarity_percentile)
    similar_pairs = pairs_df[pairs_df["similarity"] >= similar_threshold]
    return similar_pairs.sort_values("economic_distance", ascending=False).head(top_n)


def build_scatter(pairs_df: pd.DataFrame, diverging_df: pd.DataFrame) -> go.Figure:
    """Build the similarity-vs-economic-divergence density scatter with outliers highlighted.

    Args:
        pairs_df: All county pairs.
        diverging_df: The ranked "textually similar but economically
            diverging" subset to highlight.

    Returns:
        Plotly Figure ready to export.
    """
    slope, intercept = np.polyfit(pairs_df["similarity"], pairs_df["economic_distance"], 1)
    trend_x = np.array([pairs_df["similarity"].min(), pairs_df["similarity"].max()])
    trend_y = slope * trend_x + intercept

    counts, x_edges, y_edges = np.histogram2d(
        pairs_df["similarity"], pairs_df["economic_distance"], bins=60
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
            name=f"Trend (slope={slope:.2e}/similarity)",
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
            x=diverging_df["similarity"],
            y=diverging_df["economic_distance"],
            mode="markers+text",
            marker=dict(
                color=OUTLIER_MARKER_COLOR, size=10, line=dict(width=2, color=SURFACE_COLOR)
            ),
            text=direct_labels,
            textposition=text_positions,
            textfont=dict(size=10, color=PRIMARY_INK_COLOR),
            customdata=pair_names,
            hovertemplate="%{customdata}<br>similarity=%{x:.3f}<br>econ. distance=%{y:.2f}<extra></extra>",
            name="Notable similar-but-diverging pairs",
        )
    )
    fig.update_layout(
        title="Source A ↔ Source C: embedding similarity vs. economic-momentum distance",
        xaxis=dict(
            title="Embedding cosine similarity",
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
    """Run the Source A / Source C joint correlation EDA pipeline."""
    configure_logging()

    embeddings_df = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    embeddings_df = drop_stub_counties(embeddings_df, MIN_CONTENT_LENGTH)
    embeddings_df = filter_to_fifty_states(embeddings_df)

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)

    merged = embeddings_df.merge(
        source_c_df[["fips_code", "unemployment_velocity", "gdp_velocity", "gdp_latest"]],
        on="fips_code",
        how="inner",
    ).dropna(subset=["unemployment_velocity", "gdp_velocity", "gdp_latest"]).reset_index(drop=True)
    logger.info(
        "Joined %d counties with both Source A embeddings and Source C velocities", len(merged)
    )

    embeddings = np.vstack(merged["embedding"].to_numpy())
    similarity_matrix = embeddings @ embeddings.T
    economic_distance_matrix = build_economic_distance_matrix(merged)

    observed_r, p_value = mantel_test(
        similarity_matrix, economic_distance_matrix, N_PERMUTATIONS, RANDOM_SEED
    )
    logger.info(
        "Mantel test (embedding similarity vs. economic distance): r=%.4f, p=%.4f (%d permutations) -- %s",
        observed_r,
        p_value,
        N_PERMUTATIONS,
        "economic momentum significantly predicts embedding similarity"
        if p_value < 0.05
        else "no significant embedding-similarity/economic-momentum association",
    )

    pairs_df = build_pairwise_table(merged, similarity_matrix, economic_distance_matrix)
    diverging_df = rank_similar_diverging_pairs(pairs_df, SIMILARITY_PERCENTILE, TOP_N_PAIRS)

    diverging_df.to_csv(OUTPUT_PAIRS_CSV_PATH, index=False)
    logger.info("Wrote %d ranked pairs to %s", len(diverging_df), OUTPUT_PAIRS_CSV_PATH)

    fig = build_scatter(pairs_df, diverging_df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote scatter to %s", OUTPUT_HTML_PATH)

    logger.info("Top %d textually-similar-but-economically-diverging pairs:", TOP_N_PAIRS)
    for row in diverging_df.itertuples():
        logger.info(
            "  %.2f econ.dist | %.3f sim | %s <-> %s",
            row.economic_distance,
            row.similarity,
            row.county_a,
            row.county_b,
        )


if __name__ == "__main__":
    main()
