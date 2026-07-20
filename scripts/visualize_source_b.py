"""US map visualization for Source B county industry-specialization data.

Loads `source_b_qcew.parquet`, derives the dominant-sector label and its LQ
magnitude (`analyze_source_b_industry_mix.compute_dominant_sector`), merges
in county centroids, and plots each county as a bubble on a US map -- one
colored by dominant NAICS-2 sector, one by specialization magnitude
(dominant LQ).

Output: `source_b_map_dominant_sector.html`, `source_b_map_specialization.html`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyze_source_b_industry_mix import (
    NAICS2_COLORS,
    SOURCE_B_PARQUET_PATH,
    compute_dominant_sector,
    load_source_b,
)
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_SECTOR_HTML_PATH: Path = OUTPUTS_DIR / "source_b_map_dominant_sector.html"
OUTPUT_SPECIALIZATION_HTML_PATH: Path = OUTPUTS_DIR / "source_b_map_specialization.html"

# Sequential palette (surface -> accent) for the specialization-magnitude map.
SPECIALIZATION_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#fcfcfb"),
    (0.5, "#e8a33d"),
    (1.0, "#e34948"),
]
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"

EXTREME_COUNTY_LABEL_COUNT: int = 3

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_dominant_sector_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by dominant NAICS-2 sector.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `dominant_industry`.

    Returns:
        Plotly Figure ready to export.
    """
    order = [label for label in NAICS2_COLORS if label in df["dominant_industry"].unique()]
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="dominant_industry",
        color_discrete_map=NAICS2_COLORS,
        category_orders={"dominant_industry": order},
        hover_name="county_name",
        hover_data={"dominant_industry": True, "dominant_lq": ":.2f", "lat": False, "lon": False},
        scope="usa",
        title="Source B: dominant industry sector (highest LQ), 2025 Q4",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(legend_title="Dominant sector")
    return fig


def build_specialization_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by specialization magnitude.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `dominant_lq`.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="dominant_lq",
        hover_name="county_name",
        hover_data={"dominant_industry": True, "dominant_lq": ":.2f", "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=SPECIALIZATION_COLORSCALE,
        title="Source B: industrial specialization magnitude (dominant-sector LQ), 2025 Q4",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title="Dominant LQ"))

    extremes = df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "dominant_lq")
    fig.add_trace(
        go.Scattergeo(
            lat=extremes["lat"],
            lon=extremes["lon"],
            mode="text",
            text=extremes["county_name"],
            textposition="top center",
            textfont=dict(size=9, color=PRIMARY_INK_COLOR),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def main() -> None:
    """Run the Source B map visualization pipeline."""
    configure_logging()

    source_b_df = load_source_b(SOURCE_B_PARQUET_PATH)
    source_b_df = source_b_df.join(compute_dominant_sector(source_b_df))

    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = source_b_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    sector_fig = build_dominant_sector_map(merged)
    sector_fig.write_html(OUTPUT_SECTOR_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_SECTOR_HTML_PATH)

    specialization_fig = build_specialization_map(merged.dropna(subset=["dominant_lq"]))
    specialization_fig.write_html(OUTPUT_SPECIALIZATION_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_SPECIALIZATION_HTML_PATH)


if __name__ == "__main__":
    main()
