"""US map visualization for Source C county economic-velocity data.

Loads `source_c_fred.parquet`, merges in county centroids, and plots each
county as a bubble on a US map at its geographic centroid -- one map colored
by unemployment velocity, one by GDP velocity.

Output: `source_c_map_unemployment.html`, `source_c_map_gdp.html`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE_C_PARQUET_PATH: Path = REPO_ROOT / "data" / "source_c_fred.parquet"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_UNEMPLOYMENT_HTML_PATH: Path = OUTPUTS_DIR / "source_c_map_unemployment.html"
OUTPUT_GDP_HTML_PATH: Path = OUTPUTS_DIR / "source_c_map_gdp.html"

# Diverging pair from the dataviz reference palette: blue <-> gray <-> red,
# matching visualize_source_a.py's chart chrome.
DIVERGING_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#2a78d6"),
    (0.5, "#f0efec"),
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


def load_source_c(parquet_path: Path) -> pd.DataFrame:
    """Load Source C velocity data.

    Args:
        parquet_path: Path to `source_c_fred.parquet`.

    Returns:
        DataFrame as written by `ingest_source_c.py`.
    """
    return pd.read_parquet(parquet_path)


def build_velocity_map(
    df: pd.DataFrame, value_column: str, title: str, colorbar_title: str
) -> "px.Figure":
    """Build an interactive US bubble map colored by a velocity column.

    Rows where `value_column` is null (e.g. counties missing a GDP series)
    are dropped rather than plotted as zero, to avoid implying "no velocity"
    where the truth is "no data".

    Args:
        df: DataFrame with `county_name`, `fips_code`, `lat`, `lon`, and `value_column`.
        value_column: Column to color by, e.g. "unemployment_velocity".
        title: Map title.
        colorbar_title: Label for the color scale.

    Returns:
        Plotly Figure ready to export.
    """
    plot_df = df.dropna(subset=[value_column])
    bound = float(plot_df[value_column].abs().max())

    fig = px.scatter_geo(
        plot_df,
        lat="lat",
        lon="lon",
        color=value_column,
        hover_name="county_name",
        hover_data={value_column: ":.3f", "fips_code": True, "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=DIVERGING_COLORSCALE,
        range_color=(-bound, bound),
        title=f"{title} ({len(plot_df)}/{len(df)} counties with data)",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title=colorbar_title))

    extremes = pd.concat(
        [
            plot_df.nlargest(EXTREME_COUNTY_LABEL_COUNT, value_column),
            plot_df.nsmallest(EXTREME_COUNTY_LABEL_COUNT, value_column),
        ]
    )
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
    """Run the Source C map visualization pipeline."""
    configure_logging()

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)

    merged = source_c_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    unemployment_fig = build_velocity_map(
        merged,
        "unemployment_velocity",
        "Source C: unemployment rate velocity (pp/year, 3-yr rolling)",
        "pp/year",
    )
    unemployment_fig.write_html(OUTPUT_UNEMPLOYMENT_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_UNEMPLOYMENT_HTML_PATH)

    gdp_fig = build_velocity_map(
        merged,
        "gdp_velocity",
        "Source C: real GDP velocity ($k/year, 3-yr rolling)",
        "$k/year",
    )
    gdp_fig.write_html(OUTPUT_GDP_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_GDP_HTML_PATH)


if __name__ == "__main__":
    main()
