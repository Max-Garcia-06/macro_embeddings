"""US map visualization for Source D trade flow data.

Loads `source_d_faf.parquet`, merges in county centroids, and plots each
county as a bubble on a US map -- one colored by log-scaled total tonnage
(tonnage spans several orders of magnitude, from Loving County, TX's ~9.8K
tons to LA County's ~256K tons, so a linear scale would wash out all but the
largest hubs), one colored by partner concentration (HHI).

Output: `source_d_map_tons.html`, `source_d_map_concentration.html`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyze_source_d_hubs import SOURCE_D_PARQUET_PATH, add_hub_signals, load_source_d
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_TONS_HTML_PATH: Path = OUTPUTS_DIR / "source_d_map_tons.html"
OUTPUT_CONCENTRATION_HTML_PATH: Path = OUTPUTS_DIR / "source_d_map_concentration.html"

# Sequential blue ramp (dataviz reference palette), matching the density
# background used in analyze_source_a_source_c_correlation.py.
SEQUENTIAL_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#fcfcfb"),
    (0.2, "#b7d3f6"),
    (0.4, "#6da7ec"),
    (0.6, "#3987e5"),
    (0.8, "#1c5cab"),
    (1.0, "#104281"),
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


def build_tons_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by log-scaled total tonnage.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `total_tons`.

    Returns:
        Plotly Figure ready to export.
    """
    log_tons = np.log10(df["total_tons"].clip(lower=1))
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color=log_tons,
        hover_name="county_name",
        hover_data={"total_tons": ":.0f", "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=SEQUENTIAL_COLORSCALE,
        title="Source D: total 2022 freight tonnage (log10 scale)",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title="log10(tons)"))

    extremes = df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "total_tons")
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


def build_concentration_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by partner concentration (HHI).

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `mean_partner_hhi`.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="mean_partner_hhi",
        hover_name="county_name",
        hover_data={"mean_partner_hhi": ":.3f", "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=SEQUENTIAL_COLORSCALE,
        title="Source D: mean trade-partner concentration (HHI, 2022)",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title="HHI"))

    extremes = df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "mean_partner_hhi")
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
    """Run the Source D map visualization pipeline."""
    configure_logging()

    source_d_df = add_hub_signals(load_source_d(SOURCE_D_PARQUET_PATH))
    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)

    merged = source_d_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    tons_fig = build_tons_map(merged)
    tons_fig.write_html(OUTPUT_TONS_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_TONS_HTML_PATH)

    concentration_fig = build_concentration_map(merged)
    concentration_fig.write_html(OUTPUT_CONCENTRATION_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_CONCENTRATION_HTML_PATH)


if __name__ == "__main__":
    main()
