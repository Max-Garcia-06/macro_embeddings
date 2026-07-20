"""US map visualization for Source E county capital-composition data.

Loads `source_e_irs_soi.parquet`, merges in county centroids, and plots each
county as a bubble on a US map colored by `capital_to_wage_ratio`, mirroring
`visualize_source_c.py`'s single-continuous-signal map structure.

Output: `source_e_map_capital_composition.html`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analyze_source_e_capital_composition import SOURCE_E_PARQUET_PATH, load_source_e
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_e_map_capital_composition.html"

# Sequential palette (surface -> accent), matching visualize_source_b.py's
# specialization-magnitude map chrome.
CAPITAL_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#fcfcfb"),
    (0.5, "#e8a33d"),
    (1.0, "#e34948"),
]
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"

EXTREME_COUNTY_LABEL_COUNT: int = 5

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_capital_composition_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by capital-to-wage ratio.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `capital_to_wage_ratio`.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="capital_to_wage_ratio",
        hover_name="county_name",
        hover_data={"capital_to_wage_ratio": ":.3f", "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=CAPITAL_COLORSCALE,
        range_color=(0, df["capital_to_wage_ratio"].quantile(0.98)),
        title="Source E: capital-to-wage ratio (net cap. gains + qual. dividends / wages), Tax Year 2022",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title="Capital/wage ratio"))

    extremes = df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "capital_to_wage_ratio")
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
    """Run the Source E map visualization pipeline."""
    configure_logging()

    source_e_df = load_source_e(SOURCE_E_PARQUET_PATH)

    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = source_e_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    fig = build_capital_composition_map(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
