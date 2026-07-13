"""Economic-momentum quadrant classification for Source C velocity data.

Classifies each county into one of four momentum quadrants by the sign of
its GDP velocity and unemployment velocity, matching the proposal's own
framing (`E_macro_extendedProposal.pdf` Source C section): a market with
accelerating GDP and falling unemployment reads as fundamentally different
momentum than one with the same static snapshot but decelerating/rising
unemployment.

Counties missing either velocity (no GDP series, see
`analyze_source_c_gdp_coverage.py`) are excluded -- a quadrant needs both axes.

Output: `source_c_quadrants.csv` (per-county quadrant assignment) and
`source_c_quadrants.html` (interactive Plotly scatter).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_c_quadrants.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_c_quadrants.html"

QUADRANT_LABELS: dict[tuple[bool, bool], str] = {
    (True, True): "Accelerating & Tightening",
    (True, False): "Growing but Loosening",
    (False, True): "Slowing but Tightening",
    (False, False): "Decelerating & Loosening",
}

# Categorical palette (dataviz reference palette), one color per quadrant.
QUADRANT_COLORS: dict[str, str] = {
    "Accelerating & Tightening": "#2a78d6",
    "Growing but Loosening": "#e8a33d",
    "Slowing but Tightening": "#6da7ec",
    "Decelerating & Loosening": "#e34948",
}
SURFACE_COLOR: str = "#fcfcfb"
GRIDLINE_COLOR: str = "#e1e0d9"
AXIS_LINE_COLOR: str = "#c3c2b7"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def classify_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each county a momentum quadrant label.

    Args:
        df: DataFrame with `unemployment_velocity` and `gdp_velocity` columns.

    Returns:
        Rows with both velocities present, plus a `quadrant` column.
        "Tightening" means unemployment_velocity <= 0 (falling/flat
        unemployment); "Accelerating" means gdp_velocity >= 0.
    """
    classified = df.dropna(subset=["unemployment_velocity", "gdp_velocity"]).copy()
    gdp_accelerating = classified["gdp_velocity"] >= 0
    unemployment_tightening = classified["unemployment_velocity"] <= 0
    classified["quadrant"] = [
        QUADRANT_LABELS[(accel, tight)]
        for accel, tight in zip(gdp_accelerating, unemployment_tightening)
    ]
    return classified


def build_scatter(df: pd.DataFrame) -> "px.Figure":
    """Build the unemployment-velocity-vs-GDP-velocity scatter, colored by quadrant.

    Args:
        df: Output of classify_quadrants.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter(
        df,
        x="unemployment_velocity",
        y="gdp_velocity",
        color="quadrant",
        color_discrete_map=QUADRANT_COLORS,
        hover_name="county_name",
        hover_data={"unemployment_velocity": ":.3f", "gdp_velocity": ":.1f", "quadrant": False},
        title="Source C: economic momentum quadrants",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.65))
    fig.add_vline(x=0, line=dict(color=AXIS_LINE_COLOR, width=1))
    fig.add_hline(y=0, line=dict(color=AXIS_LINE_COLOR, width=1))
    fig.update_layout(
        xaxis=dict(
            title="Unemployment velocity (pp/year) → rising unemployment",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=AXIS_LINE_COLOR,
        ),
        yaxis=dict(
            title="GDP velocity ($k/year) → accelerating growth",
            gridcolor=GRIDLINE_COLOR,
            zerolinecolor=AXIS_LINE_COLOR,
        ),
        legend_title="Quadrant",
    )
    return fig


def main() -> None:
    """Run the momentum-quadrant classification pipeline."""
    configure_logging()

    df = load_source_c(SOURCE_C_PARQUET_PATH)
    classified = classify_quadrants(df)
    dropped = len(df) - len(classified)
    if dropped:
        logger.info("Excluded %d county(ies) missing a velocity value.", dropped)

    classified[["county_name", "fips_code", "unemployment_velocity", "gdp_velocity", "quadrant"]].to_csv(
        OUTPUT_CSV_PATH, index=False
    )
    logger.info("Wrote %d classified counties to %s", len(classified), OUTPUT_CSV_PATH)

    fig = build_scatter(classified)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote scatter to %s", OUTPUT_HTML_PATH)

    logger.info("Quadrant counts:")
    for quadrant, count in classified["quadrant"].value_counts().items():
        logger.info("  %-28s %5d (%.1f%%)", quadrant, count, 100 * count / len(classified))


if __name__ == "__main__":
    main()
