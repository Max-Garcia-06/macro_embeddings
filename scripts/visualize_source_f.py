"""US map visualization for Source F county typology data.

Loads `source_f_usda_typology.parquet`, derives two summary columns not
directly present in the file (a single dominant-industry label collapsed
from the one-hot `industry_dependence_*` columns, and a demographic
"distress count" across the six non-exclusive demographic flags), merges in
county centroids, and plots each county as a bubble on a US map -- one
colored by dominant industry, one by demographic distress count.

Output: `source_f_map_dependence.html`, `source_f_map_distress.html`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SOURCE_F_PARQUET_PATH: Path = REPO_ROOT / "data" / "source_f_usda_typology.parquet"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_DEPENDENCE_HTML_PATH: Path = OUTPUTS_DIR / "source_f_map_dependence.html"
OUTPUT_DISTRESS_HTML_PATH: Path = OUTPUTS_DIR / "source_f_map_distress.html"

# One-hot industry-dependence column -> display label, in the mutually
# exclusive order the ERS documentation presents them.
INDUSTRY_DEPENDENCE_LABELS: dict[str, str] = {
    "industry_dependence_none": "None (nonspecialized)",
    "industry_dependence_farming": "Farming",
    "industry_dependence_mining": "Mining",
    "industry_dependence_manufacturing": "Manufacturing",
    "industry_dependence_government": "Government",
    "industry_dependence_recreation": "Recreation",
}
NOT_CLASSIFIED_LABEL: str = "Not classified"

# The six non-exclusive demographic risk flags.
DEMOGRAPHIC_FLAG_COLUMNS: list[str] = [
    "low_postsecondary_ed",
    "low_employment",
    "population_loss",
    "housing_stress",
    "retirement_destination",
    "persistent_poverty",
]

# Qualitative palette (dataviz reference palette family) for the dependence
# map; "Not classified" gets a muted gray rather than a hue.
DEPENDENCE_COLORS: dict[str, str] = {
    "None (nonspecialized)": "#8c8b85",
    "Farming": "#3d9a5c",
    "Mining": "#a8562e",
    "Manufacturing": "#2a78d6",
    "Government": "#8558c2",
    "Recreation": "#e0a72e",
    NOT_CLASSIFIED_LABEL: "#d6d5cd",
}
# Sequential palette (surface -> accent) for the distress-count map.
DISTRESS_COLORSCALE: list[tuple[float, str]] = [
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


def load_source_f(parquet_path: Path) -> pd.DataFrame:
    """Load Source F county typology data.

    Args:
        parquet_path: Path to `source_f_usda_typology.parquet`.

    Returns:
        DataFrame as written by `ingest_source_f.py`.
    """
    return pd.read_parquet(parquet_path)


def dominant_industry_label(df: pd.DataFrame) -> pd.Series:
    """Collapse the one-hot `industry_dependence_*` columns to a single label.

    The six columns are mutually exclusive by construction (see
    `ingest_source_f.py`'s `_one_hot_industry_dependence`): at most one is
    True per row, and none are True for a county with ERS's not-classified
    sentinel code.

    Args:
        df: Source F DataFrame with the `industry_dependence_*` columns.

    Returns:
        Series of display labels, `NOT_CLASSIFIED_LABEL` where no flag is set.
    """
    label = pd.Series(NOT_CLASSIFIED_LABEL, index=df.index, dtype="object")
    for column, display_name in INDUSTRY_DEPENDENCE_LABELS.items():
        label = label.mask(df[column].fillna(False), display_name)
    return label


def compute_distress_count(df: pd.DataFrame) -> pd.Series:
    """Sum the six non-exclusive demographic risk flags into a 0-6 count.

    Args:
        df: Source F DataFrame with the DEMOGRAPHIC_FLAG_COLUMNS columns.

    Returns:
        Integer Series; nulls in individual flags (ERS not-classified/
        insufficient-data sentinels) are treated as 0 (not counted), not as
        missing the whole row.
    """
    return df[DEMOGRAPHIC_FLAG_COLUMNS].sum(axis=1, skipna=True).astype(int)


def build_dependence_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by dominant industry dependence.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `dependence_label`.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="dependence_label",
        color_discrete_map=DEPENDENCE_COLORS,
        category_orders={"dependence_label": list(DEPENDENCE_COLORS)},
        hover_name="county_name",
        hover_data={"dependence_label": True, "lat": False, "lon": False},
        scope="usa",
        title="Source F: dominant industry dependence (2025 ERS typology)",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(legend_title="Dominant industry")
    return fig


def build_distress_map(df: pd.DataFrame) -> "px.Figure":
    """Build an interactive US bubble map colored by demographic distress count.

    Args:
        df: DataFrame with `county_name`, `lat`, `lon`, `distress_count`.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="distress_count",
        hover_name="county_name",
        hover_data={"distress_count": True, "lat": False, "lon": False},
        scope="usa",
        color_continuous_scale=DISTRESS_COLORSCALE,
        range_color=(0, 6),
        title="Source F: demographic distress count (0-6 flags, 2025 ERS typology)",
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color=SURFACE_COLOR)))
    fig.update_layout(coloraxis_colorbar=dict(title="Flags"))

    extremes = df.nlargest(EXTREME_COUNTY_LABEL_COUNT, "distress_count")
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
    """Run the Source F map visualization pipeline."""
    configure_logging()

    source_f_df = load_source_f(SOURCE_F_PARQUET_PATH)
    source_f_df["dependence_label"] = dominant_industry_label(source_f_df)
    source_f_df["distress_count"] = compute_distress_count(source_f_df)

    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = source_f_df.merge(centroids_df, on="fips_code", how="left")
    unmatched = merged[merged["lat"].isna()]
    if not unmatched.empty:
        logger.warning(
            "Dropping %d county(ies) with no centroid match: %s",
            len(unmatched),
            ", ".join(unmatched["county_name"]),
        )
    merged = merged.dropna(subset=["lat", "lon"])

    dependence_fig = build_dependence_map(merged)
    dependence_fig.write_html(OUTPUT_DEPENDENCE_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_DEPENDENCE_HTML_PATH)

    distress_fig = build_distress_map(merged)
    distress_fig.write_html(OUTPUT_DISTRESS_HTML_PATH)
    logger.info("Wrote map to %s", OUTPUT_DISTRESS_HTML_PATH)


if __name__ == "__main__":
    main()
