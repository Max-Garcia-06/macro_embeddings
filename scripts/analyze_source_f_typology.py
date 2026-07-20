"""Industry-dependence and demographic-distress characterization for Source F.

Two questions the proposal's Source F framing (`E_macro_extendedProposal.pdf`,
"mutually exclusive economic dependencies: Farming-dependent,
Mining-dependent, ...") doesn't answer on its own: how much of the country
actually falls into one of those five dependent categories versus none of
them, and does that split track the metro/nonmetro split the way the
underlying ERS methodology (nonmetro-focused economic typology) implies.

Output: `source_f_typology_breakdown.csv` (per-county label + distress
count) and `source_f_typology.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from visualize_source_f import (
    DEPENDENCE_COLORS,
    NOT_CLASSIFIED_LABEL,
    SOURCE_F_PARQUET_PATH,
    compute_distress_count,
    dominant_industry_label,
    load_source_f,
)

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_f_typology_breakdown.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_f_typology.html"

SURFACE_COLOR: str = "#fcfcfb"
GRIDLINE_COLOR: str = "#e1e0d9"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def summarize_industry_dependence(df: pd.DataFrame) -> dict:
    """Compute the dominant-industry category breakdown, overall and by metro status.

    Args:
        df: DataFrame with `dependence_label` and `metro_2023` columns.

    Returns:
        Dict with `counts` (label -> count), `rates` (label -> share of all
        counties), and `none_rate_by_metro` (metro status -> share of that
        group classified `NOT_CLASSIFIED_LABEL`-excluded "None" dependence).
    """
    counts = df["dependence_label"].value_counts().to_dict()
    total = len(df)
    rates = {label: count / total for label, count in counts.items()}

    classified = df[df["dependence_label"] != NOT_CLASSIFIED_LABEL]
    none_rate_by_metro = (
        classified.groupby("metro_2023")["dependence_label"]
        .apply(lambda s: (s == "None (nonspecialized)").mean())
        .to_dict()
    )

    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "rates": {str(k): float(v) for k, v in rates.items()},
        "none_rate_by_metro": {str(k): float(v) for k, v in none_rate_by_metro.items()},
    }


def summarize_demographic_distress(df: pd.DataFrame) -> dict:
    """Compute the demographic distress-count distribution, overall and by metro status.

    Args:
        df: DataFrame with `distress_count` and `metro_2023` columns.

    Returns:
        Dict with `distribution` (count -> number of counties),
        `mean_by_metro` (metro status -> mean distress count).
    """
    distribution = df["distress_count"].value_counts().sort_index().to_dict()
    mean_by_metro = df.groupby("metro_2023")["distress_count"].mean().to_dict()

    return {
        "distribution": {str(k): int(v) for k, v in distribution.items()},
        "mean_by_metro": {str(k): float(v) for k, v in mean_by_metro.items()},
    }


def build_dependence_bar_chart(df: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of dominant-industry category counts.

    Args:
        df: DataFrame with a `dependence_label` column.

    Returns:
        Plotly Figure ready to export.
    """
    order = [label for label in DEPENDENCE_COLORS if label in df["dependence_label"].unique()]
    counts = df["dependence_label"].value_counts().reindex(order)

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=DEPENDENCE_COLORS,
        labels={"x": "Dominant industry", "y": "County count"},
        title="Source F: dominant industry dependence, all counties",
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR),
    )
    return fig


def main() -> None:
    """Run the industry-dependence / demographic-distress characterization."""
    configure_logging()

    df = load_source_f(SOURCE_F_PARQUET_PATH)
    df["dependence_label"] = dominant_industry_label(df)
    df["distress_count"] = compute_distress_count(df)

    df[
        ["county_name", "fips_code", "metro_2023", "dependence_label", "distress_count"]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(df), OUTPUT_CSV_PATH)

    dependence_summary = summarize_industry_dependence(df)
    logger.info("Industry dependence breakdown:")
    for label, count in dependence_summary["counts"].items():
        logger.info("  %-24s %5d (%.1f%%)", label, count, 100 * dependence_summary["rates"][label])

    distress_summary = summarize_demographic_distress(df)
    logger.info("Mean distress count by metro status: %s", distress_summary["mean_by_metro"])

    fig = build_dependence_bar_chart(df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
