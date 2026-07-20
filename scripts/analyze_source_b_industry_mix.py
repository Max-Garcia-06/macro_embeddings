"""Industry-specialization characterization for Source B (BLS QCEW).

The proposal frames Source B as identifying "which industries actually drive
a county's economy at scale-invariant resolution" (`source_b_plan.md`
Context). This script collapses each county's 20 NAICS-2 sector LQs down to
its single dominant sector (highest LQ, analogous to Source F's one-hot
`dominant_industry_label` collapse) plus that sector's LQ magnitude as a
continuous "how specialized" companion signal -- a county with a dominant LQ
of 8.0 is far more singularly concentrated than one whose dominant LQ is
1.2, even if both share the same top sector.

Output: `source_b_industry_mix.csv` (per-county dominant sector + magnitude)
and `source_b_industry_mix.html` (interactive bar chart of dominant-sector
counts across all counties).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from ingest_source_b import NAICS2_CODES

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
SOURCE_B_PARQUET_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "source_b_qcew.parquet"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_b_industry_mix.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_b_industry_mix.html"

LQ_COLUMNS: list[str] = [f"lq_emp_{code}" for code in NAICS2_CODES]

# Standard 2-digit NAICS sector titles for the 20 codes QCEW ships LQs for.
NAICS2_LABELS: dict[str, str] = {
    "11": "Agriculture, Forestry, Fishing & Hunting",
    "21": "Mining, Quarrying, Oil & Gas",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation & Warehousing",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate & Rental & Leasing",
    "54": "Professional, Scientific & Technical Services",
    "55": "Management of Companies & Enterprises",
    "56": "Administrative & Support & Waste Management",
    "61": "Educational Services",
    "62": "Health Care & Social Assistance",
    "71": "Arts, Entertainment & Recreation",
    "72": "Accommodation & Food Services",
    "81": "Other Services (except Public Administration)",
    "99": "Unclassified",
}
NOT_CLASSIFIED_LABEL: str = "Not classified"

# Qualitative palette with enough distinct hues for 20 sectors plus the
# not-classified sentinel.
NAICS2_COLORS: dict[str, str] = {
    label: px.colors.qualitative.Alphabet[i % len(px.colors.qualitative.Alphabet)]
    for i, label in enumerate(NAICS2_LABELS.values())
}
NAICS2_COLORS[NOT_CLASSIFIED_LABEL] = "#8c8b85"

SURFACE_COLOR: str = "#fcfcfb"
GRIDLINE_COLOR: str = "#e1e0d9"

TOP_N_SPECIALIZED: int = 25

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_source_b(parquet_path: Path) -> pd.DataFrame:
    """Load Source B QCEW data.

    Args:
        parquet_path: Path to `source_b_qcew.parquet`.

    Returns:
        DataFrame as written by `ingest_source_b.py`.
    """
    return pd.read_parquet(parquet_path)


def compute_dominant_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Identify each county's single highest-LQ sector.

    Args:
        df: Source B DataFrame with the 20 `lq_emp_{naics2}` columns.

    Returns:
        DataFrame (same index as `df`) with `dominant_naics2` (code, null if
        every sector is null for that county), `dominant_industry` (display
        label, `NOT_CLASSIFIED_LABEL` if null), and `dominant_lq` (float).
    """
    lq = df[LQ_COLUMNS]
    has_any_value = lq.notna().any(axis=1)

    # idxmax(skipna=True) raises ValueError on an all-NA row rather than
    # returning NaN (a handful of counties have every sector suppressed or
    # absent) -- fill with -inf just for the argmax, then mask back to null.
    dominant_col = lq.fillna(float("-inf")).idxmax(axis=1)
    dominant_naics2 = dominant_col.str[len("lq_emp_"):].where(has_any_value)
    dominant_lq = lq.max(axis=1, skipna=True).where(has_any_value)
    dominant_industry = dominant_naics2.map(NAICS2_LABELS).fillna(NOT_CLASSIFIED_LABEL)

    return pd.DataFrame(
        {
            "dominant_naics2": dominant_naics2,
            "dominant_industry": dominant_industry,
            "dominant_lq": dominant_lq,
        },
        index=df.index,
    )


def summarize_industry_mix(df: pd.DataFrame) -> dict:
    """Compute the dominant-sector breakdown and specialization-magnitude stats.

    Args:
        df: DataFrame with `dominant_industry` and `dominant_lq` columns.

    Returns:
        Dict with `counts` (label -> count), `rates` (label -> share of all
        counties), and `dominant_lq_stats` (mean/median/max).
    """
    counts = df["dominant_industry"].value_counts().to_dict()
    total = len(df)
    rates = {label: count / total for label, count in counts.items()}

    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "rates": {str(k): float(v) for k, v in rates.items()},
        "dominant_lq_stats": {
            "mean": float(df["dominant_lq"].mean()),
            "median": float(df["dominant_lq"].median()),
            "max": float(df["dominant_lq"].max()),
        },
    }


def build_dominant_sector_bar_chart(df: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of dominant-sector counts across all counties.

    Args:
        df: DataFrame with a `dominant_industry` column.

    Returns:
        Plotly Figure ready to export.
    """
    order = [label for label in NAICS2_COLORS if label in df["dominant_industry"].unique()]
    counts = df["dominant_industry"].value_counts().reindex(order)

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        color_discrete_map=NAICS2_COLORS,
        labels={"x": "Dominant NAICS-2 sector", "y": "County count"},
        title="Source B: dominant industry sector (highest LQ), all counties",
    )
    fig.update_layout(
        showlegend=False,
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR),
        xaxis=dict(tickangle=30),
    )
    return fig


def main() -> None:
    """Run the Source B industry-mix characterization."""
    configure_logging()

    df = load_source_b(SOURCE_B_PARQUET_PATH)
    dominant = compute_dominant_sector(df)
    df = df.join(dominant)

    df[
        ["county_name", "fips_code", "dominant_naics2", "dominant_industry", "dominant_lq"]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(df), OUTPUT_CSV_PATH)

    summary = summarize_industry_mix(df)
    logger.info("Dominant-industry breakdown:")
    for label, count in summary["counts"].items():
        logger.info("  %-52s %5d (%.1f%%)", label, count, 100 * summary["rates"][label])
    logger.info("Dominant LQ stats: %s", summary["dominant_lq_stats"])

    most_specialized = df.nlargest(TOP_N_SPECIALIZED, "dominant_lq")
    logger.info("Top %d most specialized counties (highest dominant_lq):", TOP_N_SPECIALIZED)
    for _, row in most_specialized.iterrows():
        logger.info("  %-32s dominant_lq=%6.2f  sector=%s", row["county_name"], row["dominant_lq"], row["dominant_industry"])

    fig = build_dominant_sector_bar_chart(df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
