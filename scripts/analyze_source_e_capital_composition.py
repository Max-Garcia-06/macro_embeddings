"""Capital-composition characterization for Source E (IRS SOI).

The proposal frames Source E as isolating "asset-rich, investment-driven
markets ... from pure labor-dependent markets" (`source_e_plan.md` Context)
via `capital_to_wage_ratio` = (net capital gains + qualified dividends) /
W-2 wages. This script buckets counties into data-driven quartiles of that
ratio (1=most labor-dependent, 4=most investment-driven), analogous to
Source B's `dominant_lq` and Source D's `tons_quartile` collapses.

Output: `source_e_capital_composition.csv` (per-county ratio + quartile)
and `source_e_capital_composition.html` (interactive histogram).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
SOURCE_E_PARQUET_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "source_e_irs_soi.parquet"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_e_capital_composition.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_e_capital_composition.html"

CAPITAL_QUARTILE_COUNT: int = 4
TOP_N_INVESTMENT_DRIVEN: int = 25

SURFACE_COLOR: str = "#fcfcfb"
GRIDLINE_COLOR: str = "#e1e0d9"
PRIMARY_FILL_COLOR: str = "#2a78d6"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_source_e(parquet_path: Path) -> pd.DataFrame:
    """Load Source E IRS SOI data.

    Args:
        parquet_path: Path to `source_e_irs_soi.parquet`.

    Returns:
        DataFrame as written by `ingest_source_e.py`.
    """
    return pd.read_parquet(parquet_path)


def add_capital_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket counties into data-driven quartiles of `capital_to_wage_ratio`.

    Args:
        df: Source E DataFrame with a `capital_to_wage_ratio` column.

    Returns:
        DataFrame (same index as `df`) with a single `capital_quartile`
        column (1=most labor-dependent, CAPITAL_QUARTILE_COUNT=most
        investment-driven).
    """
    quartile = pd.qcut(
        df["capital_to_wage_ratio"], CAPITAL_QUARTILE_COUNT, labels=range(1, CAPITAL_QUARTILE_COUNT + 1)
    )
    return pd.DataFrame({"capital_quartile": quartile}, index=df.index)


def summarize_capital_composition(df: pd.DataFrame) -> dict:
    """Compute the ratio distribution summary.

    Args:
        df: DataFrame with a `capital_to_wage_ratio` column.

    Returns:
        Dict with mean/median/max/min of the ratio.
    """
    return {
        "mean": float(df["capital_to_wage_ratio"].mean()),
        "median": float(df["capital_to_wage_ratio"].median()),
        "max": float(df["capital_to_wage_ratio"].max()),
        "min": float(df["capital_to_wage_ratio"].min()),
    }


def build_ratio_histogram(df: pd.DataFrame) -> "px.Figure":
    """Build a histogram of `capital_to_wage_ratio` across all counties.

    Args:
        df: DataFrame with a `capital_to_wage_ratio` column.

    Returns:
        Plotly Figure ready to export.
    """
    fig = px.histogram(
        df,
        x="capital_to_wage_ratio",
        nbins=60,
        labels={"capital_to_wage_ratio": "Capital-to-wage ratio (cap. gains + qual. dividends) / wages"},
        title="Source E: capital-to-wage ratio distribution, all counties",
    )
    fig.update_traces(marker_color=PRIMARY_FILL_COLOR)
    fig.update_layout(
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR, title="County count"),
        xaxis=dict(gridcolor=GRIDLINE_COLOR),
    )
    return fig


def main() -> None:
    """Run the Source E capital-composition characterization."""
    configure_logging()

    df = load_source_e(SOURCE_E_PARQUET_PATH)
    df = df.join(add_capital_quartile(df))

    df[
        ["county_name", "fips_code", "capital_to_wage_ratio", "capital_quartile"]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(df), OUTPUT_CSV_PATH)

    summary = summarize_capital_composition(df)
    logger.info("Capital-to-wage ratio stats: %s", summary)

    most_investment_driven = df.nlargest(TOP_N_INVESTMENT_DRIVEN, "capital_to_wage_ratio")
    logger.info("Top %d most investment-driven counties (highest capital_to_wage_ratio):", TOP_N_INVESTMENT_DRIVEN)
    for _, row in most_investment_driven.iterrows():
        logger.info("  %-32s capital_to_wage_ratio=%6.3f", row["county_name"], row["capital_to_wage_ratio"])

    fig = build_ratio_histogram(df)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote histogram to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
