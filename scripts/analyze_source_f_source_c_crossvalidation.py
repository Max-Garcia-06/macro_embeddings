"""Cross-validate Source F's demographic distress flags against Source C's velocity.

Source C's own findings (`analysis-output/source-c-findings.md` SS7) flag
that no cross-source validation had been done yet -- this is the first: does
a county's static/structural demographic distress classification (population
loss, persistent poverty, etc.) actually track its short-term economic
momentum (unemployment/GDP velocity)? Also re-derives Source C's own
GDP-velocity-as-percentage fix (findings SS5) rather than using the raw
absolute-dollar column directly, since that column is confounded with
economy size (metro counties have both larger economies and, per
`analyze_source_f_typology.py`, systematically lower distress counts).

Output: `source_f_source_c_crossvalidation.csv` (per-county merged table)
and `source_f_source_c_crossvalidation.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from stats_utils import permutation_test_corr
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c
from visualize_source_f import (
    DEMOGRAPHIC_FLAG_COLUMNS,
    SOURCE_F_PARQUET_PATH,
    compute_distress_count,
    load_source_f,
)

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_f_source_c_crossvalidation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_f_source_c_crossvalidation.html"

# Matches Source A's Mantel-test protocol (analyze_source_a_clusters.py) so
# the two rounds' significance tests are directly comparable.
RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 499

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


def build_crossvalidation_table(source_f: pd.DataFrame, source_c: pd.DataFrame) -> pd.DataFrame:
    """Join Source F typology flags onto Source C velocity data.

    Args:
        source_f: Source F DataFrame (`fips_code`, demographic flags, `metro_2023`).
        source_c: Source C DataFrame (`fips_code`, `gdp_velocity`, `gdp_latest`,
            `unemployment_velocity`).

    Returns:
        Merged DataFrame with an added `distress_count` (0-6) and
        `gdp_velocity_pct` (= gdp_velocity / gdp_latest, a size-invariant
        counterpart to Source C's absolute-dollar velocity column).
    """
    merged = source_f.merge(
        source_c.drop(columns="county_name"), on="fips_code", how="inner"
    )
    merged["distress_count"] = compute_distress_count(merged)
    merged["gdp_velocity_pct"] = merged["gdp_velocity"] / merged["gdp_latest"]
    return merged


def summarize_crossvalidation(merged: pd.DataFrame) -> dict:
    """Compute correlations (with permutation-test p-values) and group means between distress flags and velocity.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with `distress_vs_unemployment_velocity_corr`/`_p`,
        `distress_vs_gdp_velocity_pct_corr`/`_p` (two-sided permutation
        p-values, `N_PERMUTATIONS` shuffles seeded with `RANDOM_SEED`),
        per-flag group means for `gdp_velocity_pct`, and
        `gdp_velocity_pct_by_metro`.
    """
    per_flag_means = {
        flag: merged.groupby(flag)["gdp_velocity_pct"].mean().to_dict()
        for flag in DEMOGRAPHIC_FLAG_COLUMNS
    }
    distress_vs_unemployment_r, distress_vs_unemployment_p = permutation_test_corr(
        merged["distress_count"], merged["unemployment_velocity"], N_PERMUTATIONS, RANDOM_SEED
    )
    distress_vs_gdp_r, distress_vs_gdp_p = permutation_test_corr(
        merged["distress_count"], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "distress_vs_unemployment_velocity_corr": distress_vs_unemployment_r,
        "distress_vs_unemployment_velocity_p": distress_vs_unemployment_p,
        "distress_vs_gdp_velocity_pct_corr": distress_vs_gdp_r,
        "distress_vs_gdp_velocity_pct_p": distress_vs_gdp_p,
        "gdp_velocity_pct_by_flag": {
            flag: {str(k): float(v) for k, v in means.items()}
            for flag, means in per_flag_means.items()
        },
        "gdp_velocity_pct_by_metro": {
            str(k): float(v)
            for k, v in merged.groupby("metro_2023")["gdp_velocity_pct"].mean().items()
        },
    }


def build_distress_velocity_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean size-normalized GDP velocity by distress count.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    grouped = merged.groupby("distress_count")["gdp_velocity_pct"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="distress_count",
        y="gdp_velocity_pct",
        labels={"distress_count": "Demographic distress count (0-6)", "gdp_velocity_pct": "Mean GDP velocity (%/year)"},
        title="Source F x C: demographic distress vs. size-normalized GDP velocity",
    )
    fig.update_traces(marker_color=PRIMARY_FILL_COLOR)
    fig.update_layout(
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR, tickformat=".1%"),
        xaxis=dict(gridcolor=GRIDLINE_COLOR, dtick=1),
    )
    return fig


def main() -> None:
    """Run the Source F x Source C cross-validation."""
    configure_logging()

    source_f = load_source_f(SOURCE_F_PARQUET_PATH)
    source_c = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_f, source_c)
    dropped = len(source_f) - len(merged.dropna(subset=["gdp_velocity_pct"]))
    if dropped:
        logger.info("%d county(ies) have no gdp_velocity_pct (missing Source C GDP series).", dropped)

    merged[
        [
            "county_name",
            "fips_code",
            "metro_2023",
            "distress_count",
            "unemployment_velocity",
            "gdp_velocity_pct",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_crossvalidation(merged)
    logger.info(
        "distress_count corr: unemployment_velocity r=%.4f (p=%.4f), gdp_velocity_pct r=%.4f (p=%.4f)",
        summary["distress_vs_unemployment_velocity_corr"],
        summary["distress_vs_unemployment_velocity_p"],
        summary["distress_vs_gdp_velocity_pct_corr"],
        summary["distress_vs_gdp_velocity_pct_p"],
    )

    fig = build_distress_velocity_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
