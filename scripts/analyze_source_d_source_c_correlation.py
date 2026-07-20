"""Cross-validate Source D's trade flow intensity against Source C's economic velocity.

Analogous to `analyze_source_f_source_c_crossvalidation.py`: does a county's
static trade-flow character (2022 freight tonnage, partner concentration)
track its short-term economic momentum (unemployment/GDP velocity)? Reuses
Source C's own GDP-velocity-as-percentage fix (`source-c-findings.md` SS5) to
avoid the raw absolute-dollar velocity column's economy-size confound.
Tonnage is log-transformed before correlating, since it spans several orders
of magnitude (Loving County, TX: ~9.8K tons vs. LA County: ~256K tons).

Output: `source_d_source_c_crossvalidation.csv` (per-county merged table)
and `source_d_source_c_crossvalidation.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from analyze_source_d_hubs import SOURCE_D_PARQUET_PATH, add_hub_signals, load_source_d
from stats_utils import permutation_test_corr
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_d_source_c_crossvalidation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_d_source_c_crossvalidation.html"

TONNAGE_QUARTILE_COUNT: int = 4

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


def build_crossvalidation_table(source_d: pd.DataFrame, source_c: pd.DataFrame) -> pd.DataFrame:
    """Join Source D trade-flow signals onto Source C velocity data.

    Args:
        source_d: Source D DataFrame with `total_tons`, `mean_partner_hhi`
            (output of `add_hub_signals`).
        source_c: Source C DataFrame (`fips_code`, `gdp_velocity`, `gdp_latest`,
            `unemployment_velocity`).

    Returns:
        Merged DataFrame with `log_total_tons` and `gdp_velocity_pct`
        (= gdp_velocity / gdp_latest, a size-invariant counterpart to Source
        C's absolute-dollar velocity column) added, plus `tons_quartile`
        (1=lowest tonnage, TONNAGE_QUARTILE_COUNT=highest).
    """
    merged = source_d.merge(source_c.drop(columns="county_name"), on="fips_code", how="inner")
    merged["log_total_tons"] = np.log10(merged["total_tons"].clip(lower=1))
    merged["gdp_velocity_pct"] = merged["gdp_velocity"] / merged["gdp_latest"]
    merged["tons_quartile"] = pd.qcut(
        merged["total_tons"], TONNAGE_QUARTILE_COUNT, labels=range(1, TONNAGE_QUARTILE_COUNT + 1)
    )
    return merged


def summarize_crossvalidation(merged: pd.DataFrame) -> dict:
    """Compute correlations (with permutation-test p-values) between trade-flow signals and velocity.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with Pearson correlations (and two-sided permutation p-values,
        `N_PERMUTATIONS` shuffles seeded with `RANDOM_SEED`) of log tonnage
        and partner concentration against both velocity measures, plus mean
        `gdp_velocity_pct` by tonnage quartile.
    """
    log_tons_vs_unemployment_r, log_tons_vs_unemployment_p = permutation_test_corr(
        merged["log_total_tons"], merged["unemployment_velocity"], N_PERMUTATIONS, RANDOM_SEED
    )
    log_tons_vs_gdp_r, log_tons_vs_gdp_p = permutation_test_corr(
        merged["log_total_tons"], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED
    )
    hhi_vs_unemployment_r, hhi_vs_unemployment_p = permutation_test_corr(
        merged["mean_partner_hhi"], merged["unemployment_velocity"], N_PERMUTATIONS, RANDOM_SEED
    )
    hhi_vs_gdp_r, hhi_vs_gdp_p = permutation_test_corr(
        merged["mean_partner_hhi"], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "log_tons_vs_unemployment_velocity_corr": log_tons_vs_unemployment_r,
        "log_tons_vs_unemployment_velocity_p": log_tons_vs_unemployment_p,
        "log_tons_vs_gdp_velocity_pct_corr": log_tons_vs_gdp_r,
        "log_tons_vs_gdp_velocity_pct_p": log_tons_vs_gdp_p,
        "hhi_vs_unemployment_velocity_corr": hhi_vs_unemployment_r,
        "hhi_vs_unemployment_velocity_p": hhi_vs_unemployment_p,
        "hhi_vs_gdp_velocity_pct_corr": hhi_vs_gdp_r,
        "hhi_vs_gdp_velocity_pct_p": hhi_vs_gdp_p,
        "gdp_velocity_pct_by_tons_quartile": {
            str(k): float(v) for k, v in merged.groupby("tons_quartile", observed=True)["gdp_velocity_pct"].mean().items()
        },
    }


def build_quartile_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean size-normalized GDP velocity by tonnage quartile.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    grouped = merged.groupby("tons_quartile", observed=True)["gdp_velocity_pct"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="tons_quartile",
        y="gdp_velocity_pct",
        labels={"tons_quartile": "Total tonnage quartile (1=lowest, 4=highest)", "gdp_velocity_pct": "Mean GDP velocity (%/year)"},
        title="Source D x C: freight tonnage quartile vs. size-normalized GDP velocity",
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
    """Run the Source D x Source C cross-validation."""
    configure_logging()

    source_d = add_hub_signals(load_source_d(SOURCE_D_PARQUET_PATH))
    source_c = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_d, source_c)
    dropped = len(source_d) - len(merged.dropna(subset=["gdp_velocity_pct"]))
    if dropped:
        logger.info("%d county(ies) have no gdp_velocity_pct (missing Source C GDP series).", dropped)

    merged[
        [
            "county_name",
            "fips_code",
            "total_tons",
            "mean_partner_hhi",
            "tons_quartile",
            "unemployment_velocity",
            "gdp_velocity_pct",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_crossvalidation(merged)
    logger.info(
        "log(tons) corr: unemployment_velocity r=%.4f (p=%.4f), gdp_velocity_pct r=%.4f (p=%.4f)",
        summary["log_tons_vs_unemployment_velocity_corr"],
        summary["log_tons_vs_unemployment_velocity_p"],
        summary["log_tons_vs_gdp_velocity_pct_corr"],
        summary["log_tons_vs_gdp_velocity_pct_p"],
    )
    logger.info(
        "partner HHI corr: unemployment_velocity r=%.4f (p=%.4f), gdp_velocity_pct r=%.4f (p=%.4f)",
        summary["hhi_vs_unemployment_velocity_corr"],
        summary["hhi_vs_unemployment_velocity_p"],
        summary["hhi_vs_gdp_velocity_pct_corr"],
        summary["hhi_vs_gdp_velocity_pct_p"],
    )

    fig = build_quartile_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
