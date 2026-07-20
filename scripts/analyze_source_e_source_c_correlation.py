"""Cross-validate Source E's capital composition against Source C's economic velocity.

Analogous to `analyze_source_d_source_c_correlation.py`: does a county's
static wealth composition (investment income vs. wage income) track its
short-term economic momentum (unemployment/GDP velocity)? Reuses Source C's
own GDP-velocity-as-percentage fix (`source-c-findings.md` SS5) to avoid the
raw absolute-dollar velocity column's economy-size confound.

The proposal's own framing for this cross-check (`E_macro_extendedProposal.pdf`
SS1): an investment-heavy county's wealth is "bound to ... global Wall Street
performance" rather than local job survival, so its GDP/unemployment velocity
should track local labor-market swings more loosely than a wage-dependent
county's does.

Output: `source_e_source_c_correlation.csv` (per-county merged table) and
`source_e_source_c_correlation.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from analyze_source_e_capital_composition import CAPITAL_QUARTILE_COUNT, SOURCE_E_PARQUET_PATH, add_capital_quartile, load_source_e
from stats_utils import permutation_test_corr
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_e_source_c_correlation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_e_source_c_correlation.html"

# Matches Source A's Mantel-test protocol (analyze_source_a_clusters.py) so
# every round's significance tests are directly comparable.
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


def build_crossvalidation_table(source_e: pd.DataFrame, source_c: pd.DataFrame) -> pd.DataFrame:
    """Join Source E capital-composition signals onto Source C velocity data.

    Args:
        source_e: Source E DataFrame with `capital_to_wage_ratio` and
            `capital_quartile` (output of `add_capital_quartile`).
        source_c: Source C DataFrame (`fips_code`, `gdp_velocity`,
            `gdp_latest`, `unemployment_velocity`).

    Returns:
        Merged DataFrame with `gdp_velocity_pct` (= gdp_velocity /
        gdp_latest, a size-invariant counterpart to Source C's absolute-
        dollar velocity column) added.
    """
    merged = source_e.merge(source_c.drop(columns="county_name"), on="fips_code", how="inner")
    merged["gdp_velocity_pct"] = merged["gdp_velocity"] / merged["gdp_latest"]
    return merged


def summarize_crossvalidation(merged: pd.DataFrame) -> dict:
    """Compute correlations (with permutation-test p-values) between capital composition and velocity.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with Pearson correlations of `capital_to_wage_ratio` against
        both velocity measures, plus mean `gdp_velocity_pct` and
        `unemployment_velocity` by capital quartile.
    """
    ratio_vs_unemployment_r, ratio_vs_unemployment_p = permutation_test_corr(
        merged["capital_to_wage_ratio"], merged["unemployment_velocity"], N_PERMUTATIONS, RANDOM_SEED
    )
    ratio_vs_gdp_r, ratio_vs_gdp_p = permutation_test_corr(
        merged["capital_to_wage_ratio"], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "capital_to_wage_ratio_vs_unemployment_velocity_corr": ratio_vs_unemployment_r,
        "capital_to_wage_ratio_vs_unemployment_velocity_p": ratio_vs_unemployment_p,
        "capital_to_wage_ratio_vs_gdp_velocity_pct_corr": ratio_vs_gdp_r,
        "capital_to_wage_ratio_vs_gdp_velocity_pct_p": ratio_vs_gdp_p,
        "gdp_velocity_pct_by_capital_quartile": {
            str(k): float(v)
            for k, v in merged.groupby("capital_quartile", observed=True)["gdp_velocity_pct"].mean().items()
        },
        "unemployment_velocity_by_capital_quartile": {
            str(k): float(v)
            for k, v in merged.groupby("capital_quartile", observed=True)["unemployment_velocity"].mean().items()
        },
    }


def build_quartile_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean size-normalized GDP velocity by capital quartile.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    grouped = merged.groupby("capital_quartile", observed=True)["gdp_velocity_pct"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="capital_quartile",
        y="gdp_velocity_pct",
        labels={
            "capital_quartile": "Capital-to-wage ratio quartile (1=most labor-dependent, 4=most investment-driven)",
            "gdp_velocity_pct": "Mean GDP velocity (%/year)",
        },
        title="Source E x C: capital composition quartile vs. size-normalized GDP velocity",
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
    """Run the Source E x Source C cross-validation."""
    configure_logging()

    source_e = load_source_e(SOURCE_E_PARQUET_PATH)
    source_e = source_e.join(add_capital_quartile(source_e))
    source_c = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_e, source_c)
    dropped = len(source_e) - len(merged.dropna(subset=["gdp_velocity_pct"]))
    if dropped:
        logger.info("%d county(ies) have no gdp_velocity_pct (missing Source C GDP series).", dropped)

    merged[
        [
            "county_name",
            "fips_code",
            "capital_to_wage_ratio",
            "capital_quartile",
            "unemployment_velocity",
            "gdp_velocity_pct",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_crossvalidation(merged)
    logger.info(
        "capital_to_wage_ratio corr: unemployment_velocity r=%.4f (p=%.4f), gdp_velocity_pct r=%.4f (p=%.4f)",
        summary["capital_to_wage_ratio_vs_unemployment_velocity_corr"],
        summary["capital_to_wage_ratio_vs_unemployment_velocity_p"],
        summary["capital_to_wage_ratio_vs_gdp_velocity_pct_corr"],
        summary["capital_to_wage_ratio_vs_gdp_velocity_pct_p"],
    )

    fig = build_quartile_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
