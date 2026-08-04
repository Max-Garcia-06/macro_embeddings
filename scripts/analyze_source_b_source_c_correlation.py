"""Cross-validate Source B's industrial specialization against Source C's economic velocity.

Analogous to `analyze_source_d_source_c_correlation.py`: does a county's
static industrial structure track its short-term economic momentum
(unemployment/GDP velocity)? Reuses Source C's GDP-velocity-as-percentage fix
(`source-c-findings.md` SS5) to avoid the raw absolute-dollar velocity
column's economy-size confound.

Two questions, both drawn from the proposal's framing of Source B
(`docs/plans/ingestion_recon.md (Source B)` Context -- "distinguishes *what kind* of growth or
decline a county is experiencing"): (1) does the *magnitude* of a county's
top-sector specialization (`dominant_lq`) predict velocity, and (2) do
*specific* sectors' LQs predict velocity better than others (e.g. does high
manufacturing LQ track differently than high information/tech LQ)?

Output: `source_b_source_c_correlation.csv` (per-county merged table) and
`source_b_source_c_correlation.html` (interactive bar chart).
"""

from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import plotly.express as px

from analyze_source_b_industry_mix import LQ_COLUMNS, NAICS2_LABELS, SOURCE_B_PARQUET_PATH, compute_dominant_sector, load_source_b
from stats_utils import benjamini_hochberg, permutation_test_corr
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_b_source_c_correlation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_b_source_c_correlation.html"

SPECIALIZATION_QUARTILE_COUNT: int = 4
TOP_SECTOR_CORRELATION_COUNT: int = 3

# Sectors disclosed in fewer counties than this are excluded from the
# top-sector ranking: BLS suppression targets small-establishment-count
# counties non-randomly, so a thinly-disclosed sector's correlation is over
# a size/structure-biased subsample and can post a spuriously large |r|.
MIN_SECTOR_DISCLOSED_COUNT: int = 100

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


def build_crossvalidation_table(source_b: pd.DataFrame, source_c: pd.DataFrame) -> pd.DataFrame:
    """Join Source B industry-mix signals onto Source C velocity data.

    Args:
        source_b: Source B DataFrame with the 20 `lq_emp_{naics2}` columns
            plus `dominant_lq` (output of `compute_dominant_sector`).
        source_c: Source C DataFrame (`fips_code`, `gdp_velocity`,
            `gdp_latest`, `unemployment_velocity`).

    Returns:
        Merged DataFrame with `gdp_velocity_pct` (= gdp_velocity /
        gdp_latest, a size-invariant counterpart to Source C's absolute-
        dollar velocity column) added, plus `dominant_lq_quartile`
        (1=least specialized, SPECIALIZATION_QUARTILE_COUNT=most).
    """
    merged = source_b.merge(source_c.drop(columns="county_name"), on="fips_code", how="inner")
    merged["gdp_velocity_pct"] = merged["gdp_velocity"] / merged["gdp_latest"]
    merged["dominant_lq_quartile"] = pd.qcut(
        merged["dominant_lq"], SPECIALIZATION_QUARTILE_COUNT, labels=range(1, SPECIALIZATION_QUARTILE_COUNT + 1)
    )
    return merged


def summarize_specialization_correlation(merged: pd.DataFrame) -> dict:
    """Compute correlations (with permutation-test p-values) between specialization magnitude and velocity.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with Pearson correlations of `dominant_lq` against both
        velocity measures, plus mean `gdp_velocity_pct` by specialization
        quartile.
    """
    lq_vs_unemployment_r, lq_vs_unemployment_p = permutation_test_corr(
        merged["dominant_lq"], merged["unemployment_velocity"], N_PERMUTATIONS, RANDOM_SEED
    )
    lq_vs_gdp_r, lq_vs_gdp_p = permutation_test_corr(
        merged["dominant_lq"], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "dominant_lq_vs_unemployment_velocity_corr": lq_vs_unemployment_r,
        "dominant_lq_vs_unemployment_velocity_p": lq_vs_unemployment_p,
        "dominant_lq_vs_gdp_velocity_pct_corr": lq_vs_gdp_r,
        "dominant_lq_vs_gdp_velocity_pct_p": lq_vs_gdp_p,
        "gdp_velocity_pct_by_specialization_quartile": {
            str(k): float(v)
            for k, v in merged.groupby("dominant_lq_quartile", observed=True)["gdp_velocity_pct"].mean().items()
        },
    }


def summarize_sector_correlations(merged: pd.DataFrame) -> list[dict]:
    """Correlate each individual NAICS-2 sector's LQ against GDP velocity.

    Answers "does a specific industry mix predict acceleration/deceleration"
    more directly than the single `dominant_lq` magnitude can -- e.g. whether
    a county's Information-sector LQ tracks growth differently than its
    Manufacturing-sector LQ, independent of which sector happens to be each
    county's single dominant one.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        List of all 20 per-sector dicts (`naics2`, `label`, `corr`, `p`, `n`
        the disclosed-county count the correlation was computed over, `p_adj`
        the Benjamini-Hochberg FDR-adjusted q-value across all 20 sectors),
        sorted by descending absolute correlation with `gdp_velocity_pct`.
        Scanning 20 sectors and ranking by |corr| inflates the apparent
        significance of whichever happen to rank highest -- `p_adj` corrects
        for that; `n` flags sectors whose correlation rests on a small,
        suppression-biased subsample (see MIN_SECTOR_DISCLOSED_COUNT).
    """
    results = []
    for column in LQ_COLUMNS:
        naics2 = column[len("lq_emp_"):]
        n = int(merged[[column, "gdp_velocity_pct"]].dropna().shape[0])
        r, p = permutation_test_corr(merged[column], merged["gdp_velocity_pct"], N_PERMUTATIONS, RANDOM_SEED)
        results.append({"naics2": naics2, "label": NAICS2_LABELS[naics2], "corr": r, "p": p, "n": n})

    for entry, q in zip(results, benjamini_hochberg([r["p"] for r in results])):
        entry["p_adj"] = q

    return sorted(results, key=lambda d: abs(d["corr"]), reverse=True)


def top_reliable_sector_correlations(sector_correlations: list[dict], count: int) -> list[dict]:
    """Select the top sectors by |corr|, excluding thinly-disclosed ones.

    Args:
        sector_correlations: Output of summarize_sector_correlations, already
            sorted by descending |corr|.
        count: Number of sectors to return.

    Returns:
        Up to `count` entries with `n >= MIN_SECTOR_DISCLOSED_COUNT`.
    """
    return [entry for entry in sector_correlations if entry["n"] >= MIN_SECTOR_DISCLOSED_COUNT][:count]


def build_quartile_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean size-normalized GDP velocity by specialization quartile.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    grouped = merged.groupby("dominant_lq_quartile", observed=True)["gdp_velocity_pct"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="dominant_lq_quartile",
        y="gdp_velocity_pct",
        labels={
            "dominant_lq_quartile": "Dominant-sector LQ quartile (1=least specialized, 4=most)",
            "gdp_velocity_pct": "Mean GDP velocity (%/year)",
        },
        title="Source B x C: industrial specialization quartile vs. size-normalized GDP velocity",
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
    """Run the Source B x Source C cross-validation."""
    configure_logging()

    source_b = load_source_b(SOURCE_B_PARQUET_PATH)
    source_b = source_b.join(compute_dominant_sector(source_b))
    source_c = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_b, source_c)
    dropped = len(source_b) - len(merged.dropna(subset=["gdp_velocity_pct"]))
    if dropped:
        logger.info("%d county(ies) have no gdp_velocity_pct (missing Source C GDP series).", dropped)

    merged[
        [
            "county_name",
            "fips_code",
            "dominant_naics2",
            "dominant_industry",
            "dominant_lq",
            "dominant_lq_quartile",
            "unemployment_velocity",
            "gdp_velocity_pct",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_specialization_correlation(merged)
    logger.info(
        "dominant_lq corr: unemployment_velocity r=%.4f (p=%.4f), gdp_velocity_pct r=%.4f (p=%.4f)",
        summary["dominant_lq_vs_unemployment_velocity_corr"],
        summary["dominant_lq_vs_unemployment_velocity_p"],
        summary["dominant_lq_vs_gdp_velocity_pct_corr"],
        summary["dominant_lq_vs_gdp_velocity_pct_p"],
    )

    sector_correlations = summarize_sector_correlations(merged)
    top_sectors = top_reliable_sector_correlations(sector_correlations, TOP_SECTOR_CORRELATION_COUNT)
    logger.info(
        "Top %d sectors by |corr| with gdp_velocity_pct (n>=%d, FDR-adjusted q across all 20):",
        TOP_SECTOR_CORRELATION_COUNT, MIN_SECTOR_DISCLOSED_COUNT,
    )
    for entry in top_sectors:
        logger.info(
            "  %-52s r=%.4f (p=%.4f, q=%.4f, n=%d)",
            entry["label"], entry["corr"], entry["p"], entry["p_adj"], entry["n"],
        )

    fig = build_quartile_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
