"""Source E stratified by county data volume.

Every reliability claim made about Source E is really a claim about one of
four groups of counties, and they behave differently enough that a single
national statistic hides all of it: the smallest tier holds 10% of the rows
and 0.1% of the dollars, and the cross-pillar link to Source B that survives
the size control nationally does not exist there at all.

Tier cutpoints are `num_returns`, on round anchors rather than quantiles so
they stay stable across refreshes: the shipped `low_return_flag` threshold,
the national median, and 100k returns.

The tiers are a diagnostic and a serving policy -- which counties a consumer
can trust the ratio on -- not a feature. County size is the open question in
`docs/PROJECT_GOAL.md`, and a tier column in the matrix would answer it by
accident.

Output: `outputs/source_e_tiers.csv` (per-county tier assignment) and
`analysis-output/source-e/source_e_tier_stats.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
STATS_PATH: Path = REPO_ROOT / "analysis-output" / "source-e" / "source_e_tier_stats.json"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_e_tiers.csv"

TIER_EDGES: tuple[float, ...] = (0, 2_200, 11_700, 100_000, np.inf)
TIER_LABELS: tuple[str, ...] = (
    "T1 thin (<2.2k returns)",
    "T2 small (2.2k-11.7k)",
    "T3 mid (11.7k-100k)",
    "T4 large (>=100k)",
)

# Source B's Real Estate & Rental & Leasing location quotient -- the strongest
# cross-pillar link in the sweep that survives the size control (r = 0.394
# raw / 0.382 controlled, analysis-output/cross-source/pillar_pair_stats.json).
REAL_ESTATE_LQ_COLUMN: str = "lq_emp_53"

LARGE_MOVE_THRESHOLD: float = 0.5

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def assign_tiers(latest: pd.DataFrame) -> pd.DataFrame:
    """Attach the data-volume tier to each county.

    Args:
        latest: Latest-year Source E frame, carrying `num_returns`.

    Returns:
        A copy carrying an ordered categorical `data_tier`. Input unchanged.
    """
    return latest.assign(
        data_tier=pd.cut(
            latest["num_returns"], bins=TIER_EDGES, labels=TIER_LABELS, right=False
        )
    )


def year_over_year_stability(panel: pd.DataFrame) -> pd.DataFrame:
    """Measure how much each county's ratio moved between the last two years.

    Args:
        panel: Long-format Source E panel with `tax_year`.

    Returns:
        DataFrame keyed on `fips_code` with `relative_move` (absolute change
        as a share of the earlier year's ratio) and both years' ratios.
    """
    years = sorted(panel["tax_year"].unique())
    previous, latest_year = years[-2], years[-1]
    wide = panel.pivot(index="fips_code", columns="tax_year", values="capital_to_wage_ratio")
    paired = wide[[previous, latest_year]].dropna()
    paired.columns = ["ratio_previous", "ratio_latest"]
    paired["relative_move"] = (
        (paired["ratio_latest"] - paired["ratio_previous"]).abs()
        / paired["ratio_previous"].replace(0, np.nan)
    )
    logger.info("Year-over-year comparison: TY%d vs TY%d", previous, latest_year)
    return paired.reset_index()


def summarize_tier(tier_frame: pd.DataFrame, national: dict[str, float]) -> dict[str, float]:
    """Compute one tier's row of the comparison table.

    Args:
        tier_frame: Counties in a single tier, already joined to the stability
            and Source B columns.
        national: National totals used to express each tier's economic share.

    Returns:
        Flat mapping of statistic name to value.
    """
    investment_income = (
        tier_frame["net_cap_gain_thousands"] + tier_frame["qualified_dividends_thousands"]
    )
    paired = tier_frame.dropna(subset=["relative_move"])
    linked = tier_frame.dropna(subset=[REAL_ESTATE_LQ_COLUMN, "capital_to_wage_ratio"])

    summary = {
        "n_counties": int(len(tier_frame)),
        "share_of_counties": len(tier_frame) / national["n_counties"],
        "share_of_returns": tier_frame["num_returns"].sum() / national["returns"],
        "share_of_investment_income": investment_income.sum() / national["investment_income"],
        "share_of_wages": tier_frame["wages_salaries_thousands"].sum() / national["wages"],
        "median_ratio": tier_frame["capital_to_wage_ratio"].median(),
        "iqr_ratio": (
            tier_frame["capital_to_wage_ratio"].quantile(0.75)
            - tier_frame["capital_to_wage_ratio"].quantile(0.25)
        ),
        "p99_ratio": tier_frame["capital_to_wage_ratio"].quantile(0.99),
        "max_ratio": tier_frame["capital_to_wage_ratio"].max(),
        "median_ratio_normalized_mean": tier_frame["capital_to_wage_ratio_normalized_mean"].median(),
        "median_capgain_participation": tier_frame["capgain_participation_rate"].median(),
        "median_gain_per_claimer": tier_frame["gain_per_claimer_thousands"].median(),
        "median_wage_per_return": tier_frame["wage_per_return_thousands"].median(),
        "median_relative_move": paired["relative_move"].median(),
        "share_moving_over_50pct": float((paired["relative_move"] > LARGE_MOVE_THRESHOLD).mean()),
        "n_thin_claimer_flag": int(tier_frame["thin_claimer_flag"].sum()),
        "n_concentrated_gain_flag": int(tier_frame["concentrated_gain_flag"].sum()),
    }

    if len(paired) > 2:
        summary["rank_stability_spearman"] = float(
            stats.spearmanr(paired["ratio_previous"], paired["ratio_latest"]).statistic
        )
    if len(linked) > 2:
        summary["n_with_source_b"] = int(len(linked))
        summary["source_b_real_estate_lq_pearson_r"] = float(
            stats.pearsonr(linked[REAL_ESTATE_LQ_COLUMN], linked["capital_to_wage_ratio"]).statistic
        )
        summary["source_b_real_estate_lq_spearman"] = float(
            stats.spearmanr(linked[REAL_ESTATE_LQ_COLUMN], linked["capital_to_wage_ratio"]).statistic
        )
    return summary


def build_tier_table(joined: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize every tier against the national totals.

    Args:
        joined: Latest-year frame with tier, stability, and Source B columns.

    Returns:
        Mapping of tier label to its statistics.
    """
    national = {
        "n_counties": len(joined),
        "returns": joined["num_returns"].sum(),
        "investment_income": (
            joined["net_cap_gain_thousands"] + joined["qualified_dividends_thousands"]
        ).sum(),
        "wages": joined["wages_salaries_thousands"].sum(),
    }
    return {
        label: summarize_tier(joined.loc[joined["data_tier"] == label], national)
        for label in TIER_LABELS
    }


def main() -> None:
    """Stratify Source E by data volume and write the tier comparison."""
    configure_logging()

    latest = pd.read_parquet(DATA_DIR / "source_e_irs_soi.parquet")
    panel = pd.read_parquet(DATA_DIR / "source_e_irs_soi_panel.parquet")
    source_b = pd.read_parquet(DATA_DIR / "source_b_qcew.parquet")

    joined = (
        assign_tiers(latest)
        .merge(year_over_year_stability(panel), on="fips_code", how="left")
        .merge(source_b[["fips_code", REAL_ESTATE_LQ_COLUMN]], on="fips_code", how="left")
    )

    tier_table = build_tier_table(joined)
    for label, row in tier_table.items():
        logger.info(
            "%s: n=%d (%.1f%% of counties, %.2f%% of investment income), "
            "median ratio %.3f, median |move| %.3f, B x E r=%.3f",
            label,
            row["n_counties"],
            100 * row["share_of_counties"],
            100 * row["share_of_investment_income"],
            row["median_ratio"],
            row["median_relative_move"],
            row.get("source_b_real_estate_lq_pearson_r", float("nan")),
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joined[
        [
            "county_name",
            "fips_code",
            "data_tier",
            "num_returns",
            "capital_to_wage_ratio",
            "capital_to_wage_ratio_normalized_mean",
            "relative_move",
            "low_return_flag",
            "thin_claimer_flag",
            "concentrated_gain_flag",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %s", OUTPUT_CSV_PATH)

    STATS_PATH.write_text(
        json.dumps(
            {
                "tier_edges": [edge if np.isfinite(edge) else None for edge in TIER_EDGES],
                "tax_years": sorted(int(year) for year in panel["tax_year"].unique()),
                "tiers": tier_table,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s", STATS_PATH)


if __name__ == "__main__":
    main()
