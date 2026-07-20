"""Persist and visualize the statistics behind Source E's EDA scripts.

Mirrors `generate_source_b_insights.py`'s role: re-invokes the existing,
side-effect-free functions from `analyze_source_e_capital_composition.py` and
`analyze_source_e_source_c_correlation.py`, writes the headline numbers to
`analysis-output/source-e/source_e_stats.json`, and renders three static
summary figures from that same data.

Output: `analysis-output/source-e/source_e_stats.json`, `analysis-output/
source-e/figures/source-e-figure-*.png`, `analysis-output/source-e/figures/
source-e-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_source_e_capital_composition import (
    SOURCE_E_PARQUET_PATH,
    TOP_N_INVESTMENT_DRIVEN,
    add_capital_quartile,
    load_source_e,
    summarize_capital_composition,
)
from analyze_source_e_source_c_correlation import build_crossvalidation_table, summarize_crossvalidation
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output" / "source-e"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "source_e_stats.json"

# Dataviz reference palette (matches the existing EDA scripts' chart chrome).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_FILL_COLOR: str = "#2a78d6"
BASELINE_COLOR: str = "#52514e"

TOP_HUB_BAR_COUNT: int = 15

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def render_top_investment_driven_figure(df: pd.DataFrame) -> None:
    """Render the main figure: top investment-driven counties by ratio.

    Args:
        df: Source E DataFrame with a `capital_to_wage_ratio` column.
    """
    top = df.nlargest(TOP_HUB_BAR_COUNT, "capital_to_wage_ratio").iloc[::-1]
    labels = [name.split(",")[0] for name in top["county_name"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.barh(labels, top["capital_to_wage_ratio"], color=PRIMARY_FILL_COLOR)
    ax.set_xlabel("Capital-to-wage ratio")
    ax.set_title(f"Source E: top {TOP_HUB_BAR_COUNT} most investment-driven counties")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-01-top-investment-driven.png", dpi=200)
    plt.close(fig)


def render_ratio_distribution_figure(df: pd.DataFrame) -> None:
    """Render the supporting figure: capital-to-wage ratio histogram.

    Args:
        df: Source E DataFrame with a `capital_to_wage_ratio` column.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.hist(df["capital_to_wage_ratio"], bins=60, color=PRIMARY_FILL_COLOR)
    ax.set_xlabel("Capital-to-wage ratio")
    ax.set_ylabel("County count")
    ax.set_title("Source E: capital-to-wage ratio distribution, all counties")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-02-ratio-distribution.png", dpi=200)
    plt.close(fig)


def render_crossvalidation_figure(merged: pd.DataFrame) -> None:
    """Render the supporting figure: capital quartile vs. size-normalized GDP velocity.

    Args:
        merged: Output of build_crossvalidation_table (Source E x Source C).
    """
    grouped = merged.groupby("capital_quartile", observed=True)["gdp_velocity_pct"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(grouped.index.astype(str), grouped.values, color=PRIMARY_FILL_COLOR)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xlabel("Capital-to-wage ratio quartile (1=most labor-dependent, 4=most investment-driven)")
    ax.set_ylabel("Mean GDP velocity (%/year)")
    ax.set_title("Source E x C: capital composition quartile vs. size-normalized GDP velocity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-03-composition-vs-velocity.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as source_e_stats.json).
    """
    lines = [
        "# Source E Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total counties | {stats['total_counties']} |",
        f"| Capital-to-wage ratio mean | {stats['ratio_stats']['mean']:.3f} |",
        f"| Capital-to-wage ratio median | {stats['ratio_stats']['median']:.3f} |",
        f"| Capital-to-wage ratio max | {stats['ratio_stats']['max']:.3f} ({stats['top_county']}) |",
        f"| Capital-to-wage ratio vs. unemployment velocity (Pearson r, permutation p) | {stats['capital_to_wage_ratio_vs_unemployment_velocity_corr']:.4f}, {stats['capital_to_wage_ratio_vs_unemployment_velocity_p']:.4f} |",
        f"| Capital-to-wage ratio vs. GDP velocity %, size-normalized (Pearson r, permutation p) | {stats['capital_to_wage_ratio_vs_gdp_velocity_pct_corr']:.4f}, {stats['capital_to_wage_ratio_vs_gdp_velocity_pct_p']:.4f} |",
    ]
    (FIGURES_DIR / "source-e-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source E's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    source_e_df = load_source_e(SOURCE_E_PARQUET_PATH)
    source_e_df = source_e_df.join(add_capital_quartile(source_e_df))
    render_top_investment_driven_figure(source_e_df)
    render_ratio_distribution_figure(source_e_df)

    ratio_summary = summarize_capital_composition(source_e_df)
    top_county = source_e_df.nlargest(1, "capital_to_wage_ratio").iloc[0]["county_name"]

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_e_df, source_c_df)
    crossvalidation_summary = summarize_crossvalidation(merged)
    render_crossvalidation_figure(merged)

    stats = {
        "total_counties": int(len(source_e_df)),
        "ratio_stats": ratio_summary,
        "top_county": top_county,
        "capital_to_wage_ratio_vs_unemployment_velocity_corr": crossvalidation_summary[
            "capital_to_wage_ratio_vs_unemployment_velocity_corr"
        ],
        "capital_to_wage_ratio_vs_unemployment_velocity_p": crossvalidation_summary[
            "capital_to_wage_ratio_vs_unemployment_velocity_p"
        ],
        "capital_to_wage_ratio_vs_gdp_velocity_pct_corr": crossvalidation_summary[
            "capital_to_wage_ratio_vs_gdp_velocity_pct_corr"
        ],
        "capital_to_wage_ratio_vs_gdp_velocity_pct_p": crossvalidation_summary[
            "capital_to_wage_ratio_vs_gdp_velocity_pct_p"
        ],
        "gdp_velocity_pct_by_capital_quartile": crossvalidation_summary["gdp_velocity_pct_by_capital_quartile"],
        "unemployment_velocity_by_capital_quartile": crossvalidation_summary[
            "unemployment_velocity_by_capital_quartile"
        ],
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
