"""Persist and visualize the statistics behind Source D's EDA scripts.

Mirrors `generate_source_f_insights.py`'s role: re-invokes the existing,
side-effect-free functions from `analyze_source_d_hubs.py` and
`analyze_source_d_source_c_correlation.py`, writes the headline numbers to
`analysis-output/source-d/source_d_stats.json`, and renders three static summary
figures from that same data.

Output: `analysis-output/source-d/source_d_stats.json`, `analysis-output/source-d/figures/
source-d-figure-*.png`, `analysis-output/source-d/figures/source-d-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_source_d_hubs import SOURCE_D_PARQUET_PATH, TOP_N_HUBS, add_hub_signals, load_source_d, rank_top_hubs
from analyze_source_d_source_c_correlation import build_crossvalidation_table, summarize_crossvalidation
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "source_d_stats.json"

TOP_HUB_BAR_COUNT: int = 15

# Dataviz reference palette (matches the existing EDA scripts' chart chrome).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
PRIMARY_FILL_COLOR: str = "#2a78d6"
ACCENT_FILL_COLOR: str = "#e34948"
BASELINE_COLOR: str = "#52514e"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def render_top_hubs_figure(top_hubs: pd.DataFrame) -> None:
    """Render the main figure: top hub counties by total tonnage.

    Args:
        top_hubs: Output of rank_top_hubs.
    """
    plot_df = top_hubs.head(TOP_HUB_BAR_COUNT).iloc[::-1]
    labels = [name.split(",")[0] for name in plot_df["county_name"]]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.barh(labels, plot_df["total_tons"], color=PRIMARY_FILL_COLOR)
    ax.set_xlabel("Total 2022 freight tonnage")
    ax.set_title(f"Source D: top {TOP_HUB_BAR_COUNT} counties by freight tonnage")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-d-figure-01-top-hubs.png", dpi=200)
    plt.close(fig)


def render_tons_vs_concentration_figure(df: pd.DataFrame) -> None:
    """Render the supporting figure: log tonnage vs. partner concentration scatter.

    Args:
        df: Output of add_hub_signals, all counties.
    """
    log_tons = np.log10(df["total_tons"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.scatter(log_tons, df["mean_partner_hhi"], color=PRIMARY_FILL_COLOR, alpha=0.3, s=10)
    ax.set_xlabel("log10(total tons)")
    ax.set_ylabel("Mean partner concentration (HHI)")
    ax.set_title("Source D: trade volume vs. partner concentration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-d-figure-02-tons-vs-concentration.png", dpi=200)
    plt.close(fig)


def render_crossvalidation_figure(merged: pd.DataFrame) -> None:
    """Render the supporting figure: tonnage quartile vs. size-normalized GDP velocity.

    Args:
        merged: Output of build_crossvalidation_table (Source D x Source C).
    """
    grouped = merged.groupby("tons_quartile", observed=True)["gdp_velocity_pct"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(grouped.index.astype(str), grouped.values, color=ACCENT_FILL_COLOR)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xlabel("Total tonnage quartile (1=lowest, 4=highest)")
    ax.set_ylabel("Mean GDP velocity (%/year)")
    ax.set_title("Source D x C: tonnage quartile vs. size-normalized GDP velocity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-d-figure-03-tons-vs-velocity.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as source_d_stats.json).
    """
    lines = [
        "# Source D Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total counties | {stats['total_counties']} |",
        f"| Top hub county (by tonnage) | {stats['top_hub_county']} ({stats['top_hub_tons']:.0f} tons) |",
        f"| log10(tons) vs. mean partner HHI (Pearson r) | {stats['log_tons_vs_hhi_corr']:.4f} |",
        f"| log(tons) vs. unemployment velocity (Pearson r, permutation p) | {stats['log_tons_vs_unemployment_velocity_corr']:.4f}, {stats['log_tons_vs_unemployment_velocity_p']:.4f} |",
        f"| log(tons) vs. GDP velocity %, size-normalized (Pearson r, permutation p) | {stats['log_tons_vs_gdp_velocity_pct_corr']:.4f}, {stats['log_tons_vs_gdp_velocity_pct_p']:.4f} |",
        f"| Partner HHI vs. GDP velocity %, size-normalized (Pearson r, permutation p) | {stats['hhi_vs_gdp_velocity_pct_corr']:.4f}, {stats['hhi_vs_gdp_velocity_pct_p']:.4f} |",
    ]
    (FIGURES_DIR / "source-d-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source D's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    source_d_df = add_hub_signals(load_source_d(SOURCE_D_PARQUET_PATH))
    top_hubs = rank_top_hubs(source_d_df, TOP_N_HUBS)
    render_top_hubs_figure(top_hubs)
    render_tons_vs_concentration_figure(source_d_df)

    log_tons = np.log10(source_d_df["total_tons"].clip(lower=1))
    log_tons_vs_hhi_corr = float(log_tons.corr(source_d_df["mean_partner_hhi"]))

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_d_df, source_c_df)
    crossvalidation_summary = summarize_crossvalidation(merged)
    render_crossvalidation_figure(merged)

    top_hub = top_hubs.iloc[0]
    stats = {
        "total_counties": int(len(source_d_df)),
        "top_hub_county": top_hub["county_name"],
        "top_hub_tons": float(top_hub["total_tons"]),
        "log_tons_vs_hhi_corr": log_tons_vs_hhi_corr,
        "log_tons_vs_unemployment_velocity_corr": crossvalidation_summary["log_tons_vs_unemployment_velocity_corr"],
        "log_tons_vs_unemployment_velocity_p": crossvalidation_summary["log_tons_vs_unemployment_velocity_p"],
        "log_tons_vs_gdp_velocity_pct_corr": crossvalidation_summary["log_tons_vs_gdp_velocity_pct_corr"],
        "log_tons_vs_gdp_velocity_pct_p": crossvalidation_summary["log_tons_vs_gdp_velocity_pct_p"],
        "hhi_vs_unemployment_velocity_corr": crossvalidation_summary["hhi_vs_unemployment_velocity_corr"],
        "hhi_vs_unemployment_velocity_p": crossvalidation_summary["hhi_vs_unemployment_velocity_p"],
        "hhi_vs_gdp_velocity_pct_corr": crossvalidation_summary["hhi_vs_gdp_velocity_pct_corr"],
        "hhi_vs_gdp_velocity_pct_p": crossvalidation_summary["hhi_vs_gdp_velocity_pct_p"],
        "gdp_velocity_pct_by_tons_quartile": crossvalidation_summary["gdp_velocity_pct_by_tons_quartile"],
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
