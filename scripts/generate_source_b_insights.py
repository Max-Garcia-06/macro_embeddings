"""Persist and visualize the statistics behind Source B's EDA scripts.

Mirrors `generate_source_f_insights.py`'s role: re-invokes the existing,
side-effect-free functions from `analyze_source_b_industry_mix.py` and
`analyze_source_b_source_c_correlation.py`, writes the headline numbers to
`analysis-output/source-b/source_b_stats.json`, and renders three static summary
figures from that same data.

Output: `analysis-output/source-b/source_b_stats.json`, `analysis-output/source-b/figures/
source-b-figure-*.png`, `analysis-output/source-b/figures/source-b-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_source_b_industry_mix import (
    NAICS2_COLORS,
    SOURCE_B_PARQUET_PATH,
    compute_dominant_sector,
    load_source_b,
    summarize_industry_mix,
)
from analyze_source_b_source_c_correlation import (
    MIN_SECTOR_DISCLOSED_COUNT,
    build_crossvalidation_table,
    summarize_sector_correlations,
    summarize_specialization_correlation,
    top_reliable_sector_correlations,
)
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output" / "source-b"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "source_b_stats.json"

# Dataviz reference palette (matches the existing EDA scripts' chart chrome).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_FILL_COLOR: str = "#2a78d6"
BASELINE_COLOR: str = "#52514e"

TOP_SECTOR_CORRELATION_COUNT: int = 3

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def render_dominant_sector_figure(df: pd.DataFrame) -> None:
    """Render the main figure: dominant-sector bar chart.

    Args:
        df: Source B DataFrame with a `dominant_industry` column.
    """
    order = [label for label in NAICS2_COLORS if label in df["dominant_industry"].unique()]
    counts = df["dominant_industry"].value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(counts.index, counts.values, color=[NAICS2_COLORS[label] for label in counts.index])
    ax.set_ylabel("County count")
    ax.set_title("Source B: dominant industry sector (highest LQ), all counties")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-b-figure-01-dominant-sector.png", dpi=200)
    plt.close(fig)


def render_specialization_figure(df: pd.DataFrame) -> None:
    """Render the supporting figure: specialization-magnitude (dominant LQ) histogram.

    Args:
        df: Source B DataFrame with a `dominant_lq` column.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.hist(df["dominant_lq"].dropna(), bins=30, color=PRIMARY_FILL_COLOR)
    ax.set_xlabel("Dominant-sector LQ")
    ax.set_ylabel("County count")
    ax.set_title("Source B: industrial specialization magnitude, all counties")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-b-figure-02-specialization-distribution.png", dpi=200)
    plt.close(fig)


def render_crossvalidation_figure(merged: pd.DataFrame) -> None:
    """Render the supporting figure: specialization quartile vs. size-normalized GDP velocity.

    Args:
        merged: Output of build_crossvalidation_table (Source B x Source C).
    """
    grouped = merged.groupby("dominant_lq_quartile", observed=True)["gdp_velocity_pct"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(grouped.index.astype(str), grouped.values, color=PRIMARY_FILL_COLOR)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xlabel("Dominant-sector LQ quartile (1=least specialized, 4=most)")
    ax.set_ylabel("Mean GDP velocity (%/year)")
    ax.set_title("Source B x C: specialization quartile vs. size-normalized GDP velocity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-b-figure-03-specialization-vs-velocity.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as source_b_stats.json).
    """
    lines = [
        "# Source B Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total counties | {stats['total_counties']} |",
    ]
    for label, count in stats["dominant_sector_counts"].items():
        rate = stats["dominant_sector_rates"][label]
        lines.append(f"| Dominant sector: {label} | {count} ({rate:.1%}) |")
    lines.extend(
        [
            f"| Dominant LQ mean | {stats['dominant_lq_stats']['mean']:.3f} |",
            f"| Dominant LQ median | {stats['dominant_lq_stats']['median']:.3f} |",
            f"| Dominant LQ max | {stats['dominant_lq_stats']['max']:.3f} |",
            f"| Dominant LQ vs. unemployment velocity (Pearson r, permutation p) | {stats['dominant_lq_vs_unemployment_velocity_corr']:.4f}, {stats['dominant_lq_vs_unemployment_velocity_p']:.4f} |",
            f"| Dominant LQ vs. GDP velocity %, size-normalized (Pearson r, permutation p) | {stats['dominant_lq_vs_gdp_velocity_pct_corr']:.4f}, {stats['dominant_lq_vs_gdp_velocity_pct_p']:.4f} |",
            "",
            f"Top {len(stats['top_sector_correlations'])} of 20 scanned sectors by \\|corr\\| with size-normalized "
            f"GDP velocity (restricted to sectors disclosed in >= {stats['min_sector_disclosed_count']} counties; "
            "q = Benjamini-Hochberg FDR-adjusted p-value across all 20 sectors, since scanning and ranking 20 "
            "correlations inflates the apparent significance of the raw top-ranked p-values):",
            "",
            "| Sector | Pearson r | raw p | FDR q | n counties |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in stats["top_sector_correlations"]:
        lines.append(
            f"| {entry['label']} | {entry['corr']:.4f} | {entry['p']:.4f} | {entry['p_adj']:.4f} | {entry['n']} |"
        )
    (FIGURES_DIR / "source-b-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source B's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    source_b_df = load_source_b(SOURCE_B_PARQUET_PATH)
    source_b_df = source_b_df.join(compute_dominant_sector(source_b_df))
    render_dominant_sector_figure(source_b_df)
    render_specialization_figure(source_b_df)

    industry_mix_summary = summarize_industry_mix(source_b_df)

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_b_df, source_c_df)
    crossvalidation_summary = summarize_specialization_correlation(merged)
    sector_correlations = summarize_sector_correlations(merged)
    render_crossvalidation_figure(merged)

    stats = {
        "total_counties": int(len(source_b_df)),
        "dominant_sector_counts": industry_mix_summary["counts"],
        "dominant_sector_rates": industry_mix_summary["rates"],
        "dominant_lq_stats": industry_mix_summary["dominant_lq_stats"],
        "dominant_lq_vs_unemployment_velocity_corr": crossvalidation_summary[
            "dominant_lq_vs_unemployment_velocity_corr"
        ],
        "dominant_lq_vs_unemployment_velocity_p": crossvalidation_summary["dominant_lq_vs_unemployment_velocity_p"],
        "dominant_lq_vs_gdp_velocity_pct_corr": crossvalidation_summary["dominant_lq_vs_gdp_velocity_pct_corr"],
        "dominant_lq_vs_gdp_velocity_pct_p": crossvalidation_summary["dominant_lq_vs_gdp_velocity_pct_p"],
        "gdp_velocity_pct_by_specialization_quartile": crossvalidation_summary[
            "gdp_velocity_pct_by_specialization_quartile"
        ],
        "min_sector_disclosed_count": MIN_SECTOR_DISCLOSED_COUNT,
        "top_sector_correlations": top_reliable_sector_correlations(sector_correlations, TOP_SECTOR_CORRELATION_COUNT),
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
