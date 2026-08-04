"""Persist and visualize the statistics behind Source F's EDA scripts.

Mirrors `generate_source_c_insights.py`'s role: re-invokes the existing,
side-effect-free functions from `analyze_source_f_typology.py` and
`analyze_source_f_source_c_crossvalidation.py`, writes the headline numbers
to `analysis-output/source-f/source_f_stats.json`, and renders three static summary
figures from that same data.

Output: `analysis-output/source-f/source_f_stats.json`, `analysis-output/source-f/figures/
source-f-figure-*.png`, `analysis-output/source-f/figures/source-f-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_source_f_source_c_crossvalidation import build_crossvalidation_table, summarize_crossvalidation
from analyze_source_f_typology import summarize_demographic_distress, summarize_industry_dependence
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c
from visualize_source_f import (
    DEPENDENCE_COLORS,
    SOURCE_F_PARQUET_PATH,
    compute_distress_count,
    dominant_industry_label,
    load_source_f,
)

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "source_f_stats.json"

# Dataviz reference palette (matches the existing EDA scripts' chart chrome).
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_INK_COLOR: str = "#0b0b0b"
GRIDLINE_COLOR: str = "#e1e0d9"
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


def render_dependence_figure(df: pd.DataFrame) -> None:
    """Render the main figure: dominant industry dependence bar chart.

    Args:
        df: Source F DataFrame with a `dependence_label` column.
    """
    order = [label for label in DEPENDENCE_COLORS if label in df["dependence_label"].unique()]
    counts = df["dependence_label"].value_counts().reindex(order)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(counts.index, counts.values, color=[DEPENDENCE_COLORS[label] for label in counts.index])
    ax.set_ylabel("County count")
    ax.set_title("Source F: dominant industry dependence, all counties")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-f-figure-01-dependence.png", dpi=200)
    plt.close(fig)


def render_distress_figure(df: pd.DataFrame) -> None:
    """Render the supporting figure: demographic distress-count histogram by metro status.

    Args:
        df: Source F DataFrame with `distress_count` and `metro_2023` columns.
    """
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    bins = range(0, 8)
    ax.hist(
        df.loc[~df["metro_2023"], "distress_count"],
        bins=bins,
        alpha=0.7,
        color=ACCENT_FILL_COLOR,
        label="Nonmetro",
        align="left",
    )
    ax.hist(
        df.loc[df["metro_2023"], "distress_count"],
        bins=bins,
        alpha=0.7,
        color=PRIMARY_FILL_COLOR,
        label="Metro",
        align="left",
    )
    ax.set_xlabel("Demographic distress count (0-6 flags)")
    ax.set_ylabel("County count")
    ax.set_title("Source F: demographic distress count, metro vs. nonmetro")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-f-figure-02-distress-distribution.png", dpi=200)
    plt.close(fig)


def render_crossvalidation_figure(merged: pd.DataFrame) -> None:
    """Render the supporting figure: distress count vs. size-normalized GDP velocity.

    Args:
        merged: Output of build_crossvalidation_table (Source F x Source C).
    """
    grouped = merged.groupby("distress_count")["gdp_velocity_pct"].mean()

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    ax.bar(grouped.index, grouped.values, color=PRIMARY_FILL_COLOR)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1)
    ax.set_xlabel("Demographic distress count (0-6 flags)")
    ax.set_ylabel("Mean GDP velocity (%/year)")
    ax.set_title("Source F x C: distress count vs. size-normalized GDP velocity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-f-figure-03-distress-vs-velocity.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as source_f_stats.json).
    """
    lines = [
        "# Source F Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total counties | {stats['total_counties']} |",
        f"| Not classified (industry dependence sentinel) | {stats['dependence_counts'].get('Not classified', 0)} |",
    ]
    for label, count in stats["dependence_counts"].items():
        rate = stats["dependence_rates"][label]
        lines.append(f"| Dependence: {label} | {count} ({rate:.1%}) |")
    lines.extend(
        [
            f"| Mean distress count, metro | {stats['distress_mean_by_metro'].get('True', 0):.3f} |",
            f"| Mean distress count, nonmetro | {stats['distress_mean_by_metro'].get('False', 0):.3f} |",
            f"| Distress vs. unemployment velocity (Pearson r, permutation p) | {stats['distress_vs_unemployment_velocity_corr']:.4f}, {stats['distress_vs_unemployment_velocity_p']:.4f} |",
            f"| Distress vs. GDP velocity %, size-normalized (Pearson r, permutation p) | {stats['distress_vs_gdp_velocity_pct_corr']:.4f}, {stats['distress_vs_gdp_velocity_pct_p']:.4f} |",
        ]
    )
    (FIGURES_DIR / "source-f-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source F's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    source_f_df = load_source_f(SOURCE_F_PARQUET_PATH)
    source_f_df["dependence_label"] = dominant_industry_label(source_f_df)
    source_f_df["distress_count"] = compute_distress_count(source_f_df)
    render_dependence_figure(source_f_df)
    render_distress_figure(source_f_df)

    dependence_summary = summarize_industry_dependence(source_f_df)
    distress_summary = summarize_demographic_distress(source_f_df)

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    merged = build_crossvalidation_table(source_f_df, source_c_df)
    crossvalidation_summary = summarize_crossvalidation(merged)
    render_crossvalidation_figure(merged)

    stats = {
        "total_counties": int(len(source_f_df)),
        "dependence_counts": dependence_summary["counts"],
        "dependence_rates": dependence_summary["rates"],
        "none_dependence_rate_by_metro": dependence_summary["none_rate_by_metro"],
        "distress_distribution": distress_summary["distribution"],
        "distress_mean_by_metro": distress_summary["mean_by_metro"],
        "distress_vs_unemployment_velocity_corr": crossvalidation_summary[
            "distress_vs_unemployment_velocity_corr"
        ],
        "distress_vs_unemployment_velocity_p": crossvalidation_summary["distress_vs_unemployment_velocity_p"],
        "distress_vs_gdp_velocity_pct_corr": crossvalidation_summary["distress_vs_gdp_velocity_pct_corr"],
        "distress_vs_gdp_velocity_pct_p": crossvalidation_summary["distress_vs_gdp_velocity_pct_p"],
        "gdp_velocity_pct_by_metro": crossvalidation_summary["gdp_velocity_pct_by_metro"],
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
