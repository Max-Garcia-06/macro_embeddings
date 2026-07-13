"""Persist and visualize the statistics behind Source C's EDA scripts.

Mirrors `generate_source_a_insights.py`'s role: re-invokes the existing,
side-effect-free functions from `analyze_source_c_quadrants.py`,
`analyze_source_c_similarity.py`, and `analyze_source_c_gdp_coverage.py`,
writes the headline numbers to `analysis-output/source_c_stats.json`, and
renders three static summary figures from that same data.

Output: `analysis-output/source_c_stats.json`, `analysis-output/figures/
source-c-figure-*.png`, `analysis-output/figures/source-c-numeric-summary.md`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_source_c_gdp_coverage import summarize_gdp_coverage
from analyze_source_c_quadrants import QUADRANT_COLORS, classify_quadrants
from analyze_source_c_similarity import build_pairwise_table, rank_diverging_neighbors
from visualize_source_a import CENTROIDS_CACHE_PATH, fetch_county_centroids
from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "analysis-output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
STATS_JSON_PATH: Path = OUTPUT_DIR / "source_c_stats.json"

TOP_N_DIVERGING_PAIRS: int = 20

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


def render_quadrant_figure(classified_df: pd.DataFrame) -> None:
    """Render the main figure: momentum quadrant scatter.

    Args:
        classified_df: Output of classify_quadrants.
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    for quadrant, color in QUADRANT_COLORS.items():
        subset = classified_df[classified_df["quadrant"] == quadrant]
        ax.scatter(
            subset["unemployment_velocity"],
            subset["gdp_velocity"],
            s=8,
            alpha=0.6,
            color=color,
            label=f"{quadrant} (n={len(subset)})",
        )
    ax.axvline(0, color=GRIDLINE_COLOR, linewidth=1)
    ax.axhline(0, color=GRIDLINE_COLOR, linewidth=1)
    ax.set_xlabel("Unemployment velocity (pp/year)")
    ax.set_ylabel("GDP velocity ($k/year)")
    ax.set_title("Source C: economic momentum quadrants")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-c-figure-01-quadrants.png", dpi=200)
    plt.close(fig)


def render_velocity_distribution_figure(df: pd.DataFrame) -> None:
    """Render the supporting figure: unemployment and GDP velocity histograms.

    Args:
        df: Source C DataFrame with `unemployment_velocity` and `gdp_velocity`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE_COLOR)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE_COLOR)

    ax1.hist(
        df["unemployment_velocity"].dropna(), bins=60, color=PRIMARY_FILL_COLOR, edgecolor=SURFACE_COLOR
    )
    ax1.axvline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Unemployment velocity (pp/year)")
    ax1.set_ylabel("County count")
    ax1.set_title("Distribution: unemployment velocity")

    ax2.hist(df["gdp_velocity"].dropna(), bins=60, color=ACCENT_FILL_COLOR, edgecolor=SURFACE_COLOR)
    ax2.axvline(0, color=BASELINE_COLOR, linestyle="--", linewidth=1.5)
    ax2.set_xlabel("GDP velocity ($k/year)")
    ax2.set_title("Distribution: GDP velocity")

    fig.suptitle("Source C: velocity distributions (3-year rolling derivative)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-c-figure-02-velocity-distributions.png", dpi=200)
    plt.close(fig)


def render_proximity_divergence_figure(pairs_df: pd.DataFrame) -> None:
    """Render the supporting figure: geographic distance vs. economic divergence hexbin.

    Args:
        pairs_df: Output of build_pairwise_table (analyze_source_c_similarity).
    """
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    hb = ax.hexbin(
        pairs_df["distance_km"], pairs_df["economic_distance"], gridsize=60, cmap="Blues", mincnt=1
    )
    fig.colorbar(hb, ax=ax, label="Pair count")

    slope, intercept = np.polyfit(pairs_df["distance_km"], pairs_df["economic_distance"], 1)
    trend_x = np.array([pairs_df["distance_km"].min(), pairs_df["distance_km"].max()])
    ax.plot(trend_x, slope * trend_x + intercept, color=BASELINE_COLOR, linestyle="--", linewidth=2)

    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("Economic distance (standardized velocity space)")
    ax.set_title("Source C: economic divergence vs. geographic distance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-c-figure-03-proximity-vs-divergence.png", dpi=200)
    plt.close(fig)


def write_numeric_summary(stats: dict) -> None:
    """Write the single required numeric summary table.

    Args:
        stats: The full stats dict (same content as source_c_stats.json).
    """
    lines = [
        "# Source C Numeric Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total counties | {stats['total_counties']} |",
        f"| Missing GDP series | {stats['missing_gdp_count']} ({stats['missing_gdp_rate']:.1%}) |",
        f"| Missing-GDP counties that are independent cities | {stats['missing_gdp_is_independent_city_rate']:.1%} |",
        f"| Counties classified into quadrants | {stats['quadrant_n_counties']} |",
        f"| Unemployment velocity vs. GDP velocity correlation (Pearson r) | {stats['velocity_correlation']:.4f} |",
    ]
    for quadrant, count in stats["quadrant_counts"].items():
        lines.append(f"| Quadrant: {quadrant} | {count} |")
    lines.append(f"| Corpus mean pairwise geographic distance (km) | {stats['corpus_mean_pairwise_km']:.0f} |")

    (FIGURES_DIR / "source-c-numeric-summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Recompute and persist Source C's EDA statistics, then render figures."""
    configure_logging()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    source_c_df = load_source_c(SOURCE_C_PARQUET_PATH)
    coverage = summarize_gdp_coverage(source_c_df)

    classified_df = classify_quadrants(source_c_df)
    render_quadrant_figure(classified_df)
    render_velocity_distribution_figure(source_c_df)

    velocity_correlation = float(
        classified_df["unemployment_velocity"].corr(classified_df["gdp_velocity"])
    )
    logger.info("Unemployment vs. GDP velocity correlation: r=%.4f", velocity_correlation)

    centroids_df = fetch_county_centroids(CENTROIDS_CACHE_PATH)
    merged = source_c_df.merge(centroids_df, on="fips_code", how="left").dropna(
        subset=["lat", "lon", "unemployment_velocity", "gdp_velocity"]
    )
    pairs_df = build_pairwise_table(merged)
    render_proximity_divergence_figure(pairs_df)
    diverging_df = rank_diverging_neighbors(pairs_df, TOP_N_DIVERGING_PAIRS)

    stats = {
        "total_counties": coverage["total_counties"],
        "missing_gdp_count": coverage["missing_gdp_count"],
        "missing_gdp_rate": coverage["missing_gdp_rate"],
        "missing_gdp_is_independent_city_rate": coverage["missing_gdp_is_independent_city_rate"],
        "missing_by_state": coverage["missing_by_state"],
        "quadrant_n_counties": int(len(classified_df)),
        "quadrant_counts": {
            str(k): int(v) for k, v in classified_df["quadrant"].value_counts().items()
        },
        "velocity_correlation": velocity_correlation,
        "corpus_mean_pairwise_km": float(pairs_df["distance_km"].mean()),
        "top_diverging_pair_economic_distance": float(diverging_df["economic_distance"].iloc[0])
        if len(diverging_df)
        else None,
    }
    STATS_JSON_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote stats to %s", STATS_JSON_PATH)

    write_numeric_summary(stats)
    logger.info("Wrote figures and numeric summary to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
