"""Round-2 figures for Source E: tier economics, vintage, and stability.

`generate_source_e_insights.py` renders the round-1 story -- the ratio's
distribution and its Source C crossvalidation. The three findings that came out
of round 2 have no figure at all, and they are the ones a reader most needs
seen rather than described (`source-e-findings.md` §10-§12):

- how far apart the four data-volume tiers are economically,
- how much of the ratio's *level* is the market year rather than the county,
- and that the round-1 noise story ran in the wrong direction.

Reads the artifacts the two round-2 scripts already wrote, so this script does
no analysis of its own beyond one groupby: `analyze_source_e_tiers.py` for the
tier table and `ingest_source_e.py` for the panel.

Output: `analysis-output/source-e/figures/source-e-figure-0{4,5,6}-*.png`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = REPO_ROOT / "analysis-output" / "source-e"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
TIER_STATS_PATH: Path = OUTPUT_DIR / "source_e_tier_stats.json"
PANEL_PARQUET_PATH: Path = REPO_ROOT / "data" / "source_e_irs_soi_panel.parquet"

# Chart chrome, matching the existing Source E figures. The two categorical
# hues are fixed in this order and never cycled; the pair validates clean on
# CVD separation (worst adjacent dE 27.0 protan) against this surface.
SURFACE_COLOR: str = "#fcfcfb"
PRIMARY_FILL_COLOR: str = "#2a78d6"
SECONDARY_FILL_COLOR: str = "#d97706"
BASELINE_COLOR: str = "#52514e"
GRIDLINE_COLOR: str = "#e1e0d9"

# Short tier labels for axes; the JSON keys carry the full cutpoints.
TIER_AXIS_LABELS: dict[str, str] = {
    "T1 thin (<2.2k returns)": "T1 thin\n<2.2k",
    "T2 small (2.2k-11.7k)": "T2 small\n2.2k-11.7k",
    "T3 mid (11.7k-100k)": "T3 mid\n11.7k-100k",
    "T4 large (>=100k)": "T4 large\n>=100k",
}

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive grid and spine treatment.

    Args:
        ax: Axes to restyle in place.
    """
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(axis="y", color=GRIDLINE_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_tier_table(path: Path) -> pd.DataFrame:
    """Load the tier statistics into a frame ordered T1 through T4.

    Args:
        path: Path to `source_e_tier_stats.json`.

    Returns:
        DataFrame indexed by tier label, in tier order.
    """
    tiers = json.loads(path.read_text())["tiers"]
    return pd.DataFrame(tiers).T.reindex(TIER_AXIS_LABELS.keys())


def render_tier_economics_figure(tiers: pd.DataFrame) -> None:
    """Figure 4: county count against economic weight, per tier.

    The asymmetry this exists to show: T1 and T4 are each about a tenth of all
    counties, and hold 0.14% and 82.6% of national investment income.

    Args:
        tiers: Frame from load_tier_table.
    """
    labels = [TIER_AXIS_LABELS[name] for name in tiers.index]
    positions = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE_COLOR)
    _style(ax)
    counties = 100 * tiers["share_of_counties"].astype(float)
    income = 100 * tiers["share_of_investment_income"].astype(float)
    ax.bar(positions - width / 2, counties, width, color=PRIMARY_FILL_COLOR, label="Share of counties")
    ax.bar(
        positions + width / 2, income, width,
        color=SECONDARY_FILL_COLOR, label="Share of national investment income",
    )

    for x, value in zip(positions - width / 2, counties, strict=True):
        ax.text(x, value + 1.5, f"{value:.1f}%", ha="center", color=BASELINE_COLOR, fontsize=9)
    for x, value in zip(positions + width / 2, income, strict=True):
        ax.text(x, value + 1.5, f"{value:.2f}%", ha="center", color=BASELINE_COLOR, fontsize=9)

    ax.set_xticks(positions, labels)
    ax.set_ylabel("Percent of national total")
    ax.set_ylim(0, 95)
    ax.set_title("Source E: a tenth of counties hold 0.14% of the money, another tenth hold 83%")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-04-tier-economics.png", dpi=200)
    plt.close(fig)


def render_vintage_figure(panel: pd.DataFrame) -> None:
    """Figure 5: the ratio's level is set by the market year.

    Both series are capital-to-wage ratios on one axis -- the national
    aggregate (total investment income over total wages) and the unweighted
    mean across counties. The gap between them is the county-equal-weighting
    effect; the shape of both is the equity market.

    Args:
        panel: Long-format Source E panel.
    """
    by_year = panel.groupby("tax_year").agg(
        national=("national_capital_to_wage_ratio", "first"),
        county_mean=("capital_to_wage_ratio", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE_COLOR)
    _style(ax)
    ax.plot(
        by_year.index, by_year["national"], color=PRIMARY_FILL_COLOR, linewidth=2,
        marker="o", markersize=8, label="National aggregate",
    )
    ax.plot(
        by_year.index, by_year["county_mean"], color=SECONDARY_FILL_COLOR, linewidth=2,
        marker="o", markersize=8, label="Unweighted county mean",
    )
    for year, value in by_year["national"].items():
        ax.annotate(f"{value:.3f}", (year, value), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=BASELINE_COLOR, fontsize=9)

    ax.set_xticks(by_year.index)
    ax.set_xlabel("Tax year")
    ax.set_ylabel("Capital-to-wage ratio")
    ax.set_ylim(0, 0.30)
    ax.set_title("Source E: TY2021 runs 64% above TY2022 nationally -- the market, not the counties")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-05-vintage-effect.png", dpi=200)
    plt.close(fig)


def render_tier_behavior_figure(tiers: pd.DataFrame) -> None:
    """Figure 6: the two round-1 corrections, side by side.

    Left: year-over-year movement rises with county size, so the round-1
    advice to weight by `num_returns` upweights the least stable counties.
    Right: the strongest surviving cross-pillar link is absent in T1.

    Args:
        tiers: Frame from load_tier_table.
    """
    labels = [TIER_AXIS_LABELS[name] for name in tiers.index]
    positions = np.arange(len(labels))
    moves = tiers["median_relative_move"].astype(float)
    correlations = tiers["source_b_real_estate_lq_pearson_r"].astype(float)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE_COLOR)

    _style(left)
    left.bar(positions, moves, color=PRIMARY_FILL_COLOR, width=0.6)
    for x, value in zip(positions, moves, strict=True):
        left.text(x, value + 0.008, f"{value:.3f}", ha="center", color=BASELINE_COLOR, fontsize=9)
    left.set_xticks(positions, labels)
    left.set_ylabel("Median |change| / ratio, TY2021 to TY2022")
    left.set_ylim(0, 0.47)
    left.set_title("Bigger counties move more, not less")

    _style(right)
    right.bar(positions, correlations, color=SECONDARY_FILL_COLOR, width=0.6)
    right.axhline(0, color=BASELINE_COLOR, linewidth=1)
    for x, value in zip(positions, correlations, strict=True):
        offset = 0.02 if value >= 0 else -0.05
        right.text(x, value + offset, f"{value:+.3f}", ha="center", color=BASELINE_COLOR, fontsize=9)
    right.set_xticks(positions, labels)
    right.set_ylabel("Pearson r, B Real Estate LQ vs E ratio")
    right.set_ylim(-0.14, 0.58)
    right.set_title("The B x E link does not exist in the thin tier")

    fig.suptitle("Source E by data-volume tier: both round-1 conclusions ran backwards", y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source-e-figure-06-tier-behavior.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Render the three round-2 Source E figures."""
    configure_logging()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tiers = load_tier_table(TIER_STATS_PATH)
    panel = pd.read_parquet(PANEL_PARQUET_PATH)

    render_tier_economics_figure(tiers)
    render_vintage_figure(panel)
    render_tier_behavior_figure(tiers)
    logger.info("Wrote round-2 figures to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
