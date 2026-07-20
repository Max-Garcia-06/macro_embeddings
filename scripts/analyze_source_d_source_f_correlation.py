"""Cross-validate Source D's trade-flow character against Source F's structural typology.

Tests the proposal's stated "Trade Logistics" synergy
(`E_macro_extendedProposal.pdf` SS3.3): does a county's structural economic
typology (industry dependence, demographic distress) explain which counties
become logistics hubs vs. sinks, beyond what tonnage/HHI alone show? Flagged
as a next action in both `source-d-findings.md` SS6 item 1 and
`source-f-findings.md` SS6 item 1, and only actionable now that both pillars
exist -- unlike the D-C and F-C crossvalidations, this is the first
crossvalidation that doesn't route through Source C.

Output: `source_d_source_f_crossvalidation.csv` (per-county merged table) and
`source_d_source_f_crossvalidation.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from analyze_source_d_hubs import SOURCE_D_PARQUET_PATH, add_hub_signals, load_source_d
from stats_utils import permutation_test_corr
from visualize_source_f import (
    DEPENDENCE_COLORS,
    SOURCE_F_PARQUET_PATH,
    compute_distress_count,
    dominant_industry_label,
    load_source_f,
)

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_d_source_f_crossvalidation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_d_source_f_crossvalidation.html"

# Matches Source A's Mantel-test protocol (analyze_source_a_clusters.py) so
# all significance tests across pillars are directly comparable.
RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 499

SURFACE_COLOR: str = "#fcfcfb"
GRIDLINE_COLOR: str = "#e1e0d9"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_crossvalidation_table(source_d: pd.DataFrame, source_f: pd.DataFrame) -> pd.DataFrame:
    """Join Source D trade-flow signals onto Source F typology labels.

    Args:
        source_d: Source D DataFrame with `total_tons`, `mean_partner_hhi`
            (output of `add_hub_signals`).
        source_f: Source F DataFrame with the `industry_dependence_*`
            columns, demographic flag columns, and `metro_2023`.

    Returns:
        Merged DataFrame with `log_total_tons`, `dependence_label`, and
        `distress_count` added. Both pillars cover the full 3,144-county
        crosswalk with no missing values, so this join drops no counties.
    """
    merged = source_d.merge(source_f.drop(columns="county_name"), on="fips_code", how="inner")
    merged["log_total_tons"] = np.log10(merged["total_tons"].clip(lower=1))
    merged["dependence_label"] = dominant_industry_label(merged)
    merged["distress_count"] = compute_distress_count(merged)
    return merged


def summarize_crossvalidation(merged: pd.DataFrame) -> dict:
    """Compute group means and permutation-tested correlations between trade-flow and typology signals.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with mean `log_total_tons`/`mean_partner_hhi` by dependence
        label and metro status, plus Pearson correlations (and two-sided
        permutation p-values, `N_PERMUTATIONS` shuffles seeded with
        `RANDOM_SEED`) of distress count against both trade-flow signals.
    """
    distress_vs_tons_r, distress_vs_tons_p = permutation_test_corr(
        merged["distress_count"], merged["log_total_tons"], N_PERMUTATIONS, RANDOM_SEED
    )
    distress_vs_hhi_r, distress_vs_hhi_p = permutation_test_corr(
        merged["distress_count"], merged["mean_partner_hhi"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "log_tons_by_dependence": {
            str(k): float(v) for k, v in merged.groupby("dependence_label")["log_total_tons"].mean().items()
        },
        "hhi_by_dependence": {
            str(k): float(v) for k, v in merged.groupby("dependence_label")["mean_partner_hhi"].mean().items()
        },
        "log_tons_by_metro": {
            str(k): float(v) for k, v in merged.groupby("metro_2023")["log_total_tons"].mean().items()
        },
        "hhi_by_metro": {
            str(k): float(v) for k, v in merged.groupby("metro_2023")["mean_partner_hhi"].mean().items()
        },
        "distress_vs_log_tons_corr": distress_vs_tons_r,
        "distress_vs_log_tons_p": distress_vs_tons_p,
        "distress_vs_hhi_corr": distress_vs_hhi_r,
        "distress_vs_hhi_p": distress_vs_hhi_p,
    }


def build_dependence_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean log-tonnage by dominant industry dependence.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    order = [label for label in DEPENDENCE_COLORS if label in merged["dependence_label"].unique()]
    grouped = merged.groupby("dependence_label")["log_total_tons"].mean().reindex(order).reset_index()
    fig = px.bar(
        grouped,
        x="dependence_label",
        y="log_total_tons",
        color="dependence_label",
        color_discrete_map=DEPENDENCE_COLORS,
        category_orders={"dependence_label": order},
        labels={"dependence_label": "Dominant industry dependence", "log_total_tons": "Mean log10(total tons)"},
        title="Source D x F: freight tonnage by industry dependence",
    )
    fig.update_layout(
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR),
        xaxis=dict(gridcolor=GRIDLINE_COLOR),
        showlegend=False,
    )
    return fig


def main() -> None:
    """Run the Source D x Source F cross-validation."""
    configure_logging()

    source_d = add_hub_signals(load_source_d(SOURCE_D_PARQUET_PATH))
    source_f = load_source_f(SOURCE_F_PARQUET_PATH)
    merged = build_crossvalidation_table(source_d, source_f)

    merged[
        [
            "county_name",
            "fips_code",
            "total_tons",
            "mean_partner_hhi",
            "dependence_label",
            "distress_count",
            "metro_2023",
        ]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_crossvalidation(merged)
    logger.info("Mean log10(tons) by industry dependence: %s", summary["log_tons_by_dependence"])
    logger.info("Mean partner HHI by industry dependence: %s", summary["hhi_by_dependence"])
    logger.info(
        "distress_count corr: log_total_tons r=%.4f (p=%.4f), mean_partner_hhi r=%.4f (p=%.4f)",
        summary["distress_vs_log_tons_corr"],
        summary["distress_vs_log_tons_p"],
        summary["distress_vs_hhi_corr"],
        summary["distress_vs_hhi_p"],
    )

    fig = build_dependence_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
