"""Cross-validate Source A's text richness against Source F's structural typology.

Tests the proposal's stated "Capital Flow" framing of Source F
(`E_macro_extendedProposal.pdf`): "for low-population or hyper-rural counties
where online text data might be sparse or uninformative... [Source F]
provides a solid baseline." Flagged as a next action in
`source-f-findings.md` SS6 item 2 -- is Source A's Wikipedia intro text
actually systematically thinner for counties Source F flags as high-distress
or non-metro? Unlike the drop_stub_counties filter used elsewhere in Source
A's own EDA, stub/thin counties are kept here rather than dropped, since
their text length is exactly the signal under test.

Output: `source_a_source_f_crossvalidation.csv` (per-county merged table) and
`source_a_source_f_crossvalidation.html` (interactive bar chart).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px

from analyze_source_a_clusters import filter_to_fifty_states
from ingest_source_a import strip_boilerplate_phrasing, strip_self_reference
from stats_utils import permutation_test_corr
from visualize_source_a import EMBEDDINGS_PARQUET_PATH, load_embeddings
from visualize_source_f import SOURCE_F_PARQUET_PATH, compute_distress_count, load_source_f

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_source_f_crossvalidation.csv"
OUTPUT_HTML_PATH: Path = OUTPUTS_DIR / "source_a_source_f_crossvalidation.html"

# Matches Source A's Mantel-test protocol (analyze_source_a_clusters.py) so
# all significance tests across pillars are directly comparable.
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


def compute_content_length(df: pd.DataFrame) -> pd.Series:
    """Compute de-boilerplated intro-text length per county, without dropping any rows.

    Args:
        df: DataFrame with `county_name` and `raw_intro_text` columns.

    Returns:
        Integer Series of stripped-text character counts, same index as `df`.
    """
    return df.apply(
        lambda row: len(
            strip_boilerplate_phrasing(strip_self_reference(row["raw_intro_text"], row["county_name"]))
        ),
        axis=1,
    )


def build_crossvalidation_table(source_a: pd.DataFrame, source_f: pd.DataFrame) -> pd.DataFrame:
    """Join Source A text-length signal onto Source F typology labels.

    Args:
        source_a: Source A DataFrame (`fips_code`, `county_name`, `raw_intro_text`).
        source_f: Source F DataFrame with `distress_count`-derivable flags and `metro_2023`.

    Returns:
        Merged DataFrame with `content_length` and `distress_count` added.
        Source A is filtered to the 50 states first (`filter_to_fifty_states`),
        matching every other Source A EDA script's convention.
    """
    source_a = filter_to_fifty_states(source_a).copy()
    source_a["content_length"] = compute_content_length(source_a)
    merged = source_a.merge(source_f.drop(columns="county_name"), on="fips_code", how="inner")
    merged["distress_count"] = compute_distress_count(merged)
    return merged


def summarize_crossvalidation(merged: pd.DataFrame) -> dict:
    """Compute group means and a permutation-tested correlation between text length and typology.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Dict with mean `content_length` by demographic distress count and
        metro status, plus Pearson correlations (and two-sided permutation
        p-values, `N_PERMUTATIONS` shuffles seeded with `RANDOM_SEED`) of
        distress count and metro status (point-biserial) against content
        length.
    """
    distress_vs_length_r, distress_vs_length_p = permutation_test_corr(
        merged["distress_count"], merged["content_length"], N_PERMUTATIONS, RANDOM_SEED
    )
    metro_vs_length_r, metro_vs_length_p = permutation_test_corr(
        merged["metro_2023"].astype(int), merged["content_length"], N_PERMUTATIONS, RANDOM_SEED
    )
    return {
        "content_length_by_distress": {
            str(k): float(v) for k, v in merged.groupby("distress_count")["content_length"].mean().items()
        },
        "content_length_by_metro": {
            str(k): float(v) for k, v in merged.groupby("metro_2023")["content_length"].mean().items()
        },
        "distress_vs_content_length_corr": distress_vs_length_r,
        "distress_vs_content_length_p": distress_vs_length_p,
        "metro_vs_content_length_corr": metro_vs_length_r,
        "metro_vs_content_length_p": metro_vs_length_p,
    }


def build_distress_length_chart(merged: pd.DataFrame) -> "px.Figure":
    """Build a bar chart of mean intro-text length by demographic distress count.

    Args:
        merged: Output of build_crossvalidation_table.

    Returns:
        Plotly Figure ready to export.
    """
    grouped = merged.groupby("distress_count")["content_length"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="distress_count",
        y="content_length",
        labels={"distress_count": "Demographic distress count (0-6)", "content_length": "Mean de-boilerplated intro length (chars)"},
        title="Source A x F: intro-text length vs. demographic distress",
    )
    fig.update_traces(marker_color=PRIMARY_FILL_COLOR)
    fig.update_layout(
        plot_bgcolor=SURFACE_COLOR,
        paper_bgcolor=SURFACE_COLOR,
        yaxis=dict(gridcolor=GRIDLINE_COLOR),
        xaxis=dict(gridcolor=GRIDLINE_COLOR, dtick=1),
    )
    return fig


def main() -> None:
    """Run the Source A x Source F cross-validation."""
    configure_logging()

    source_a = load_embeddings(EMBEDDINGS_PARQUET_PATH)
    source_f = load_source_f(SOURCE_F_PARQUET_PATH)
    merged = build_crossvalidation_table(source_a, source_f)

    merged[
        ["county_name", "fips_code", "content_length", "distress_count", "metro_2023"]
    ].to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d counties to %s", len(merged), OUTPUT_CSV_PATH)

    summary = summarize_crossvalidation(merged)
    logger.info("Mean content_length by distress count: %s", summary["content_length_by_distress"])
    logger.info("Mean content_length by metro status: %s", summary["content_length_by_metro"])
    logger.info(
        "distress_count vs. content_length: r=%.4f (p=%.4f)",
        summary["distress_vs_content_length_corr"],
        summary["distress_vs_content_length_p"],
    )
    logger.info(
        "metro vs. content_length (point-biserial): r=%.4f (p=%.4f)",
        summary["metro_vs_content_length_corr"],
        summary["metro_vs_content_length_p"],
    )

    fig = build_distress_length_chart(merged)
    fig.write_html(OUTPUT_HTML_PATH)
    logger.info("Wrote bar chart to %s", OUTPUT_HTML_PATH)


if __name__ == "__main__":
    main()
