"""Split counties by how much distinctive content their Wikipedia intro carries.

Source A's corpus is not uniform. After boilerplate stripping, the distinctive
content of a county intro spans 33 to 4,403 characters -- p05 is 73, the median
283, p95 1,047. Treating all 3,144 counties identically means the counties with
something to say are averaged against the majority that have nothing, which is
the mechanism behind two of this pillar's recorded negative results (§4.1 of
`source-a-findings.md` found LLM cleaning kept geographic anchoring for some
counties and dropped it for others, adding noise rather than signal).

This module defines the tiers and reports what each one actually contains. Two
things it is deliberately **not**:

- **Not a shipped feature.** Tier membership tracks county size -- `content_length`
  correlates with metro status at r = 0.247 -- so shipping it would smuggle a
  size proxy into a feature set whose central open question is whether size is a
  control or a target. Tiers exist to route work and to break out results, and
  the composition table below reports each tier's metro share precisely so that
  confound stays visible rather than implicit.
- **Not a branching schema.** `extract_source_a_features.py` writes identical
  columns for every county. A stub county returns False across the board, and
  that sparsity is what encodes its tier. Downstream consumers see one schema.

Run after `extract_source_a_features.py`, whose columns this module reports on --
the content lexicons live there and are not duplicated here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from extract_source_a_features import ALL_LEXICONS, TEXT_FEATURES_PATH

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_tiers.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_tier_stats.json"

# Cut points on `content_length`, in characters. The stub band is set at 100
# rather than at a quantile because it is the threshold Source A's existing
# clustering analysis already uses to drop counties as having no real content
# (`drop_stub_counties`, §2 of the findings) -- reusing it keeps this split
# comparable to those results. The remaining three bands are the tercile
# boundaries of what is left, so no band is so small that its composition rates
# are noise.
TIER_EDGES: tuple[int, ...] = (0, 100, 284, 462)
TIER_LABELS: tuple[str, ...] = ("stub", "thin", "mid", "rich")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def assign_tiers(content_length: pd.Series) -> pd.Series:
    """Assign each county to a content tier.

    Args:
        content_length: Characters of distinctive content per county.

    Returns:
        Ordered categorical Series aligned to the input, with categories
        `TIER_LABELS`.
    """
    return pd.cut(
        content_length,
        bins=[*TIER_EDGES, int(content_length.max()) + 1],
        right=False,
        labels=list(TIER_LABELS),
        ordered=True,
    )


def build_composition(frame: pd.DataFrame) -> pd.DataFrame:
    """Report per-tier rates for every extracted content flag.

    Args:
        frame: Source A text features carrying `tier` plus the extracted columns.

    Returns:
        DataFrame indexed by feature name, one column per tier plus `corpus`,
        holding the share of counties where the flag is True.
    """
    flags = [item.column for item in ALL_LEXICONS] + [
        "has_metro_attachment",
        "has_namesake",
        "has_usda_echo",
    ]
    grouped = frame.groupby("tier", observed=True)[flags].mean().T
    grouped["corpus"] = frame[flags].mean()
    return grouped


def build_tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize size and content volume per tier.

    The `mean_industry_mentions` and `mean_proper_nouns` columns are the two
    continuous content measures; `content_length` is included so the tier's own
    defining variable is visible alongside them.

    Args:
        frame: Source A text features carrying `tier` and the extracted columns.

    Returns:
        One row per tier.
    """
    grouped = frame.groupby("tier", observed=True)
    return pd.DataFrame(
        {
            "n_counties": grouped.size(),
            "min_length": grouped["content_length"].min(),
            "max_length": grouped["content_length"].max(),
            "mean_length": grouped["content_length"].mean().round(1),
            "share_any_industry": (
                frame.assign(any_industry=frame["n_industry_mentions"] > 0)
                .groupby("tier", observed=True)["any_industry"]
                .mean()
                .round(4)
            ),
            "mean_industry_mentions": grouped["n_industry_mentions"].mean().round(3),
            "mean_proper_nouns": grouped["n_distinct_proper_nouns"].mean().round(2),
            "share_founding_year": grouped["founding_year"].apply(
                lambda s: round(float(s.notna().mean()), 4)
            ),
        }
    )


def main() -> None:
    """Assign tiers, report composition, and write the per-county tier CSV."""
    configure_logging()

    try:
        frame = pd.read_parquet(TEXT_FEATURES_PATH)
    except FileNotFoundError:
        logger.error("Missing Source A text features: %s", TEXT_FEATURES_PATH)
        raise

    if "n_industry_mentions" not in frame.columns:
        raise ValueError(
            "Extracted columns absent -- run extract_source_a_features.py first."
        )

    frame = frame.assign(tier=assign_tiers(frame["content_length"]))
    summary = build_tier_summary(frame)
    composition = build_composition(frame)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    frame[["fips_code", "county_name", "content_length", "tier"]].to_csv(
        OUTPUT_CSV_PATH, index=False
    )
    OUTPUT_STATS_PATH.write_text(
        json.dumps(
            {
                "tier_edges": list(TIER_EDGES),
                "tier_labels": list(TIER_LABELS),
                "summary": json.loads(summary.to_json(orient="index")),
                "composition": json.loads(composition.to_json(orient="index")),
            },
            indent=2,
        )
    )

    logger.info("tier summary:\n%s", summary.to_string())
    logger.info("content composition by tier:\n%s", composition.round(3).to_string())
    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
