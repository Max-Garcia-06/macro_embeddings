"""Extract typed features from article body sections, not just the lead.

§13 established that typed extraction from Wikipedia intros beats the shipped
`content_length` scalar by 2.6x, and that its largest wins come from named
industry content -- tourism mentions predicting Accommodation & Food Services LQ,
for instance. The limit on that result is coverage: only 6.5% of counties name an
industry in their intro at all, and 25.2% even in the richest content tier.

Body sections are where the rest of it should be. §4 of the findings measured
~25% of counties as having a dedicated Economy section, four times the intro's
industry coverage. §4/§4.1 nonetheless closed section expansion -- but on
Mantel-r against *geographic* distance, the yardstick this project has since
rejected for this pillar (see the header of
`analyze_source_a_representation.py`). Two things also make that negative result
weaker evidence against *extraction* than it was against embedding:

- The finding was that body sections are **more templated** than the lead. That
  is a serious problem for a dense embedding, which absorbs boilerplate into
  every dimension. Targeted extraction reads named facts and ignores prose, so
  templating costs it far less.
- The failure mode §4.1 diagnosed was *inconsistency* -- the LLM cleaner kept
  geographic anchoring for some counties and dropped it for others. A fixed
  lexicon applied to a named section cannot be inconsistent in that way.

This module therefore reads only sections whose title marks them as economic, and
extracts the same industry lexicon used on the lead. Everything else in the
article is left alone: this is not section expansion, it is targeted extraction
from one named section.

Run after `ingest_source_a.py` has written `source_a_sections.parquet`, and after
`extract_source_a_features.py`. Idempotent, like that module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from extract_source_a_features import INDUSTRY_LEXICON, TEXT_FEATURES_PATH

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

SECTIONS_PARQUET_PATH: Path = DATA_DIR / "source_a_sections.parquet"
SECTION_STATS_PATH: Path = ANALYSIS_DIR / "source_a_section_stats.json"

# Section titles that mark economic content. Matched case-insensitively against
# the whole title, so "Economy and industry" and "Economy" both qualify while
# "Economic history of the county seat" does not slip in via a bare substring.
ECONOMY_TITLE_PATTERN: str = (
    r"^(?:economy|economics|industry|industries|economy and industry|"
    r"economic development|agriculture|business|employment|"
    r"economy and infrastructure|transportation and industry)$"
)

# Prefix distinguishing section-derived columns from the lead-derived ones, so a
# consumer can tell which part of the article a fact came from.
SECTION_PREFIX: str = "sec_"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# Written to the parquet but excluded from the scored feature set. An ablation
# found the section block's advantage survives its removal almost intact
# (+0.00320 against +0.00328 mean lift), while it is the most size-dependent
# column in Source A -- r = 0.550 against log tax returns, well above
# `content_length`'s 0.359. A column that adds 2.4% of the gain and that much
# size dependence is a poor trade in a feature set whose central open question is
# whether size is a control or a target. Kept as a diagnostic.
SECTION_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("n_body_sections",)


def section_output_columns() -> list[str]:
    """List every column this module writes, in output order.

    Returns:
        Column names, including the diagnostics excluded from scoring.
    """
    return [
        *(f"{SECTION_PREFIX}{item.column}" for item in INDUSTRY_LEXICON),
        f"{SECTION_PREFIX}n_industry_mentions",
        "has_economy_section",
        *SECTION_DIAGNOSTIC_COLUMNS,
    ]


def section_feature_columns() -> list[str]:
    """List the section-derived columns eligible to be scored as features.

    Returns:
        `section_output_columns()` minus `SECTION_DIAGNOSTIC_COLUMNS`.
    """
    return [c for c in section_output_columns() if c not in SECTION_DIAGNOSTIC_COLUMNS]


def select_economy_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Keep only sections whose title marks them as economic.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Subset of `sections`, one or more rows per county that has such a
        section.
    """
    titles = sections["section_title"].str.strip().str.lower()
    return sections[titles.str.match(ECONOMY_TITLE_PATTERN, na=False)]


def build_section_features(frame: pd.DataFrame, sections: pd.DataFrame) -> pd.DataFrame:
    """Compute the section-derived feature block for every county.

    Counties with no economy section get False and zero throughout, matching the
    lead-derived block's convention that absence is a value rather than a gap.

    Args:
        frame: Source A text features, indexed by row with a `fips_code` column.
        sections: Long-format section frame.

    Returns:
        DataFrame aligned to `frame`'s index with `section_output_columns()`.
    """
    economy = select_economy_sections(sections)
    # One text blob per county, since a county may have both "Economy" and
    # "Agriculture" and the lexicon should see both.
    economy_text = economy.groupby("fips_code")["section_text"].agg(" ".join)

    features = pd.DataFrame(index=frame.index)
    aligned_text = frame["fips_code"].map(economy_text).fillna("")

    for item in INDUSTRY_LEXICON:
        features[f"{SECTION_PREFIX}{item.column}"] = aligned_text.str.contains(
            item.pattern, regex=True, na=False
        )

    industry_columns = [f"{SECTION_PREFIX}{item.column}" for item in INDUSTRY_LEXICON]
    features[f"{SECTION_PREFIX}n_industry_mentions"] = (
        features[industry_columns].sum(axis=1).astype("int64")
    )
    features["has_economy_section"] = frame["fips_code"].isin(economy["fips_code"]).to_numpy()
    features["n_body_sections"] = (
        frame["fips_code"].map(sections.groupby("fips_code").size()).fillna(0).astype("int64")
    )

    return features[section_output_columns()]


def summarize(frame: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    """Report section coverage and the marginal yield over intro-only extraction.

    The headline number is `share_industry_added`: counties whose economy section
    names an industry that the intro did not. That is the entire case for having
    refetched the article bodies, so it is reported directly rather than left to
    be inferred from the harness.

    Args:
        frame: Text features carrying the intro-derived columns and `tier`.
        features: Output of `build_section_features`.

    Returns:
        JSON-serializable summary, overall and per content tier.
    """
    intro_has = frame["n_industry_mentions"] > 0
    section_has = features[f"{SECTION_PREFIX}n_industry_mentions"] > 0
    added = section_has & ~intro_has

    combined = pd.concat([frame[["tier"]], features], axis=1).assign(
        intro_has=intro_has, section_has=section_has, added=added
    )
    by_tier = combined.groupby("tier", observed=True).agg(
        n_counties=("added", "size"),
        share_economy_section=("has_economy_section", "mean"),
        mean_body_sections=("n_body_sections", "mean"),
        share_intro_industry=("intro_has", "mean"),
        share_section_industry=("section_has", "mean"),
        share_industry_added=("added", "mean"),
    )

    return {
        "n_counties": int(len(frame)),
        "share_economy_section": float(features["has_economy_section"].mean()),
        "mean_body_sections": float(features["n_body_sections"].mean()),
        "share_intro_industry": float(intro_has.mean()),
        "share_section_industry": float(section_has.mean()),
        "share_industry_added": float(added.mean()),
        "n_industry_added": int(added.sum()),
        "by_tier": json.loads(by_tier.to_json(orient="index")),
    }


def main() -> None:
    """Extract section features and write them into the Source A parquet."""
    configure_logging()

    from analyze_source_a_tiers import assign_tiers

    try:
        frame = pd.read_parquet(TEXT_FEATURES_PATH)
        sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    except FileNotFoundError:
        logger.error(
            "Need both %s and %s -- run ingest_source_a.py then "
            "extract_source_a_features.py first.",
            TEXT_FEATURES_PATH,
            SECTIONS_PARQUET_PATH,
        )
        raise

    if "n_industry_mentions" not in frame.columns:
        raise ValueError("Intro features absent -- run extract_source_a_features.py first.")

    frame = frame.drop(columns=section_output_columns(), errors="ignore")
    features = build_section_features(frame, sections)
    stats = summarize(frame.assign(tier=assign_tiers(frame["content_length"])), features)

    enriched = pd.concat([frame, features], axis=1)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(TEXT_FEATURES_PATH, index=False)
    SECTION_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %d section features to %s", len(features.columns), TEXT_FEATURES_PATH)
    logger.info("wrote %s", SECTION_STATS_PATH)
    logger.info(
        "economy section: %.1f%% of counties | industry named in intro %.1f%% -> "
        "with sections %.1f%% (+%d counties)",
        100 * stats["share_economy_section"],
        100 * stats["share_intro_industry"],
        100 * (stats["share_intro_industry"] + stats["share_industry_added"]),
        stats["n_industry_added"],
    )


if __name__ == "__main__":
    main()
