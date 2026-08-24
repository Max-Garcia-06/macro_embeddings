"""What a Source A article is made of, by share of characters.

The representation section of the status notebook opens on the claim that the
encoder was reading mostly boilerplate: census tables rendered as prose are the
largest single category of text in the corpus and economy-titled sections are a
rounding error. That claim was quoted from a comment in
`analyze_source_a_tiered_embedding.py` and typed into the notebook by hand. This
script computes it instead, so a number that moves upstream moves in the
notebook.

Categories are the title patterns the text-scope arms already use, applied in
precedence order -- `CENSUS_TITLE_PATTERN` before `LIST_TITLE_PATTERN` because
`population ranking` appears in both, and the census reading is the one that
matters for leakage. Anything a pattern does not claim falls to `other`, which
is geography, government, education and the long tail of one-off titles.

Lead sections are not in `data/source_a_sections.parquet` and are therefore not
in these shares: this measures the *body*, which is what every text-scope arm
beyond `lead_only` is deciding whether to read.

Output: `analysis-output/source-a/source_a_section_composition_stats.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from analyze_source_a_section_scope import NARRATIVE_TITLE_PATTERN
from extract_source_a_section_features import ECONOMY_TITLE_PATTERN, SECTIONS_PARQUET_PATH
from source_a_text_leakage import (
    CENSUS_TITLE_PATTERN,
    HIGHWAY_TITLE_PATTERN,
    LIST_TITLE_PATTERN,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
STATS_PATH: Path = (
    REPO_ROOT / "analysis-output" / "source-a" / "source_a_section_composition_stats.json"
)

# Ordered: the first pattern to match a title wins. `population ranking` is in
# both the census and list patterns, and `transportation` reads as highway
# rather than economy, so the order is load-bearing and not alphabetical.
CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("census", "census tables rendered as prose", CENSUS_TITLE_PATTERN),
    ("lists", "lists of place names, adjacent counties, highways", LIST_TITLE_PATTERN),
    ("lists", "lists of place names, adjacent counties, highways", HIGHWAY_TITLE_PATTERN),
    ("narrative", "history and notable people", NARRATIVE_TITLE_PATTERN),
    ("economy", "economy-titled sections", ECONOMY_TITLE_PATTERN),
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def compose(sections: pd.DataFrame) -> dict[str, object]:
    """Split body-section characters across the title categories.

    Args:
        sections: Every body section, one row per section, as persisted by
            `ingest_source_a.py`.

    Returns:
        Mapping with the per-category character shares and the counts the
        notebook quotes alongside them.
    """
    titles = sections["section_title"].str.strip().str.lower()
    chars = sections["section_text"].str.len()
    total = float(chars.sum())

    claimed = pd.Series(False, index=sections.index)
    shares: dict[str, float] = {}
    labels: dict[str, str] = {}
    for key, label, pattern in CATEGORIES:
        matched = titles.str.match(pattern, na=False) & ~claimed
        shares[key] = shares.get(key, 0.0) + float(chars[matched].sum()) / total
        labels[key] = label
        claimed |= matched

    shares["other"] = float(chars[~claimed].sum()) / total
    labels["other"] = "geography, government, education, other"

    economy_counties = int(
        sections.loc[titles.str.match(ECONOMY_TITLE_PATTERN, na=False), "fips_code"].nunique()
    )
    n_counties = int(sections["fips_code"].nunique())

    return {
        "n_sections": int(len(sections)),
        "n_counties": n_counties,
        "total_characters": total,
        "labels": labels,
        "share_of_characters": shares,
        "n_counties_with_economy_section": economy_counties,
        "share_counties_without_economy_section": 1 - economy_counties / n_counties,
    }


def main() -> None:
    """Compute the composition and write the stats artifact."""
    configure_logging()
    sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    logger.info("Read %s (%d sections)", SECTIONS_PARQUET_PATH, len(sections))

    stats = compose(sections)
    for key, share in stats["share_of_characters"].items():
        logger.info("%-10s %.4f", key, share)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    logger.info("Wrote %s", STATS_PATH)


if __name__ == "__main__":
    main()
