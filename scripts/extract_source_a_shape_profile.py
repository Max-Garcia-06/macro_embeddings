"""Shape profile: where an article's sections sit, and what they look like.

Round one (`extract_source_a_structure_features.py`) asked how many sections a
county's article has, how long they are, and which titles are present. It found
+0.00269 mean lift, and the branch review then showed roughly three quarters of
that was curvature in county size that a linear-in-logs control could not
absorb (`analysis-output/source-a/source-a-findings.md` §23).

This module adds the four families round one never built:

- **order and position** -- where each section sits, which is editorial priority
  and the one signal here with no obvious reading as a volume measure
- **template conformity** -- how far the article departs from the house skeleton
  county articles follow, which is editorial attention rather than county size
- **surface statistics** -- character-class densities; a census table rendered as
  prose is roughly 30% digits, and that is a fact about shape, not content
- **length curve** -- the sorted section-length curve beyond round one's Gini

It reads section titles and the *characters* of section text. It never reads
meaning: no lexicon, no word matching. That is the boundary and it is not
crossed.

**This module does not touch `data/source_a_structure_features.parquet`.** §23
cites that artifact; mutating it would silently invalidate a committed finding.
The new families go to `data/source_a_shape_profile.parquet` and
`analyze_source_a_shape_profile.py` joins the two.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from extract_source_a_structure_features import (
    SECTIONS_PARQUET_PATH,
    assign_buckets,
    flag_vocabulary,
    normalize_titles,
    slugify,
)
from pillar_matrix import DATA_DIR

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

SHAPE_PROFILE_PATH: Path = DATA_DIR / "source_a_shape_profile.parquet"
SHAPE_PROFILE_STATS_PATH: Path = ANALYSIS_DIR / "source_a_shape_profile_stats.json"

# Sentinel for "this county has no such section". Deliberately outside the
# [0, 1] range a real position occupies, and deliberately *not* 0.0: position
# 0.0 means "this section comes first", which is the opposite of absent. A tree
# can split the sentinel off cleanly; a linear model reads it as one step below
# the earliest possible position, which is the correct direction.
POSITION_ABSENT: float = -1.0

POSITION_PREFIX: str = "pos_"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def ordered_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Attach each section's normalized position within its county's article.

    Order is `section_id` order, which is the order the sections appear in the
    rendered article. Position is normalized to `[0, 1]` so a 40-section article
    and a 6-section article are comparable: what matters is whether the economy
    section sits a fifth of the way down or four fifths.

    A county with one section gets position 0.0 -- it is both first and last, and
    0.0 is the reading that keeps the sentinel at `POSITION_ABSENT` unambiguous.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        `sections` with `_position`, `_title` and `_chars` columns added, sorted
        by county and `section_id`.
    """
    frame = sections.assign(
        _title=normalize_titles(sections),
        _chars=sections["section_text"].fillna("").str.len().astype("float64"),
    ).sort_values(["fips_code", "section_id"])

    rank = frame.groupby("fips_code").cumcount().astype("float64")
    size = frame.groupby("fips_code")["section_id"].transform("size").astype("float64")
    # size - 1 is the number of gaps between sections; a one-section county has
    # none, and dividing by zero there would produce NaN rather than 0.0.
    span = (size - 1.0).replace(0.0, np.nan)
    frame["_position"] = (rank / span).fillna(0.0)
    return frame


def position_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Where each common section sits in the article.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.
        vocabulary: Titles to locate, from `flag_vocabulary`. Passed in rather
            than re-derived so `pos_<title>` and round one's `has_section_<title>`
            can never cover different title sets.

    Returns:
        DataFrame indexed by `fips_code` with one `pos_<title>` column per
        vocabulary entry plus `pos_longest_section`, `pos_first_economy`,
        `pos_first_census`, `pos_first_narrative`, `history_before_economy` and
        `position_spread`.
    """
    frame = ordered_sections(sections)
    frame["_bucket"] = assign_buckets(frame)
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    features = pd.DataFrame(index=index)

    for title in vocabulary:
        matched = frame.loc[frame["_title"] == title].groupby("fips_code")["_position"].min()
        features[f"{POSITION_PREFIX}{slugify(title)}"] = matched.reindex(index).fillna(
            POSITION_ABSENT
        )

    longest = frame.loc[frame.groupby("fips_code")["_chars"].idxmax()]
    features["pos_longest_section"] = longest.set_index("fips_code")["_position"].reindex(index)

    first_by_bucket: dict[str, pd.Series] = {}
    for bucket in ("economy", "census", "narrative"):
        first = frame.loc[frame["_bucket"] == bucket].groupby("fips_code")["_position"].min()
        first_by_bucket[bucket] = first.reindex(index)
        features[f"pos_first_{bucket}"] = first.reindex(index).fillna(POSITION_ABSENT)

    # Both present, and the narrative one earlier. A county missing either gets
    # 0.0: "no, history does not come first here" is the honest reading of an
    # article that has no history section.
    both = first_by_bucket["narrative"].notna() & first_by_bucket["economy"].notna()
    earlier = first_by_bucket["narrative"] < first_by_bucket["economy"]
    features["history_before_economy"] = (both & earlier).astype("float64")

    flagged = frame.loc[frame["_title"].isin(vocabulary)]
    features["position_spread"] = (
        flagged.groupby("fips_code")["_position"].std().reindex(index).fillna(0.0)
    )
    return features
