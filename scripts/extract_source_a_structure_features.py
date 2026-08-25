"""Structural features of a county's Wikipedia article -- shape, not content.

Everything here is derived from two things: a section's title and how many
characters it contains. No section text is read. That restriction is the point
of the round: `analyze_source_a_representation.py` scores what the article
*says*, and this module asks what the article's skeleton knows on its own.

The prior is that the answer is "county size". `n_body_sections` was computed
during the section round, correlated r = 0.550 against log tax returns -- above
`content_length`'s 0.359 -- and cut from the scored block for that reason
(`pillar_matrix.SOURCE_A_DIAGNOSTIC_COLUMNS`). So this module only builds the
block; `analyze_source_a_structure.py` scores it on a baseline that already
holds three size measures, where a pure size proxy is worth approximately
nothing.

Output: `data/source_a_structure_features.parquet`, one row per county.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_source_a_section_scope import NARRATIVE_TITLE_PATTERN
from extract_source_a_section_features import ECONOMY_TITLE_PATTERN, SECTIONS_PARQUET_PATH
from pillar_matrix import DATA_DIR
from source_a_text_leakage import (
    CENSUS_TITLE_PATTERN,
    HIGHWAY_TITLE_PATTERN,
    LIST_TITLE_PATTERN,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

STRUCTURE_FEATURES_PATH: Path = DATA_DIR / "source_a_structure_features.parquet"
STRUCTURE_FEATURE_STATS_PATH: Path = ANALYSIS_DIR / "source_a_structure_feature_stats.json"

# A section shorter than this is a stub -- a heading with a sentence under it.
# Not arbitrary: the corpus-wide first quartile of section length is 108
# characters and the median is 340, so 200 splits the bottom of the
# distribution rather than trimming a tail.
STUB_CHAR_THRESHOLD: int = 200

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def normalize_titles(sections: pd.DataFrame) -> pd.Series:
    """Strip and case-fold section titles.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized title per row, aligned to `sections.index`. Untitled
        sections -- 2,009 rows corpus-wide -- become the empty string rather
        than being dropped.
    """
    return sections["section_title"].fillna("").str.strip().str.lower()


def gini(values: np.ndarray) -> float:
    """Gini coefficient over a county's section lengths.

    Measures how unevenly the article's characters are distributed across its
    sections: 0 when every section is the same length, approaching 1 when one
    section holds nearly everything.

    Args:
        values: Section lengths for one county.

    Returns:
        Gini coefficient, or 0.0 for an empty or all-zero input.
    """
    if len(values) == 0:
        return 0.0
    ordered = np.sort(values.astype("float64"))
    total = float(ordered.sum())
    if total <= 0:
        return 0.0
    n = len(ordered)
    index = np.arange(1, n + 1)
    return float(2.0 * float((index * ordered).sum()) / (n * total) - (n + 1) / n)


def count_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Count how many sections a county's article has, and of what kind.

    `section_id` is Parsoid's numbering and is not contiguous within a county:
    ids are skipped where sections nest or were dropped during ingestion, so
    `n_id_gaps` is a free structural signal about how deep the article's
    hierarchy goes.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with the five count columns.
    """
    titles = normalize_titles(sections)
    frame = sections.assign(_title=titles, _untitled=(titles == "").astype("int64"))
    grouped = frame.groupby("fips_code")

    counts = pd.DataFrame(
        {
            "n_body_sections": grouped.size(),
            "n_distinct_titles": grouped["_title"].nunique(),
            "n_untitled_sections": grouped["_untitled"].sum(),
            "max_section_id": grouped["section_id"].max(),
        }
    )
    counts["n_id_gaps"] = counts["max_section_id"] - counts["n_body_sections"]
    counts.index.name = "fips_code"
    return counts


def length_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Summarize how long a county's sections are and how evenly.

    Spread statistics are filled rather than left null. A one-section county has
    an undefined sample standard deviation, and "undefined" here means "there is
    no spread", not "unknown" -- letting it through as NaN would hand the
    imputer a median from counties that are not comparable.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with the nine length columns.
    """
    chars = sections["section_text"].fillna("").str.len().astype("float64")
    frame = sections[["fips_code"]].assign(
        _chars=chars, _stub=(chars < STUB_CHAR_THRESHOLD).astype("int64")
    )
    grouped = frame.groupby("fips_code")

    lengths = pd.DataFrame(
        {
            "total_body_chars": grouped["_chars"].sum(),
            "mean_section_chars": grouped["_chars"].mean(),
            "median_section_chars": grouped["_chars"].median(),
            "max_section_chars": grouped["_chars"].max(),
            "sd_section_chars": grouped["_chars"].std().fillna(0.0),
            "n_stub_sections": grouped["_stub"].sum(),
            "section_length_gini": grouped["_chars"].apply(lambda s: gini(s.to_numpy())),
        }
    )
    totals = lengths["total_body_chars"].replace(0.0, np.nan)
    lengths["share_in_largest_section"] = (lengths["max_section_chars"] / totals).fillna(0.0)
    lengths["share_stub_sections"] = lengths["n_stub_sections"] / grouped.size()
    lengths.index.name = "fips_code"
    return lengths


# A title earns a flag when it appears in more than this share of counties.
# 5% of 3,144 counties is a ~157-county floor, which keeps the head of the
# distribution -- `demographics` at 3,142 counties down through the mid-tail --
# and drops the one-off titles that are really county names in disguise.
TITLE_FLAG_MIN_SHARE: float = 0.05

# Prefix marking a column as "this section title was present", so a consumer can
# tell these apart from the shipped lexicon flags (`has_university`,
# `has_economy_section`) that describe what the text says.
TITLE_FLAG_PREFIX: str = "has_section_"


def slugify(title: str) -> str:
    """Convert a section title into a valid, stable column suffix.

    Args:
        title: Normalized section title.

    Returns:
        Lowercase alphanumeric-and-underscore slug, or `"untitled"` when the
        title has no usable characters.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug or "untitled"


def flag_vocabulary(sections: pd.DataFrame) -> list[str]:
    """Choose which section titles are common enough to flag.

    Computed from the corpus rather than hardcoded, so the vocabulary moves when
    the corpus does. The chosen set is written to the stats file, which is what
    makes a shifting vocabulary auditable instead of silent.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized titles held by more than `TITLE_FLAG_MIN_SHARE` of counties,
        most common first. Untitled sections are excluded -- their count is
        already carried by `n_untitled_sections`.
    """
    titles = normalize_titles(sections)
    n_counties = int(sections["fips_code"].nunique())
    titled = sections.assign(_title=titles).loc[titles != ""]
    per_title = titled.groupby("_title")["fips_code"].nunique()
    kept = per_title[per_title / n_counties > TITLE_FLAG_MIN_SHARE]
    return list(kept.sort_values(ascending=False).index)


def title_flag_features(sections: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Flag which of the common section titles each county's article carries.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.
        vocabulary: Normalized titles to flag, from `flag_vocabulary`.

    Returns:
        DataFrame indexed by `fips_code`, one float64 0.0/1.0 column per title.
        A title appearing twice in one article is still one flag: this asks
        which structures are present, not how many times.
    """
    titles = normalize_titles(sections)
    frame = sections[["fips_code"]].assign(_title=titles)
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")

    flags = pd.DataFrame(index=index)
    for title in vocabulary:
        holders = frame.loc[frame["_title"] == title, "fips_code"].unique()
        flags[f"{TITLE_FLAG_PREFIX}{slugify(title)}"] = index.isin(holders).astype("float64")
    return flags


# Physical-setting sections. `national protected area` is deliberately absent:
# `LIST_TITLE_PATTERN` already claims it and runs first, so listing it here
# would be a dead alternative that reads as if it did something.
GEOGRAPHY_TITLE_PATTERN: str = (
    r"^(?:geography|geography and climate|climate|geology|topography|"
    r"terrain|environment|physical geography)$"
)

# Civic sections. Education sits here rather than in its own bucket: it is a
# county-government function in these articles, and splitting it would produce
# a bucket too thin to carry a share.
GOVERNMENT_TITLE_PATTERN: str = (
    r"^(?:government|politics|government and politics|law and government|"
    r"politics and government|education|law enforcement|elections|voting)$"
)

# Ordered: the first pattern to claim a title wins, and the order is
# load-bearing rather than alphabetical. Inherited from
# `analyze_source_a_section_composition.CATEGORIES`, with two deliberate
# differences:
#
# - Highways get their own bucket instead of folding into `lists`. The
#   composition script merges them because it is asking how much of the corpus
#   is content-free for an encoder; this round is asking which *structures* are
#   present, and a highway section is a different structure from a list of towns.
# - `geography` and `government` are split out of the `other` residual. They are
#   its two largest occupants, and leaving them in would put most of the corpus
#   in a bucket named "other".
STRUCTURE_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("census", CENSUS_TITLE_PATTERN),
    ("lists", LIST_TITLE_PATTERN),
    ("highways", HIGHWAY_TITLE_PATTERN),
    ("narrative", NARRATIVE_TITLE_PATTERN),
    ("economy", ECONOMY_TITLE_PATTERN),
    ("geography", GEOGRAPHY_TITLE_PATTERN),
    ("government", GOVERNMENT_TITLE_PATTERN),
)

BUCKET_KEYS: tuple[str, ...] = tuple(key for key, _ in STRUCTURE_CATEGORIES) + ("other",)

# Prefix for the character-share columns.
BUCKET_SHARE_PREFIX: str = "share_chars_"


def assign_buckets(sections: pd.DataFrame) -> pd.Series:
    """Label every section with the first thematic bucket that claims it.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Bucket key per row, aligned to `sections.index`. Anything no pattern
        claims is `"other"`.
    """
    titles = normalize_titles(sections)
    buckets = pd.Series("other", index=sections.index)
    claimed = pd.Series(False, index=sections.index)
    for key, pattern in STRUCTURE_CATEGORIES:
        matched = titles.str.match(pattern, na=False) & ~claimed
        buckets[matched] = key
        claimed |= matched
    return buckets


def bucket_share_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Split each county's body characters across the thematic buckets.

    Shares rather than counts, so the block describes the article's composition
    rather than its length -- length is already carried, and carried better, by
    `total_body_chars`.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with one `share_chars_<bucket>` column
        per bucket, each county's row summing to 1 unconditionally. A county
        whose sections are all zero-length has nothing to divide up; it is
        routed entirely to `share_chars_other` rather than left at all-zero,
        since an article with no characters is, by the bucket vocabulary's own
        logic, unclassifiable rather than a hole in the invariant.
    """
    chars = sections["section_text"].fillna("").str.len().astype("float64")
    frame = sections[["fips_code"]].assign(_chars=chars, _bucket=assign_buckets(sections))

    per_bucket = (
        frame.pivot_table(index="fips_code", columns="_bucket", values="_chars", aggfunc="sum")
        .reindex(columns=list(BUCKET_KEYS))
        .fillna(0.0)
    )
    totals = per_bucket.sum(axis=1)
    zero_total = totals == 0.0
    shares = per_bucket.div(totals.replace(0.0, np.nan), axis=0).fillna(0.0)
    shares.loc[zero_total, :] = 0.0
    shares.loc[zero_total, "other"] = 1.0
    shares.columns = [f"{BUCKET_SHARE_PREFIX}{key}" for key in shares.columns]
    shares.index.name = "fips_code"
    return shares


def structure_feature_columns(features: pd.DataFrame) -> list[str]:
    """List the scored columns of an assembled structural block.

    Args:
        features: Output of `build_structure_features`.

    Returns:
        Every column except the `fips_code` key.
    """
    return [column for column in features.columns if column != "fips_code"]


def build_structure_features(sections: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the full structural block for every county.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Tuple of (features, vocabulary). `features` has `fips_code` as a column
        and every feature as float64; `vocabulary` is the title set the flags
        were built from, which the caller records so a shifting corpus is
        auditable.
    """
    vocabulary = flag_vocabulary(sections)
    parts = [
        count_features(sections),
        length_features(sections),
        title_flag_features(sections, vocabulary),
        bucket_share_features(sections),
    ]
    features = pd.concat(parts, axis=1).astype("float64")
    features.index.name = "fips_code"
    return features.reset_index(), vocabulary


def summarize(features: pd.DataFrame, vocabulary: list[str]) -> dict[str, object]:
    """Describe the block for the notebook and for later auditing.

    Args:
        features: Output of `build_structure_features`.
        vocabulary: Title set the flags were built from.

    Returns:
        Counts, the chosen vocabulary and thresholds, and per-column summary
        statistics keyed by column name.
    """
    columns = structure_feature_columns(features)
    block = features[columns]
    return {
        "n_counties": int(len(features)),
        "n_features": len(columns),
        "title_flag_vocabulary": vocabulary,
        "title_flag_min_share": TITLE_FLAG_MIN_SHARE,
        "stub_char_threshold": STUB_CHAR_THRESHOLD,
        "bucket_keys": list(BUCKET_KEYS),
        "mean_bucket_share": {
            f"{BUCKET_SHARE_PREFIX}{key}": float(block[f"{BUCKET_SHARE_PREFIX}{key}"].mean())
            for key in BUCKET_KEYS
        },
        "column_summary": {
            column: {
                "mean": float(block[column].mean()),
                "sd": float(block[column].std()),
                "min": float(block[column].min()),
                "max": float(block[column].max()),
            }
            for column in columns
        },
    }


def main() -> None:
    """Build the structural block from the section parquet and write it out."""
    configure_logging()

    try:
        sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    except FileNotFoundError:
        logger.error("Need %s -- run ingest_source_a.py first.", SECTIONS_PARQUET_PATH)
        raise

    features, vocabulary = build_structure_features(sections)
    stats = summarize(features, vocabulary)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(STRUCTURE_FEATURES_PATH, index=False)
    STRUCTURE_FEATURE_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info(
        "wrote %d structural features for %d counties to %s",
        stats["n_features"],
        stats["n_counties"],
        STRUCTURE_FEATURES_PATH,
    )
    logger.info("title flags: %d titles above %.0f%% of counties", len(vocabulary), TITLE_FLAG_MIN_SHARE * 100)
    logger.info("wrote %s", STRUCTURE_FEATURE_STATS_PATH)


if __name__ == "__main__":
    main()
