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
    BUCKET_KEYS,
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

    Raises:
        ValueError: If two distinct titles slugify to the same column name.
            Assigning both would keep the second and drop the first, leaving the
            result reporting N vocabulary entries against N-1 position columns
            and quietly breaking the auditability that using `flag_vocabulary`
            promises. Not reachable on the current corpus, which is exactly why
            it needs a guard rather than a comment.
    """
    frame = ordered_sections(sections)
    frame["_bucket"] = assign_buckets(frame)
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    features = pd.DataFrame(index=index)

    claimed_by: dict[str, str] = {}
    for title in vocabulary:
        column = f"{POSITION_PREFIX}{slugify(title)}"
        if column in claimed_by:
            raise ValueError(
                f"section titles {claimed_by[column]!r} and {title!r} share the slug "
                f"{column!r}; the position vocabulary must map one title to one column"
            )
        claimed_by[column] = title
        matched = frame.loc[frame["_title"] == title].groupby("fips_code")["_position"].min()
        features[column] = matched.reindex(index).fillna(POSITION_ABSENT)

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


# A title is part of the house skeleton if more than half of counties carry it.
# Half is the natural cut for "modal": below it, the set stops describing what a
# typical county article looks like and starts describing a large minority.
MODAL_TITLE_MIN_SHARE: float = 0.5

# A title held by under 1% of counties is unusual -- roughly 31 counties. Set
# well below round one's 5% flag floor on purpose: this measures the tail that
# floor excludes, so the two cuts describe different populations rather than
# two views of the same one.
UNUSUAL_TITLE_MAX_SHARE: float = 0.01


def title_county_shares(sections: pd.DataFrame) -> pd.Series:
    """Share of counties holding each section title.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Share per normalized title, indexed by title. Untitled sections are
        excluded -- round one's `n_untitled_sections` already counts them.
    """
    titles = normalize_titles(sections)
    n_counties = int(sections["fips_code"].nunique())
    titled = sections.assign(_title=titles).loc[titles != ""]
    return titled.groupby("_title")["fips_code"].nunique() / n_counties


def modal_title_set(sections: pd.DataFrame) -> list[str]:
    """The house skeleton: titles more than half of counties carry.

    Computed from the corpus rather than hardcoded, and written to the stats
    file, so a shifting skeleton is auditable rather than silent.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Normalized titles above `MODAL_TITLE_MIN_SHARE`, most common first.
    """
    shares = title_county_shares(sections)
    kept = shares[shares > MODAL_TITLE_MIN_SHARE]
    return list(kept.sort_values(ascending=False).index)


def template_features(sections: pd.DataFrame) -> pd.DataFrame:
    """How far the county's article departs from the house skeleton.

    Deviation from the template is editorial attention, which is a different
    quantity from county size: a small county someone cared about carries
    sections a large county's boilerplate article does not.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `template_jaccard`,
        `n_core_missing`, `n_unusual_sections`, `share_unusual_sections`,
        `mean_title_rarity` and `n_title_words`.
    """
    titles = normalize_titles(sections)
    shares = title_county_shares(sections)
    modal = set(modal_title_set(sections))
    unusual = set(shares[shares < UNUSUAL_TITLE_MAX_SHARE].index)

    frame = sections.assign(_title=titles).loc[titles != ""]
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    grouped = frame.groupby("fips_code")

    held = grouped["_title"].apply(set).reindex(index)
    held = held.where(held.notna(), other=pd.Series([set()] * len(index), index=index))

    features = pd.DataFrame(index=index)
    features["template_jaccard"] = [
        len(county & modal) / len(county | modal) if county | modal else 1.0 for county in held
    ]
    features["n_core_missing"] = [float(len(modal - county)) for county in held]
    features["n_unusual_sections"] = (
        frame.assign(_unusual=frame["_title"].isin(unusual).astype("float64"))
        .groupby("fips_code")["_unusual"]
        .sum()
        .reindex(index)
        .fillna(0.0)
    )
    # Denominator is titled sections, matching the numerator: an untitled section
    # cannot be unusual because it has no title to be rare.
    n_titled = grouped.size().reindex(index).fillna(0.0)
    features["share_unusual_sections"] = (
        features["n_unusual_sections"] / n_titled.replace(0.0, np.nan)
    ).fillna(0.0)
    features["mean_title_rarity"] = (
        frame.assign(_rarity=1.0 - frame["_title"].map(shares).astype("float64"))
        .groupby("fips_code")["_rarity"]
        .mean()
        .reindex(index)
        .fillna(0.0)
    )
    features["n_title_words"] = (
        frame.assign(_words=frame["_title"].str.split().str.len().astype("float64"))
        .groupby("fips_code")["_words"]
        .mean()
        .reindex(index)
        .fillna(0.0)
    )
    return features


# Buckets that get their own character-class densities. The four largest only:
# `economy`, `government`, `highways` and `other` are absent or near-empty for a
# large share of counties, and a density over zero characters is not a number --
# it is a zero standing where "no data" belongs, which a model reads as a
# measurement.
DENSITY_BUCKETS: tuple[str, ...] = ("census", "lists", "narrative", "geography")


def _class_counts(text: pd.Series) -> pd.DataFrame:
    """Count character classes per section.

    Args:
        text: Section text.

    Returns:
        DataFrame with `chars`, `word_chars`, `digits`, `letters`, `uppers`,
        `punct` and `words` columns, aligned to `text.index`. `word_chars`
        excludes whitespace, so a mean built on it reports the length of the
        words themselves rather than diluting it with the spaces between them.
    """
    filled = text.fillna("")
    return pd.DataFrame(
        {
            "chars": filled.str.len().astype("float64"),
            "word_chars": filled.str.count(r"\S").astype("float64"),
            "digits": filled.str.count(r"\d").astype("float64"),
            "letters": filled.str.count(r"[A-Za-z]").astype("float64"),
            "uppers": filled.str.count(r"[A-Z]").astype("float64"),
            "punct": filled.str.count(r"[^\w\s]").astype("float64"),
            "words": filled.str.split().str.len().fillna(0.0).astype("float64"),
        },
        index=filled.index,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, treating a zero denominator as zero rather than as NaN or an error.

    Args:
        numerator: Top of the ratio.
        denominator: Bottom of the ratio.

    Returns:
        The ratio, with zero-denominator rows set to 0.0.
    """
    return (numerator / denominator.replace(0.0, np.nan)).fillna(0.0)


def surface_features(sections: pd.DataFrame) -> pd.DataFrame:
    """Character-class densities, overall and for the four largest buckets.

    These read characters and never meaning. The signal they are after is
    documented: a census table rendered as prose is roughly 30% digits, a list of
    place names is short-worded and heavily capitalized, and narrative prose is
    neither -- all three are facts about an article's shape.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `digit_density`, `punct_density`,
        `capital_ratio`, `mean_word_length`, `numeral_to_letter`, and
        `digit_density_<bucket>` for each of `DENSITY_BUCKETS`.
    """
    counts = _class_counts(sections["section_text"])
    frame = counts.assign(fips_code=sections["fips_code"].to_numpy(), _bucket=assign_buckets(sections))
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    totals = (
        frame.groupby("fips_code")[
            ["chars", "word_chars", "digits", "letters", "uppers", "punct", "words"]
        ]
        .sum()
        .reindex(index)
        .fillna(0.0)
    )

    features = pd.DataFrame(index=index)
    features["digit_density"] = _safe_ratio(totals["digits"], totals["chars"])
    features["punct_density"] = _safe_ratio(totals["punct"], totals["chars"])
    features["capital_ratio"] = _safe_ratio(totals["uppers"], totals["letters"])
    features["mean_word_length"] = _safe_ratio(totals["word_chars"], totals["words"])
    features["numeral_to_letter"] = _safe_ratio(totals["digits"], totals["letters"])

    for bucket in DENSITY_BUCKETS:
        within = frame.loc[frame["_bucket"] == bucket].groupby("fips_code")[["digits", "chars"]].sum().reindex(index).fillna(0.0)
        features[f"digit_density_{bucket}"] = _safe_ratio(within["digits"], within["chars"])
    return features


def length_curve_features(sections: pd.DataFrame) -> pd.DataFrame:
    """The shape of the sorted section-length curve, and absolute bucket lengths.

    Round one shipped a Gini over section lengths and the share in the largest
    section. Both are scale-free, so neither can express "this county has a long
    economy section" -- only "a large fraction of this county's article is its
    economy section". The absolute `chars_<bucket>` columns close that gap.

    `chars_<bucket>` is `share_chars_<bucket>` times `total_body_chars`, both of
    which round one already ships, so it is derivable rather than new. It earns
    its place for the ridge learner, which cannot form products, and is redundant
    for the boosting learner, which can.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        DataFrame indexed by `fips_code` with `top3_length_share`,
        `length_decay_slope` and `chars_<bucket>` for every bucket.
    """
    frame = sections.assign(
        _chars=sections["section_text"].fillna("").str.len().astype("float64"),
        _bucket=assign_buckets(sections),
    )
    index = pd.Index(sorted(sections["fips_code"].unique()), name="fips_code")
    grouped = frame.groupby("fips_code")["_chars"]

    totals = grouped.sum().reindex(index).fillna(0.0)
    top3 = grouped.apply(lambda s: float(np.sort(s.to_numpy())[::-1][:3].sum())).reindex(index).fillna(0.0)

    features = pd.DataFrame(index=index)
    features["top3_length_share"] = _safe_ratio(top3, totals)
    features["length_decay_slope"] = (
        grouped.apply(lambda s: _decay_slope(s.to_numpy())).reindex(index).fillna(0.0)
    )

    per_bucket = (
        frame.pivot_table(index="fips_code", columns="_bucket", values="_chars", aggfunc="sum")
        .reindex(index=index, columns=list(BUCKET_KEYS))
        .fillna(0.0)
    )
    for bucket in BUCKET_KEYS:
        features[f"chars_{bucket}"] = per_bucket[bucket]
    return features


def _decay_slope(lengths: np.ndarray) -> float:
    """OLS slope of log length on rank, over a county's sections longest-first.

    Measures how fast the article falls away from its main section: a steep
    negative slope is one substantial section and a tail of stubs, a flat slope
    is an evenly developed article.

    Args:
        lengths: Section lengths for one county.

    Returns:
        The slope, or 0.0 when the county has fewer than two sections.
    """
    if len(lengths) < 2:
        return 0.0
    ordered = np.sort(lengths.astype("float64"))[::-1]
    ranks = np.arange(len(ordered), dtype="float64")
    slope, _ = np.polyfit(ranks, np.log1p(ordered), 1)
    return float(slope)
