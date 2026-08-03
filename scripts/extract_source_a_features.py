"""Extract typed features from county Wikipedia intros, replacing the cut embedding.

Source A shipped a 1024-dim `bge-m3` embedding until it was cut on cost: it beat
`content_length` on 23 of 28 cross-pillar targets but by +0.0030 mean R2 lift
against +0.0010, for a 2.2GB model and CPU inference over 3,144 articles. What
remained was one scalar. This module is the attempt to do better than that scalar
without paying the embedding's price.

The design follows from one measurement. Splitting the corpus into
`content_length` quartiles and scanning for content types shows economic language
is not spread evenly -- it is concentrated:

    industry / economy words   9.4% (thin)  ->  43.1% (rich quartile)
    "named for / after"       42.9%         ->  45.7%

Corpus-wide only 19.7% of intros mention industry at all. An embedding averaged
over all 3,144 articles is therefore dominated by counties with nothing economic
to say, while the founding-narrative axis that PC1 latched onto (see
`source-a-findings.md` §3.2) is flat across quartiles and separates nothing.
Extraction inverts that: a county with no economic content returns False rather
than contributing noise to a dense vector.

Two conventions that matter to consumers:

- **Absence is False, not null.** Every flag is populated for every county. A
  demographic stub returns False across the board, and that sparsity is the
  signal -- it is what distinguishes it from a county that had something to say.
  `founding_year` is the sole exception: it is genuinely unknown when no
  founding clause is present, so it stays nullable.
- **Extraction reads `raw_intro_text`, never `embedding_text`.** The corpus
  frequency stripper that produces `embedding_text` removes the county name, the
  state name, and the phrase "U.S. state of" along with the boilerplate:

      RAW:   Nelson County is a county in the U.S. state of North Dakota. As of
             the 2020 census, the population was 3,015...
      CLEAN: the population was 3,015, and was estimated to be 2,963 in 2025.

  Everything below would be extracting from damaged input if it read the latter.

Run after `ingest_source_a.py`. The script is idempotent: it drops any columns it
previously wrote before recomputing, so re-running never doubles up. It extends
`source_a_text_features.parquet` in place rather than writing a second file,
because `pillar_matrix._derive_pillar_columns` already forwards every Source A
column except the two raw-text ones -- so new columns reach the feature matrix
with no change to that module.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

TEXT_FEATURES_PATH: Path = DATA_DIR / "source_a_text_features.parquet"
EXTRACTION_STATS_PATH: Path = ANALYSIS_DIR / "source_a_extraction_stats.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Lexicon:
    """One named boolean feature and the pattern that populates it.

    Attributes:
        column: Output column name.
        pattern: Case-sensitive regex. Case matters more than it looks -- the
            proper nouns being matched ("Fort Bragg", "Interstate 40") are
            capitalized, and lowercasing would match ordinary prose.
        group: Feature family, used to assemble the nested variants scored by
            `analyze_source_a_representation.py`.
    """

    column: str
    pattern: str
    group: str


# Industry families. Deliberately narrow: these match named economic activity,
# not the word "economy" on its own, because "the county's economy" appears in
# templated prose across counties that say nothing further about it.
INDUSTRY_LEXICON: tuple[Lexicon, ...] = (
    Lexicon("has_manufacturing", r"manufactur\w*|\bfactor(?:y|ies)\b|\bmills?\b|industrial park", "industry"),
    Lexicon("has_mining", r"\bmining\b|\bmines?\b|\bcoal\b|\bquarr(?:y|ies)\b", "industry"),
    Lexicon("has_oil_gas", r"\boil\b|\bpetroleum\b|natural gas|\brefiner(?:y|ies)\b|oilfield", "industry"),
    Lexicon("has_agriculture", r"agricultur\w*|\bfarming\b|\bfarmland\b|\branching\b|\bcrops?\b|\blivestock\b", "industry"),
    Lexicon("has_tourism", r"\btouris(?:m|t)\w*|\bresorts?\b|\bski\b|\bcasinos?\b|vacation", "industry"),
    Lexicon("has_timber", r"\btimber\b|\blogging\b|\blumber\b|\bforestry\b|\bsawmill", "industry"),
    Lexicon("has_logistics", r"\bshipping\b|\bfreight\b|distribution cent(?:er|re)|\bwarehous\w*", "industry"),
)

# Institutions large enough to shape a county's employment base. Each is a named
# entity in practice, which is why the patterns are capitalized.
#
# Two of these were tightened after a precision check on sampled matches, which
# is the step that makes lexicon extraction defensible rather than plausible:
#
# - `has_military_base` originally matched `\bFort [A-Z]` and `\bArmy\b`. Five of
#   six sampled hits were false: "Fort Wayne" and "Fort Yates" are city names,
#   "Fort Lemhi" an 1855 Mormon settlement, and `Army` catches Civil War prose.
#   It now requires an installation noun, so a county seat called Fort Benton no
#   longer reads as a military employer.
# - `has_tribal_land` originally matched bare `Indian` and `Tribe`, which caught
#   "American Indian Wars" and reservations dissolved in the 1830s. It now
#   requires a present-tense land or sovereignty term.
INSTITUTION_LEXICON: tuple[Lexicon, ...] = (
    Lexicon("has_university", r"\bUniversity\b|\bCollege\b", "institution"),
    Lexicon(
        "has_military_base",
        r"Air Force (?:Base|Station)|Naval (?:Base|Station|Air|Shipyard)|Marine Corps "
        r"(?:Base|Air)|Army (?:Base|Depot|Arsenal|Garrison)|military (?:base|installation)|"
        r"\bFort [A-Z]\w+ (?:Army|Air|Military|Base)",
        "institution",
    ),
    Lexicon("has_protected_land", r"National Park|National Forest|National Monument|Wildlife Refuge", "institution"),
    Lexicon(
        "has_tribal_land",
        r"Indian Reservation|\bReservation of\b|federally recognized|"
        r"\b(?:Navajo|Sioux|Cherokee|Apache|Hopi) Nation\b|Tribe of\b",
        "institution",
    ),
)

# Physical and transport geography. `has_coast` avoids the bare word "coast",
# which appears in directional prose ("the West Coast"), in favour of features a
# county either has or does not.
TRANSPORT_LEXICON: tuple[Lexicon, ...] = (
    Lexicon("has_river", r"\bRiver\b|\bCreek\b", "transport"),
    Lexicon("has_interstate", r"\bInterstate\b|\bU\.S\. Route\b|\bHighway\b", "transport"),
    Lexicon("has_port", r"\bPort of\b|\bHarbor\b|\bHarbour\b|\bseaport\b", "transport"),
)

# Metropolitan attachment. This is the one feature that restates something the
# other pillars also measure (Source F's `metro_2023`), kept because the intro
# names *which* metro area, which no other pillar does.
METRO_PATTERN: str = r"[Mm]etropolitan [Ss]tatistical [Aa]rea|[Mm]etropolitan area|[Mm]icropolitan"

# USDA typology phrasing echoed verbatim in some intros, e.g. Marquette County WI:
# "considered a high-recreation retirement destination by the U.S. Department of
# Agriculture". Extracted only so it can be *excluded* when scoring against Source
# F, whose `distress_count` is built from those same classifications -- otherwise
# Source A would be credited for predicting a label it copied.
USDA_ECHO_PATTERN: str = (
    r"[Dd]epartment of Agriculture|retirement destination|persistent poverty|"
    r"high-recreation|farming.dependent|mining.dependent"
)

NAMESAKE_PATTERN: str = r"[Nn]amed (?:for|after|in honor of)"

# Founding clauses. The year is captured from the clause rather than from the
# whole intro, because intros also carry census years, incorporation dates of the
# county seat, and battle dates, none of which are the county's founding.
FOUNDING_PATTERN: str = (
    r"(?:created|formed|organized|founded|established|incorporated)"
    r"(?:[^.]{0,40}?)\b(1[5-9]\d{2}|20[0-2]\d)\b"
)

# Tokens that are capitalized in every county intro by construction. Excluded
# from the proper-noun count so it measures *distinctive* named content rather
# than how many times the template fired.
PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset(
    {
        "County", "Parish", "Borough", "Census", "Area", "City", "Town", "Village",
        "United", "States", "State", "America", "American", "As", "The", "It", "Its",
        "This", "In", "On", "At", "By", "For", "From", "According", "Bureau",
        "Population", "Estimated", "Seat", "Largest", "Most", "North", "South",
        "East", "West", "Northern", "Southern", "Eastern", "Western", "Not",
        "Confused", "U", "S", "District", "Columbia", "Municipio",
    }
)

_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\b")

ALL_LEXICONS: tuple[Lexicon, ...] = INDUSTRY_LEXICON + INSTITUTION_LEXICON + TRANSPORT_LEXICON

# `has_usda_echo` is a diagnostic, not a feature, and is excluded from every
# variant below. It marks intros that restate USDA's own county classification
# ("considered a high-recreation retirement destination by the U.S. Department of
# Agriculture"), which is the input Source F's `distress_count` is built from.
# Scoring Source A against Source F with that column included would credit the
# pillar for predicting a label it copied. 16 counties carry it.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("has_usda_echo",)


# Nested feature groups scored as separate variants, answering "which columns earn
# their slot" rather than "how many principal components". Reduction on this block
# is a column-selection question, not a variance question -- PCA on Source A is a
# measured dead end (PCA-50 kept 42% of the full embedding's advantage, because
# its leading direction is the Texas founding-narrative artifact, not economics).
#
# The three widths are nested, so a flat profile across them means the extra
# columns are shrinking to zero and the narrow block is what should ship.
# `extracted_full` is filled in below, once `extracted_columns` is defined.
VARIANT_COLUMNS: dict[str, tuple[str, ...]] = {
    # The four columns with enough non-zero mass to plausibly move a model:
    # metro attachment fires on 51% of counties, proper-noun count is continuous
    # everywhere, and industry mentions carry the steepest tier gradient (23x).
    "extracted_min": (
        "content_length",
        "n_industry_mentions",
        "has_metro_attachment",
        "n_distinct_proper_nouns",
    ),
    "extracted_mid": (
        "content_length",
        "n_industry_mentions",
        "has_metro_attachment",
        "n_distinct_proper_nouns",
        "has_university",
        "has_military_base",
        "has_river",
        "has_interstate",
    ),
}


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def extracted_columns() -> list[str]:
    """List every column this module writes, in output order.

    Returns:
        Column names, excluding `content_length`, which `ingest_source_a.py`
        owns and this module never modifies.
    """
    return [
        *(item.column for item in ALL_LEXICONS),
        "n_industry_mentions",
        "has_metro_attachment",
        "has_namesake",
        "has_usda_echo",
        "founding_year",
        "n_distinct_proper_nouns",
    ]


VARIANT_COLUMNS["extracted_full"] = (
    "content_length",
    *(column for column in extracted_columns() if column not in DIAGNOSTIC_COLUMNS),
)


def count_distinct_proper_nouns(text: str, county_name: str) -> int:
    """Count distinct capitalized tokens that are not template boilerplate.

    Unlike `content_length`, this ignores the length of the templated prose and
    counts only named entities, so a long intro that says nothing specific does
    not score higher than a short one that names a river and an employer.

    Args:
        text: Raw intro text.
        county_name: The county's own display name, e.g. "Autauga County,
            Alabama". Its tokens are excluded, since every intro repeats them.

    Returns:
        Number of distinct qualifying proper-noun tokens.
    """
    own_tokens = set(re.findall(r"\b[A-Z][a-z]+\b", county_name))
    found = set(_PROPER_NOUN_PATTERN.findall(text))
    return len(found - own_tokens - PROPER_NOUN_STOPWORDS)


def extract_founding_year(text: str) -> float:
    """Extract the county's founding year from a founding clause.

    Args:
        text: Raw intro text.

    Returns:
        Earliest year appearing in a founding/creation clause, or NaN when no
        such clause is present. Earliest rather than first-mentioned, because
        intros that give both a creation and an organization date list them in
        inconsistent order.
    """
    years = [int(match) for match in re.findall(FOUNDING_PATTERN, text)]
    return float(min(years)) if years else float("nan")


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the full typed feature block for every county.

    Args:
        frame: Source A text features, carrying `raw_intro_text` and
            `county_name`.

    Returns:
        DataFrame indexed like `frame` with one column per entry in
        `extracted_columns()`. Boolean columns are populated for every row --
        absence is False, never null.
    """
    text = frame["raw_intro_text"]
    features = pd.DataFrame(index=frame.index)

    for item in ALL_LEXICONS:
        features[item.column] = text.str.contains(item.pattern, regex=True, na=False)

    industry_columns = [item.column for item in INDUSTRY_LEXICON]
    features["n_industry_mentions"] = features[industry_columns].sum(axis=1).astype("int64")

    features["has_metro_attachment"] = text.str.contains(METRO_PATTERN, regex=True, na=False)
    features["has_namesake"] = text.str.contains(NAMESAKE_PATTERN, regex=True, na=False)
    features["has_usda_echo"] = text.str.contains(USDA_ECHO_PATTERN, regex=True, na=False)

    features["founding_year"] = text.map(extract_founding_year)
    features["n_distinct_proper_nouns"] = [
        count_distinct_proper_nouns(row_text, name)
        for row_text, name in zip(text, frame["county_name"])
    ]

    return features[extracted_columns()]


def summarize(features: pd.DataFrame) -> dict[str, object]:
    """Collapse the extracted block to reportable coverage rates.

    Args:
        features: Output of `extract_features`.

    Returns:
        JSON-serializable summary: per-flag corpus rate, plus the counts that
        decide whether the block has enough non-zero mass to move a model.
    """
    flags = [column for column in features.columns if column.startswith("has_")]
    all_zero = (features[flags].sum(axis=1) == 0) & (features["n_distinct_proper_nouns"] == 0)
    return {
        "n_counties": int(len(features)),
        "n_features": int(features.shape[1]),
        "flag_rates": {column: float(features[column].mean()) for column in flags},
        "mean_industry_mentions": float(features["n_industry_mentions"].mean()),
        "share_any_industry": float((features["n_industry_mentions"] > 0).mean()),
        "mean_distinct_proper_nouns": float(features["n_distinct_proper_nouns"].mean()),
        "n_founding_year_present": int(features["founding_year"].notna().sum()),
        "n_usda_echo": int(features["has_usda_echo"].sum()),
        "n_all_flags_empty": int(all_zero.sum()),
    }


def main() -> None:
    """Extract the feature block and write it back into the Source A parquet."""
    configure_logging()

    try:
        frame = pd.read_parquet(TEXT_FEATURES_PATH)
    except FileNotFoundError:
        logger.error("Missing Source A text features: %s", TEXT_FEATURES_PATH)
        raise

    # Idempotence: a previous run's columns are dropped before recomputing, so
    # re-running never appends duplicates or leaves a stale column behind.
    frame = frame.drop(columns=extracted_columns(), errors="ignore")

    features = extract_features(frame)
    enriched = pd.concat([frame, features], axis=1)
    stats = summarize(features)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(TEXT_FEATURES_PATH, index=False)
    EXTRACTION_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %d features x %d counties to %s",
                stats["n_features"], stats["n_counties"], TEXT_FEATURES_PATH)
    logger.info("wrote %s", EXTRACTION_STATS_PATH)
    logger.info(
        "any industry: %.1f%% | mean proper nouns: %.1f | founding year: %d | "
        "USDA echo: %d | fully empty: %d",
        100 * stats["share_any_industry"],
        stats["mean_distinct_proper_nouns"],
        stats["n_founding_year_present"],
        stats["n_usda_echo"],
        stats["n_all_flags_empty"],
    )


if __name__ == "__main__":
    main()
