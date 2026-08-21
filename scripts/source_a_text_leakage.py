"""Which Wikipedia sections restate the external targets, and how to drop them.

`analyze_external_target.py` already guards one direction of target restatement:
`TARGET_RESTATEMENTS` ablates pillar columns that define rather than predict a
target. Nothing guarded the other direction. Wikipedia census sections state the
targets in words -- "The median age was 38.9 years" -- and the MiniLM arms read
those sections while the typed block, which extracts lexicon counts and no
numbers, cannot.

Dropping census sections is therefore a leakage control rather than a tuning
choice, and it is a precondition of widening the external basket rather than an
arm to be scored against it.
"""
from __future__ import annotations

import re

import pandas as pd

from analyze_source_a_section_scope import NARRATIVE_TITLE_PATTERN

# Sections that render a census table as prose. These carry the target values
# verbatim and are 36.4% of all section characters -- about 42% of what the
# `uniform` arm reads, since `uniform` already excludes narrative titles.
CENSUS_TITLE_PATTERN: str = (
    r"^(?:(?:19|20)\d0 census|census|demographics|population|"
    r"racial and ethnic composition|population ranking|"
    r"race and ethnicity|income and poverty)$"
)

# Name lists. Near content-free for a sentence encoder, and `adjacent counties`
# additionally acts as a geographic identifier rather than an economic signal.
LIST_TITLE_PATTERN: str = (
    r"^(?:communities|cities|towns?|townships|villages?|city|village|"
    r"unincorporated communities|other unincorporated communities|"
    r"census-designated places?|ghost towns?|adjacent counties|"
    r"national protected areas?|protected areas|lakes|population ranking)$"
)

HIGHWAY_TITLE_PATTERN: str = (
    r"^(?:major highways|major roads|highways|transportation|"
    r"airports?|railroads?|transit)$"
)

# Everything the `prose_only` family removes. `NARRATIVE_TITLE_PATTERN` is
# included because `build_variant_texts` applies it to every variant today, so
# folding it in here keeps one exclusion rule rather than two.
PROSE_EXCLUDE_PATTERN: str = (
    r"^(?:"
    + r"|".join(
        p.removeprefix("^(?:").removesuffix(")$")
        for p in (
            CENSUS_TITLE_PATTERN,
            LIST_TITLE_PATTERN,
            HIGHWAY_TITLE_PATTERN,
            NARRATIVE_TITLE_PATTERN,
        )
    )
    + r")$"
)

# Phrases that restate a target in prose. Deliberately narrow: this flags a
# target as leakage-exposed, it does not attempt to parse the value.
RESTATEMENT_PHRASES: dict[str, tuple[str, ...]] = {
    "median_age": (r"median age",),
    "median_household_income": (r"median income for a household", r"median household income"),
    "median_family_income": (r"median income for a family", r"median family income"),
    "per_capita_income": (r"per capita income",),
    "poverty_rate": (r"below the poverty line", r"poverty line", r"poverty level"),
    "median_home_value": (r"median value of",  r"median home value"),
    "mean_household_size": (r"average household size", r"average family size"),
    "owner_occupied_share": (r"owner-occupied",),
}


def census_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Subset `sections` to the census-table sections.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Rows whose title matches `CENSUS_TITLE_PATTERN`.
    """
    titles = sections["section_title"].str.strip().str.lower()
    return sections[titles.str.match(CENSUS_TITLE_PATTERN, na=False)]


def restated_targets(sections: pd.DataFrame) -> dict[str, int]:
    """Count counties whose census sections restate each target.

    Args:
        sections: Long-format section frame.

    Returns:
        Target column to the number of distinct counties restating it. Targets
        with no configured phrase are absent from the mapping.
    """
    census = census_sections(sections)
    counts: dict[str, int] = {}
    for column, phrases in RESTATEMENT_PHRASES.items():
        pattern = "|".join(phrases)
        hit = census["section_text"].str.contains(pattern, case=False, na=False, regex=True)
        counts[column] = int(census[hit]["fips_code"].nunique())
    return counts
