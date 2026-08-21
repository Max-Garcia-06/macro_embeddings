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
from ingest_external_targets import EXTERNAL_TARGETS

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
# target as leakage-exposed, it does not attempt to parse the value. Every
# phrase below was checked against `census_sections()` text with `str.contains`
# before being added -- the same way a line number gets verified -- because the
# obvious phrase for several targets turns out to name a *different* construct
# with the same vocabulary. Two rejected candidates, kept here as a record:
#   - "female householder" / "male householder" reads as the household-type
#     share from B11001 ("28.2% were households with a female householder and
#     no spouse present"), not the B09002 *children's* family-type share the
#     `children_*_householder_share` targets actually measure. Using it would
#     have flagged those targets as leaked on a different table's number.
#   - "mortgage" in census-section text is almost always the split-out monthly
#     *cost* figure ("$1,083 with a mortgage, $312 without"), not the B25081
#     share of owner-occupied units carrying a mortgage that
#     `mortgaged_share` measures. Kept unscreened rather than matched on
#     vocabulary that names a different table.
RESTATEMENT_PHRASES: dict[str, tuple[str, ...]] = {
    "median_age": (r"median age",),
    "median_household_income": (r"median income for a household", r"median household income"),
    "median_family_income": (r"median income for a family", r"median family income"),
    "per_capita_income": (r"per capita income",),
    "poverty_rate": (r"below the poverty line", r"poverty line", r"poverty level"),
    "median_home_value": (r"median value of",  r"median home value"),
    "mean_household_size": (r"average household size", r"average family size"),
    "owner_occupied_share": (r"owner-occupied",),
    # Verified 2026-08-21 while covering the remaining 34 EXTERNAL_TARGETS
    # columns (finding 1, review of task-4/5/6). "vacant" alone reads as the
    # overall vacancy figure ("X% were vacant") in every sampled hit; the
    # homeowner/rental sub-rates ride along in the same sentence.
    "housing_vacancy_rate": (r"\bvacant\b", r"vacancy rate"),
    "family_household_share": (r"family households?",),
    "bachelors_share": (r"bachelor'?s degree",),
    "masters_share": (r"master'?s degree",),
    "foreign_born_share": (r"foreign[- ]born",),
    "naturalized_share_of_foreign_born": (r"naturalized",),
    "median_gross_rent": (r"gross rent",),
    "median_monthly_housing_cost": (r"monthly housing costs?",),
    "computer_ownership_share": (r"households? had a computer",),
    "broadband_rate": (r"broadband",),
    "labor_force_participation": (
        r"labor force participation",
        r"(?:were|was) in the (?:civilian )?labor force",
    ),
    "drove_alone_share": (r"drove alone",),
    "carpooled_share": (r"carpooled",),
    "walked_share": (r"walked to work",),
    "work_from_home_share": (r"work(?:ed)? from home", r"worked at home"),
    "public_transit_share": (r"public transportation", r"public transit"),
    # Shared marker for all three B09002 "own children" targets. Generic
    # household-type vocabulary ("female/male householder") was rejected for
    # this family precisely because it names B11001's construct instead --
    # see the module comment above. "own children" is the one phrase sampled
    # hits actually used in that construct's context; it is rare (single
    # digits of counties) rather than absent, and reported as such.
    "children_married_couple_share": (r"own children",),
    "children_female_householder_share": (r"own children",),
    "children_male_householder_share": (r"own children",),
}

# Targets in EXTERNAL_TARGETS with no phrase above: checked against the real
# section text and found either zero genuine hits, or hits that share
# vocabulary with a different construct (see the two rejected candidates in
# the comment on RESTATEMENT_PHRASES). `restated_targets` maps every one of
# these to `None` rather than omitting them, so a caller cannot mistake "not
# screened" for "screened, found nothing" -- comparing `None > 500` raises
# instead of silently passing.
UNSCREENED_TARGETS: frozenset[str] = frozenset(
    {
        "mean_commute_minutes",
        "median_contract_rent",
        "median_year_built",
        "single_unit_share",
        "electric_heating_share",
        "gas_heating_share",
        "bottled_gas_heating_share",
        "fuel_oil_heating_share",
        "no_fuel_used_share",
        "same_house_share",
        "moved_within_county_share",
        "moved_different_state_share",
        "household_earnings_share",
        "household_ss_income_share",
        "mortgaged_share",
    }
)

_EXTERNAL_TARGET_COLUMNS: frozenset[str] = frozenset(t.column for t in EXTERNAL_TARGETS)
_accounted = frozenset(RESTATEMENT_PHRASES) | UNSCREENED_TARGETS
assert _accounted == _EXTERNAL_TARGET_COLUMNS, (
    "RESTATEMENT_PHRASES and UNSCREENED_TARGETS must partition EXTERNAL_TARGETS exactly -- "
    f"missing: {_EXTERNAL_TARGET_COLUMNS - _accounted}, "
    f"extra: {_accounted - _EXTERNAL_TARGET_COLUMNS}"
)


def census_sections(sections: pd.DataFrame) -> pd.DataFrame:
    """Subset `sections` to the census-table sections.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.

    Returns:
        Rows whose title matches `CENSUS_TITLE_PATTERN`.
    """
    titles = sections["section_title"].str.strip().str.lower()
    return sections[titles.str.match(CENSUS_TITLE_PATTERN, na=False)]


def restated_targets(sections: pd.DataFrame) -> dict[str, int | None]:
    """Count counties whose census sections restate each target.

    Every column in `EXTERNAL_TARGETS` is a key in the result. A target with a
    configured phrase (`RESTATEMENT_PHRASES`) maps to its county count, which
    may legitimately be 0. A target in `UNSCREENED_TARGETS` -- no plausible
    restatement phrase was found -- maps to `None`. The two must never collapse
    to the same value: `found.get(column, 0)` on this mapping is a bug, because
    it silently treats "not screened" as "screened, clean".

    Args:
        sections: Long-format section frame.

    Returns:
        Target column to county count (screened) or `None` (unscreened).
    """
    census = census_sections(sections)
    counts: dict[str, int | None] = {}
    for column, phrases in RESTATEMENT_PHRASES.items():
        pattern = "|".join(phrases)
        hit = census["section_text"].str.contains(pattern, case=False, na=False, regex=True)
        counts[column] = int(census[hit]["fips_code"].nunique())
    for column in UNSCREENED_TARGETS:
        counts[column] = None
    return counts
