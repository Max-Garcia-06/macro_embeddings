"""Section-scope rules and the text-leakage screen."""
from __future__ import annotations

import re

import pytest

import source_a_text_leakage as leak


def _matches(pattern: str, title: str) -> bool:
    return re.match(pattern, title, flags=re.IGNORECASE) is not None


@pytest.mark.parametrize(
    "title",
    ["2020 census", "2010 census", "2000 census", "demographics",
     "racial and ethnic composition", "population ranking"],
)
def test_census_titles_are_matched(title: str) -> None:
    assert _matches(leak.CENSUS_TITLE_PATTERN, title)


@pytest.mark.parametrize("title", ["economy", "geography", "politics", "education"])
def test_substantive_titles_are_not_census(title: str) -> None:
    assert not _matches(leak.CENSUS_TITLE_PATTERN, title)


@pytest.mark.parametrize(
    "title",
    ["communities", "unincorporated communities", "cities", "towns",
     "townships", "ghost towns", "adjacent counties", "census-designated places"],
)
def test_list_titles_are_matched(title: str) -> None:
    assert _matches(leak.LIST_TITLE_PATTERN, title)


def test_economy_survives_the_prose_exclusion() -> None:
    """The scope must never drop the 1.5% of text it exists to preserve."""
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "economy")
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "agriculture")
    assert not _matches(leak.PROSE_EXCLUDE_PATTERN, "industry")


def test_restatement_screen_finds_median_age(sections_frame) -> None:
    """median_age is stated verbatim in census sections; the screen must see it."""
    found = leak.restated_targets(sections_frame)
    assert found.get("median_age", 0) > 1000


def test_prose_scope_drops_at_least_a_third_of_characters(sections_frame) -> None:
    titles = sections_frame["section_title"].str.strip().str.lower()
    dropped = titles.str.match(leak.PROSE_EXCLUDE_PATTERN, na=False)
    share = sections_frame[dropped]["section_text"].str.len().sum() / (
        sections_frame["section_text"].str.len().sum()
    )
    assert share > 0.33, f"prose exclusion only drops {share:.1%} of characters"
