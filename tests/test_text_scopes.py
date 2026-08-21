"""Section-scope rules and the text-leakage screen."""
from __future__ import annotations

import re

import numpy as np
import pytest

import source_a_text_leakage as leak
from ingest_external_targets import EXTERNAL_TARGETS


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


def test_every_external_target_is_accounted_for_by_the_screen() -> None:
    """No target may fall through the crack between screened and unscreened.

    `restated_targets` maps every EXTERNAL_TARGETS column to either a county
    count (screened, phrase configured) or `None` (unscreened, no plausible
    phrase). A column missing from both `RESTATEMENT_PHRASES` and
    `UNSCREENED_TARGETS` would silently read as "screened, found nothing" to
    a caller doing `.get(column, 0)` -- this asserts that can't happen.
    """
    accounted = set(leak.RESTATEMENT_PHRASES) | leak.UNSCREENED_TARGETS
    for target in EXTERNAL_TARGETS:
        assert target.column in accounted, f"{target.column} is neither screened nor unscreened"


def test_unscreened_targets_map_to_none_not_zero(sections_frame) -> None:
    """The screened/unscreened distinction must survive into the return value."""
    found = leak.restated_targets(sections_frame)
    for column in leak.UNSCREENED_TARGETS:
        assert found[column] is None
    for column in leak.RESTATEMENT_PHRASES:
        assert isinstance(found[column], int)


def test_prose_scope_drops_at_least_a_third_of_characters(sections_frame) -> None:
    titles = sections_frame["section_title"].str.strip().str.lower()
    dropped = titles.str.match(leak.PROSE_EXCLUDE_PATTERN, na=False)
    share = sections_frame[dropped]["section_text"].str.len().sum() / (
        sections_frame["section_text"].str.len().sum()
    )
    assert share > 0.33, f"prose exclusion only drops {share:.1%} of characters"


import analyze_source_a_tiered_embedding as tiered


def test_new_variants_are_registered() -> None:
    keys = {variant.key for variant in tiered.TEXT_VARIANTS}
    assert {"prose_only", "prose_plus_history", "economy_all_tiers", "prose_by_tier"} <= keys


def test_variant_exclusion_defaults_to_narrative() -> None:
    """Existing variants must keep the behaviour they were measured under."""
    uniform = next(v for v in tiered.TEXT_VARIANTS if v.key == "uniform")
    assert uniform.exclude is None


def test_prose_plus_history_readmits_narrative() -> None:
    variant = next(v for v in tiered.TEXT_VARIANTS if v.key == "prose_plus_history")
    assert "history" not in variant.exclude


def test_prose_by_tier_reads_lead_for_thin_tiers() -> None:
    variant = next(v for v in tiered.TEXT_VARIANTS if v.key == "prose_by_tier")
    assert variant.tier_scope["stub"] is None
    assert variant.tier_scope["thin"] is None
    assert variant.tier_scope["mid"] is not None
    assert variant.tier_scope["rich"] is not None


def test_common_component_removal_centres_the_corpus() -> None:
    rng = np.random.default_rng(42)
    shared = np.ones(8) * 5.0
    vectors = shared + rng.normal(scale=0.1, size=(100, 8))

    result = tiered.remove_common_component(vectors)

    assert result.shape == vectors.shape
    assert np.abs(result.mean(axis=0)).max() < 0.2, "shared component should be gone"
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_common_component_removal_handles_a_zero_row() -> None:
    """Counties with no text get a zero vector; it must not produce NaN."""
    vectors = np.vstack([np.ones((5, 4)), np.zeros((1, 4))])
    result = tiered.remove_common_component(vectors)
    assert np.isfinite(result).all()
