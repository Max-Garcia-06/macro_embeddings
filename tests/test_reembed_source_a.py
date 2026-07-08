"""Tests for build_embedding_texts variant selection."""

import pandas as pd
import pytest

from reembed_source_a import build_embedding_texts


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_name": ["Lincoln County, Kansas"],
            "raw_intro_text": [
                "Lincoln County is a county in the U.S. state of Kansas."
            ],
        }
    )


def test_raw_variant_returns_unmodified_text() -> None:
    texts = build_embedding_texts(_fixture_df(), "raw")
    assert texts == ["Lincoln County is a county in the U.S. state of Kansas."]


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        build_embedding_texts(_fixture_df(), "bogus")


def test_v4_skips_frequency_filter_when_result_too_short() -> None:
    # Verified by direct inspection (scratch script, not committed) against
    # text_cleaning.clean_for_embedding and boilerplate_frequency's
    # find_common_templates/drop_common_sentences:
    #   - common_sentence survives strip_self_reference/strip_boilerplate_phrasing
    #     unchanged for both rows (no county/state name overlap, no boilerplate
    #     regex match), then masks to the same "<name> local histories ..."
    #     template for both rows, clearing the min-fraction threshold (2 of 2).
    #   - Millbrook's v2 text is 181 chars; after drop_common_sentences removes
    #     the common sentence, the remaining unique sentence is exactly 100
    #     chars (== MIN_CONTENT_LENGTH, chosen to also catch an off-by-one
    #     >/>= mutation), so v4 uses the filtered candidate.
    #   - Thistle's v2 text is 104 chars; after the same removal, the
    #     remaining unique sentence is only 23 chars (< MIN_CONTENT_LENGTH),
    #     so v4 falls back to the unfiltered 104-char v2 text.
    #   Both rows have candidate < original, i.e. real removal happens for
    #   both -- this is not a no-op for either row.
    common_sentence = (
        "Many local histories mention faraway explorers visiting "
        "centuries ago for trade."
    )
    long_unique = (
        "The local economy relies on small family farms and seasonal "
        "orchard harvests that supply nearby now."
    )
    assert len(long_unique) == 100
    rural_unique = "Farming is common here."

    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [
                f"{common_sentence} {long_unique}",
                f"{common_sentence} {rural_unique}",
            ],
        }
    )

    texts = build_embedding_texts(df, "v4")

    # Long-article county: boilerplate stripped as normal (matches v3) --
    # the filtered candidate is exactly 100 chars, i.e. >= MIN_CONTENT_LENGTH.
    assert texts[0] == long_unique
    assert "faraway explorers" not in texts[0]
    # Short/rural county: filtering would leave only 23 chars, below
    # MIN_CONTENT_LENGTH, so v4 falls back to the unfiltered 104-char v2
    # text and the boilerplate sentence is kept.
    assert "faraway explorers" in texts[1]
    assert rural_unique in texts[1]
    assert len(texts[1]) == 104


class FakeGemmaClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_v5_matches_v3_when_no_sentences_dropped(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [
                "The county maintains one of the state's oldest grain elevators.",
                "Farming is common in this part of the state.",
            ],
        }
    )
    v3_texts = build_embedding_texts(df, "v3")
    v5_texts = build_embedding_texts(
        df, "v5", gemma_client=FakeGemmaClient(response="{}"), cache_path=tmp_path / "cache.jsonl"
    )
    assert v5_texts == v3_texts


def test_v5_restores_sentence_gemma_marks_non_boilerplate(tmp_path) -> None:
    common_sentence = (
        "Many local histories mention faraway explorers visiting "
        "centuries ago for trade."
    )
    unique_a = "The county maintains one of the state's oldest grain elevators."
    unique_b = "Farming is common here."
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [f"{common_sentence} {unique_a}", f"{common_sentence} {unique_b}"],
        }
    )
    # Confirm the frequency filter does drop the common sentence under v3.
    v3_texts = build_embedding_texts(df, "v3")
    assert common_sentence not in v3_texts[0]

    client = FakeGemmaClient(response='{"0": true}')
    v5_texts = build_embedding_texts(
        df, "v5", gemma_client=client, cache_path=tmp_path / "cache.jsonl"
    )

    # Restored, and in original order (common sentence first, as in raw text).
    assert v5_texts[0] == f"{common_sentence} {unique_a}"
    assert v5_texts[1] == f"{common_sentence} {unique_b}"


def test_v5_falls_back_to_v3_when_gemma_call_fails(tmp_path) -> None:
    common_sentence = (
        "Many local histories mention faraway explorers visiting "
        "centuries ago for trade."
    )
    unique_a = "The county maintains one of the state's oldest grain elevators."
    unique_b = "Farming is common here."
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [f"{common_sentence} {unique_a}", f"{common_sentence} {unique_b}"],
        }
    )
    v3_texts = build_embedding_texts(df, "v3")
    v5_texts = build_embedding_texts(
        df,
        "v5",
        gemma_client=FakeGemmaClient(response="not json"),
        cache_path=tmp_path / "cache.jsonl",
    )
    assert v5_texts == v3_texts
