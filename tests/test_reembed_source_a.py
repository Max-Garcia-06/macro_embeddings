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
