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
    common_sentence = (
        "This is a shared boilerplate sentence repeated across many "
        "counties for testing purposes today."
    )
    long_unique_base = (
        "has a very long and detailed unique history full of "
        "specific narrative content that goes well beyond the minimum "
        "content length threshold"
    )
    short_unique = "Short unique bit."

    df = pd.DataFrame(
        {
            "county_name": [f"County {i}, Somestate" for i in range(10)]
            + ["Rural County, Somestate"],
            "raw_intro_text": [
                f"{common_sentence} County {i} {long_unique_base}."
                for i in range(10)
            ]
            + [f"{common_sentence} {short_unique}"],
        }
    )

    texts = build_embedding_texts(df, "v4")

    # Long-article counties: boilerplate stripped as normal (matches v3).
    assert "shared boilerplate" not in texts[0]
    assert long_unique_base in texts[0]
    # Short/rural county: filtering would leave < MIN_CONTENT_LENGTH chars,
    # so the boilerplate sentence is kept instead of stripped.
    assert "shared boilerplate" in texts[-1]
