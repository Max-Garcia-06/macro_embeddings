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
