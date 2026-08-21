"""Arms entering the marginal decision."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyze_source_a_representation_marginal as marginal
import source_a_typed_transform as typed_transform


def test_pca_arms_are_declared() -> None:
    assert "minilm_uniform_pca29" in marginal.EMBEDDING_ARMS
    assert "minilm_uniform_pca64" in marginal.EMBEDDING_ARMS


def test_pca_widths_match_their_names() -> None:
    assert marginal.EMBEDDING_ARMS["minilm_uniform_pca29"] == 29
    assert marginal.EMBEDDING_ARMS["minilm_uniform_pca64"] == 64


def test_reduce_fits_only_on_training_rows() -> None:
    """A reduction fitted on all rows leaks held-out states into the design."""
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 10))
    test = rng.normal(size=(20, 10)) + 100.0

    fitted = marginal.fit_reduction(train, n_components=3)
    reduced_test = fitted.transform(test)

    assert reduced_test.shape == (20, 3)
    assert np.abs(reduced_test).max() > 10, "test rows should sit far from the training centre"


def test_counts_are_log_transformed() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [0.0, 9.0], "has_economy_section": [0.0, 1.0]})
    tier = pd.Series(["stub", "rich"])

    design, names = typed_transform.transform_typed(
        frame, ["sec_n_industry_mentions", "has_economy_section"], tier
    )

    assert "log1p_sec_n_industry_mentions" in names
    column = design[:, names.index("log1p_sec_n_industry_mentions")]
    assert column[0] == pytest.approx(0.0)
    assert column[1] == pytest.approx(np.log1p(9.0))


def test_binary_columns_are_not_log_transformed() -> None:
    frame = pd.DataFrame({"has_economy_section": [0.0, 1.0]})
    tier = pd.Series(["stub", "rich"])
    _, names = typed_transform.transform_typed(frame, ["has_economy_section"], tier)
    assert "log1p_has_economy_section" not in names


def test_industry_mentions_interacts_with_tier() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [2.0, 4.0]})
    tier = pd.Series(["stub", "rich"])
    design, names = typed_transform.transform_typed(frame, ["sec_n_industry_mentions"], tier)

    assert "sec_n_industry_mentions_x_rich" in names
    column = design[:, names.index("sec_n_industry_mentions_x_rich")]
    assert column[0] == pytest.approx(0.0)
    assert column[1] == pytest.approx(4.0)


def test_original_columns_survive() -> None:
    frame = pd.DataFrame({"sec_n_industry_mentions": [2.0, 4.0]})
    tier = pd.Series(["stub", "rich"])
    _, names = typed_transform.transform_typed(frame, ["sec_n_industry_mentions"], tier)
    assert "sec_n_industry_mentions" in names
