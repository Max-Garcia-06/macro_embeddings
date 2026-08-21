"""Arms entering the marginal decision."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyze_source_a_representation_marginal as marginal


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
