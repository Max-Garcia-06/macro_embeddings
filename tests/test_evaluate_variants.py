"""Unit test for evaluate_variant on a synthetic 6-county fixture."""

import numpy as np
import pandas as pd

from evaluate_source_a_variants import evaluate_variant


def _make_fixture() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    names = [f"County {i}, Somestate" for i in range(6)]
    emb = rng.normal(size=(6, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    variant_df = pd.DataFrame({"county_name": names, "embedding": list(emb)})
    analysis_df = pd.DataFrame(
        {"county_name": names, "lat": np.linspace(30, 45, 6), "lon": np.linspace(-120, -75, 6)}
    )
    n = 6
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) * 500.0
    return variant_df, analysis_df, dist


def test_evaluate_variant_metrics() -> None:
    variant_df, analysis_df, dist = _make_fixture()
    result = evaluate_variant(variant_df, analysis_df, dist)
    emb = np.vstack(variant_df["embedding"].to_numpy())
    sims = (emb @ emb.T)[np.triu_indices(6, k=1)]
    assert result["n_counties"] == 6
    assert result["pairwise_similarity_mean"] == float(sims.mean())
    assert result["pairwise_similarity_std"] == float(sims.std())
    assert -1.0 <= result["mantel_r"] <= 1.0
    assert 0.0 < result["mantel_p"] <= 1.0
    assert result["tracked_pair_mean"] is None  # fixture has no tracked pairs
    assert len(result["top_far_similar_pairs"]) >= 1


def test_evaluate_variant_rejects_missing_counties() -> None:
    variant_df, analysis_df, dist = _make_fixture()
    import pytest

    with pytest.raises(ValueError):
        evaluate_variant(variant_df.iloc[:4], analysis_df, dist)
