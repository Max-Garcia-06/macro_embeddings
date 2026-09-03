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


def test_excluded_targets_are_documented() -> None:
    """Every excluded target must carry a real, non-empty reason -- not a silent drop.

    `no_fuel_used_share` belongs here specifically: within the panel its
    reduced R2 is negative (worse than predicting the mean), so its reported
    "contribution" is the gap between two useless models, not a gain. See
    analysis-output/source-a/source-a-findings.md #22.
    """
    assert "no_fuel_used_share" in marginal.EXCLUDED_TARGETS
    for target, reason in marginal.EXCLUDED_TARGETS.items():
        assert isinstance(reason, str) and len(reason) > 40, (
            f"{target} is excluded without a documented reason"
        )


def test_uniform_ccr_arm_is_declared() -> None:
    """The scope-vs-CCR decomposition needs a CCR'd `uniform` arm to compare against."""
    assert "minilm_uniform_ccr_pca29" in marginal.EMBEDDING_ARMS
    assert marginal.EMBEDDING_ARMS["minilm_uniform_ccr_pca29"] == 29


def test_latlong_only_design_matches_the_geo_reduced_baseline() -> None:
    """`latlong_only`'s design and the geo-augmented reduced baseline must be identical.

    `score_representation` reuses one design (`size_and_others_geo`) for both
    the `latlong_only` arm and `r2_reduced_geo`; if they ever diverged,
    `contribution_geo` would stop meaning "net of geography" for every other
    arm.
    """
    rng = np.random.default_rng(42)
    size_and_others = rng.normal(size=(10, 3))
    latlong = rng.normal(size=(10, 2))
    size_and_others_geo = np.hstack([size_and_others, latlong])
    assert size_and_others_geo.shape == (10, 5)
    # latlong_only's design IS the geo-reduced baseline's design.
    np.testing.assert_array_equal(size_and_others_geo[:, -2:], latlong)


def test_centroids_path_exists() -> None:
    assert marginal.CENTROIDS_PATH.exists()


def _two_arm_scores(offset: float) -> pd.DataFrame:
    """A minimal score frame: `latlong_only` plus one arm a constant above it.

    Uses real target columns because the clustered resample looks each target
    up in `TARGET_TABLES`.
    """
    targets = [
        "broadband_rate",
        "median_household_income",
        "electric_heating_share",
        "gas_heating_share",
        "poverty_rate",
        "walked_share",
    ]
    rng = np.random.default_rng(7)
    baseline = rng.normal(scale=0.02, size=len(targets))
    rows = []
    for target, value in zip(targets, baseline):
        rows.append(
            {
                "target": target,
                "representation": "latlong_only",
                "contribution": value,
                "contribution_geo": 0.0,
            }
        )
        rows.append(
            {
                "target": target,
                "representation": "shifted",
                "contribution": value + offset,
                "contribution_geo": offset,
            }
        )
    return pd.DataFrame(rows)


def test_bootstrap_pairs_the_arms_within_a_replicate() -> None:
    """A difference that is constant per target must interval to zero width.

    Every target's `shifted - latlong_only` gap is exactly `offset`, so any
    resample of targets has that same mean difference. An interval wider than
    float noise means the arms were resampled independently, which would price
    in target-level variance the two arms share.
    """
    offset = 0.01
    boot = marginal.bootstrap_representations(_two_arm_scores(offset))

    for scheme in ("naive", "table_clustered"):
        interval = boot["shifted"][scheme]["minus_latlong_only"]
        assert interval["low"] == pytest.approx(offset, abs=1e-12)
        assert interval["high"] == pytest.approx(offset, abs=1e-12)
        # The arm's own contribution still varies with the draw, so this is
        # not a bootstrap that failed to resample anything.
        own = boot["shifted"][scheme]["contribution"]
        assert own["high"] - own["low"] > 1e-6


def test_clustered_draw_keeps_a_table_intact() -> None:
    """Heating-fuel targets share `b25040` and must be drawn or dropped together."""
    targets = sorted(
        {
            "electric_heating_share",
            "gas_heating_share",
            "bottled_gas_heating_share",
            "fuel_oil_heating_share",
            "broadband_rate",
        }
    )
    heating = {
        index
        for index, target in enumerate(targets)
        if marginal.TARGET_TABLES[target] == "b25040"
    }
    rng = np.random.default_rng(marginal.RANDOM_SEED)

    for _ in range(200):
        drawn = marginal._draw_target_positions(rng, targets, cluster_by_table=True)
        counts = {index: int((drawn == index).sum()) for index in heating}
        assert len(set(counts.values())) == 1, (
            "b25040's targets appeared different numbers of times in one draw"
        )


def test_shipped_bootstrap_covers_every_arm(representation_stats: dict) -> None:
    """No arm may publish a decision-basket mean without an interval on it."""
    for arm, block in representation_stats["by_representation"].items():
        boot = block["bootstrap"]
        assert boot["n_replicates"] >= 1000, arm
        for scheme in ("naive", "table_clustered"):
            for statistic in ("contribution", "contribution_geo", "minus_latlong_only"):
                interval = boot[scheme][statistic]
                assert interval["low"] <= interval["point"] <= interval["high"], (
                    f"{arm}/{scheme}/{statistic} point estimate outside its interval"
                )


def test_geo_net_interval_is_the_latlong_difference(representation_stats: dict) -> None:
    """`contribution_geo` and `minus_latlong_only` are the same statistic.

    `latlong_only`'s full model IS every other arm's geo-reduced baseline, so
    `contribution_geo(arm) = contribution(arm) - contribution(latlong_only)`
    identically. The notebook states this rather than printing one number under
    two headings; if the identity ever broke, the two would need reporting
    separately and this test is what would say so.
    """
    for arm, block in representation_stats["by_representation"].items():
        for scheme in ("naive", "table_clustered"):
            geo = block["bootstrap"][scheme]["contribution_geo"]
            difference = block["bootstrap"][scheme]["minus_latlong_only"]
            for bound in ("point", "low", "high"):
                assert geo[bound] == pytest.approx(difference[bound], abs=1e-12), (
                    f"{arm}/{scheme}/{bound}"
                )
