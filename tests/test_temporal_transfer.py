"""The temporal-transfer design: what it controls for, and what it refuses to claim.

The failure modes worth locking here are design failures rather than arithmetic
ones. A change-prediction test that omits the lagged level scores mean reversion
as signal; one whose baseline and treatment arms differ by more than the pillars
scores that difference instead; one that averages a target which is mostly
differenced sampling noise reports a null nothing could have beaten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyze_temporal_transfer as att


PILLARS = "ABCDEF"


def test_every_model_except_the_fixed_effect_controls_for_the_lagged_level() -> None:
    """Without y(early) in the design, regression to the mean scores as prediction."""
    for model in att.MODELS:
        if model.name == "fixed_effect":
            assert not model.uses_lagged, "the FE bar is an intercept, by definition"
            continue
        assert model.uses_lagged, f"{model.name} would credit mean reversion to its features"


def test_the_geo_lift_pairs_arms_that_differ_only_by_the_pillars() -> None:
    """The geo-controlled lift is only the pillars' worth if both sides hold lat/lon."""
    by_name = {model.name: model for model in att.MODELS}
    treatment, baseline = by_name["lagged_size_geo_emacro"], by_name["lagged_size_geo"]

    assert treatment.uses_geo and baseline.uses_geo
    assert treatment.uses_lagged == baseline.uses_lagged
    assert treatment.uses_size == baseline.uses_size
    assert treatment.uses_pillars and not baseline.uses_pillars


def test_each_arm_differs_from_its_reference_only_by_the_pillars() -> None:
    """A contribution is the pillars' worth only if nothing else moves with them."""
    by_name = {model.name: model for model in att.MODELS}
    for model in att.MODELS:
        if not model.reference:
            continue
        reference = by_name[model.reference]
        assert model.uses_lagged == reference.uses_lagged
        assert model.uses_size == reference.uses_size
        assert model.uses_geo == reference.uses_geo, (
            f"{model.name} and {model.reference} differ in geography as well as pillars"
        )


def test_the_two_pillar_arms_state_a_lift_rather_than_a_contribution() -> None:
    """`contribution` is reference minus self, which only reads correctly when
    self is the reduced model. Both pillar arms are the fuller one, so neither
    may carry a reference -- their lift is differenced the other way."""
    by_name = {model.name: model for model in att.MODELS}
    for name in ("lagged_size_emacro", "lagged_size_geo_emacro"):
        assert by_name[name].uses_pillars
        assert by_name[name].reference == "", f"{name} would report its lift negated"
    assert by_name["lagged_size_geo_emacro"].uses_geo
    assert by_name["lagged_size_geo"].uses_geo


def test_every_pillar_has_a_drop_arm() -> None:
    names = {model.name for model in att.MODELS}
    for pillar in PILLARS:
        assert f"lagged_size_emacro_drop_{pillar}" in names


def test_build_design_orders_lagged_size_geo_then_pillars() -> None:
    usable = pd.DataFrame(
        {
            "income_early": [1.0, 2.0, 3.0],
            "log_population": [4.0, 5.0, 6.0],
            "log_agi": [7.0, 8.0, 9.0],
            "log_gdp_latest": [1.5, 2.5, 3.5],
            "lat": [30.0, 40.0, 50.0],
            "lon": [-90.0, -95.0, -100.0],
            "pillar_one": [0.0, 1.0, 2.0],
            "pillar_two": [2.0, 1.0, 0.0],
        }
    )
    pillar_columns = ["pillar_one", "pillar_two"]
    by_name = {model.name: model for model in att.MODELS}

    lagged = att.build_design(usable, by_name["lagged"], pillar_columns, "income_early")
    assert lagged.shape == (3, 1)
    np.testing.assert_allclose(lagged[:, 0], [1.0, 2.0, 3.0])

    full = att.build_design(
        usable, by_name["lagged_size_geo_emacro"], pillar_columns, "income_early"
    )
    assert full.shape[1] == 1 + len(att.SIZE_FEATURES) + len(att.GEO_FEATURES) + 2

    dropped = att.build_design(
        usable,
        by_name["lagged_size_emacro"],
        pillar_columns,
        "income_early",
        ablate=("pillar_two",),
    )
    assert dropped.shape[1] == 1 + len(att.SIZE_FEATURES) + 1


def test_fixed_effect_design_is_a_constant() -> None:
    """A geographic fixed effect has no parameter for movement; its design is an intercept."""
    usable = pd.DataFrame({"income_early": [1.0, 2.0, 3.0]})
    by_name = {model.name: model for model in att.MODELS}
    design = att.build_design(usable, by_name["fixed_effect"], [], "income_early")

    assert design.shape == (3, 1)
    assert len(np.unique(design)) == 1


def test_noise_share_adds_both_vintages_variances() -> None:
    """Differencing two estimates adds their sampling variances, it does not average them."""
    usable = pd.DataFrame(
        {
            "income_se_early": [3.0, 3.0, 3.0, 3.0],
            "income_se_late": [4.0, 4.0, 4.0, 4.0],
        }
    )
    change = np.array([0.0, 10.0, 20.0, 30.0])

    share = att.noise_share_of_change(usable, "income", change)

    expected = (9.0 + 16.0) / float(np.var(change, ddof=1))
    assert share == pytest.approx(expected)


def test_noise_share_is_nan_when_nothing_moved() -> None:
    usable = pd.DataFrame({"x_se_early": [1.0, 1.0], "x_se_late": [1.0, 1.0]})
    assert np.isnan(att.noise_share_of_change(usable, "x", np.array([5.0, 5.0])))


def test_overlap_is_reported_and_is_real() -> None:
    """Two five-year windows three years apart share two years of sample."""
    assert att.LATE_YEAR - att.EARLY_YEAR == 3
    assert 5 - (att.LATE_YEAR - att.EARLY_YEAR) == 2


def _fabricate_scores(targets: list[str], lift: float) -> pd.DataFrame:
    """Build a score frame with a fixed lift on every target."""
    rows: list[dict[str, object]] = []
    for target in targets:
        for model in att.MODELS:
            r2, contribution, model_lift, geo_lift = 0.10, np.nan, 0.0, 0.0
            if model.name == "lagged_size_emacro":
                r2, model_lift = 0.10 + lift, lift
            elif model.name == "lagged_size_geo_emacro":
                r2, geo_lift = 0.10 + lift, lift / 2
            elif model.name.startswith("lagged_size_emacro_drop_"):
                contribution = lift / 6
            rows.append(
                {
                    "target": target,
                    "model": model.name,
                    "r2_change": r2,
                    "contribution": contribution,
                    "lift_over_lagged_size": model_lift,
                    "lift_over_lagged_size_geo": geo_lift,
                    "noise_share_of_change": 0.1,
                    "mostly_noise": False,
                }
            )
    return pd.DataFrame(rows)


def test_bootstrap_collapses_when_every_target_agrees(monkeypatch) -> None:
    targets = sorted(att.TARGET_TABLES)[:6]
    monkeypatch.setattr(att, "N_BOOTSTRAP", 200)

    result = att.bootstrap_headline(_fabricate_scores(targets, 0.02), targets)

    for scheme in ("naive", "table_clustered"):
        interval = result[scheme]["lift_over_lagged_size"]
        assert interval["point"] == pytest.approx(0.02)
        assert interval["low"] == pytest.approx(0.02)
        assert interval["high"] == pytest.approx(0.02)
        assert not interval["covers_zero"]


def test_bootstrap_rejects_an_unscored_basket_target(monkeypatch) -> None:
    targets = sorted(att.TARGET_TABLES)[:4]
    scores = _fabricate_scores(targets, 0.01)
    scores = scores[
        ~((scores["model"] == "lagged_size_emacro") & (scores["target"] == targets[0]))
    ]
    monkeypatch.setattr(att, "N_BOOTSTRAP", 50)

    with pytest.raises(ValueError, match="every model must score every basket target"):
        att.bootstrap_headline(scores, targets)


def test_mostly_noise_targets_leave_the_headline_basket(monkeypatch) -> None:
    """A target nothing could predict must not be averaged into a null result."""
    targets = sorted(att.TARGET_TABLES)[:5]
    scores = _fabricate_scores(targets, 0.02)
    noisy = targets[0]
    scores.loc[scores["target"] == noisy, "mostly_noise"] = True
    scores.loc[scores["target"] == noisy, "noise_share_of_change"] = 0.95
    monkeypatch.setattr(att, "N_BOOTSTRAP", 100)

    stats = att.summarize(scores, pd.DataFrame(), targets)

    assert stats["excluded_mostly_noise"] == [noisy]
    assert noisy not in stats["basket"]
    assert stats["n_targets_in_basket"] == len(targets) - 1
    assert noisy not in stats["by_target"]


def test_summary_reports_the_geo_controlled_lift_separately(monkeypatch) -> None:
    targets = sorted(att.TARGET_TABLES)[:4]
    monkeypatch.setattr(att, "N_BOOTSTRAP", 100)

    stats = att.summarize(_fabricate_scores(targets, 0.02), pd.DataFrame(), targets)

    assert stats["mean_lift_over_lagged_size"] == pytest.approx(0.02)
    assert stats["mean_lift_over_lagged_size_geo"] == pytest.approx(0.01)
    assert stats["overlap_years"] == 2
