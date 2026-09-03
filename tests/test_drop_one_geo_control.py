"""The geography control and the intervals on the drop-one figure.

Two additions to `analyze_external_target.py`, tested against synthetic score
frames rather than a re-run of the sweep: every statistic here is a pure
function of the per-target contributions the sweep already wrote, so a fixture
that fabricates those contributions exercises the same code path in
milliseconds.

The one thing that does need the real designs is that `lat`/`lon` land in a geo
model's predictor array and stay out of a plain one. That is a shape assertion
and is tested directly on `build_design`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analyze_external_target as aet


PILLARS = "ABCDEF"


def _score_frame(
    plain: dict[str, list[float]],
    geo: dict[str, list[float]],
    targets: list[str],
) -> pd.DataFrame:
    """Fabricate a per-target, per-model score frame.

    Args:
        plain: Pillar letter to its ablated contribution per target.
        geo: The same, for the geography-controlled family.
        targets: Target column names, aligned to the lists above.

    Returns:
        A frame carrying the columns the summary functions read.
    """
    rows: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        for pillar in PILLARS:
            rows.append(
                {
                    "target": target,
                    "model": f"size_emacro_drop_{pillar}",
                    "withheld_pillars": pillar,
                    "contribution_ablated": plain[pillar][index],
                    "lift_over_size": np.nan,
                    "lift_over_size_ablated": np.nan,
                }
            )
            rows.append(
                {
                    "target": target,
                    "model": f"size_geo_emacro_drop_{pillar}",
                    "withheld_pillars": pillar,
                    "contribution_ablated": geo[pillar][index],
                    "lift_over_size": np.nan,
                    "lift_over_size_ablated": np.nan,
                }
            )
        rows.append(
            {
                "target": target,
                "model": "size_emacro",
                "withheld_pillars": "",
                "contribution_ablated": np.nan,
                "lift_over_size": 0.20,
                "lift_over_size_ablated": 0.19,
            }
        )
        rows.append(
            {
                "target": target,
                "model": "size_geo",
                "withheld_pillars": "",
                "contribution_ablated": np.nan,
                "lift_over_size": 0.09,
                "lift_over_size_ablated": 0.08,
                "r2_ablated": 0.30,
            }
        )
        rows.append(
            {
                "target": target,
                "model": "size_geo_emacro",
                "withheld_pillars": "",
                "contribution_ablated": np.nan,
                "lift_over_size": np.nan,
                "lift_over_size_ablated": np.nan,
                # 0.41 - 0.30 = +0.11, the matrix's lift over a geo baseline.
                "r2_ablated": 0.41,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def two_table_scores() -> tuple[pd.DataFrame, list[str]]:
    """Four targets drawn from two ACS tables, so clustering has something to do."""
    targets = sorted(aet.TARGET_TABLES)[:4]
    plain = {pillar: [0.01 * (ord(pillar) - 64)] * 4 for pillar in PILLARS}
    geo = {pillar: [0.005 * (ord(pillar) - 64)] * 4 for pillar in PILLARS}
    return _score_frame(plain, geo, targets), targets


def test_geo_models_carry_lat_lon_and_plain_models_do_not() -> None:
    """The control is only a control if the columns actually enter the design."""
    panel = pd.DataFrame(
        {
            "log_population": [1.0, 2.0, 3.0],
            "log_agi": [1.0, 2.0, 3.0],
            "log_gdp_latest": [1.0, 2.0, 3.0],
            "lat": [30.0, 40.0, 50.0],
            "lon": [-90.0, -95.0, -100.0],
            "pillar_one": [0.0, 1.0, 2.0],
        }
    )
    pillar_columns = ["pillar_one"]

    plain = aet.build_design(panel, aet.MODELS[1], pillar_columns)
    geo = aet.build_design(
        panel, next(m for m in aet.MODELS if m.name == "size_geo"), pillar_columns
    )

    assert plain.shape[1] == len(aet.SIZE_FEATURES)
    assert geo.shape[1] == len(aet.SIZE_FEATURES) + len(aet.GEO_FEATURES)
    # The two extra columns are the coordinates themselves, in order.
    np.testing.assert_allclose(geo[:, -2:], panel[["lat", "lon"]].to_numpy())


def test_every_pillar_has_a_geo_twin_referencing_the_geo_full_model() -> None:
    """A geo contribution is only net of location if both sides carry it."""
    by_name = {model.name: model for model in aet.MODELS}
    for pillar in PILLARS:
        twin = by_name[f"size_geo_emacro_drop_{pillar}"]
        assert twin.uses_geo
        assert twin.drop_pillars == (pillar,)
        assert twin.reference == "size_geo_emacro"
        assert by_name[twin.reference].uses_geo, "reference must also hold lat/lon"


def test_geo_control_summary_reports_share_retained(two_table_scores) -> None:
    scores, targets = two_table_scores
    summary = aet.geo_control_summary(scores, targets)

    # Source A: plain +0.01, geo +0.005 by construction.
    assert summary["by_pillar"]["A"]["contribution"] == pytest.approx(0.01)
    assert summary["by_pillar"]["A"]["contribution_geo"] == pytest.approx(0.005)
    assert summary["by_pillar"]["A"]["share_retained"] == pytest.approx(0.5)
    assert summary["latlong_lift_over_size"] == pytest.approx(0.08)
    assert summary["matrix_lift_over_size"] == pytest.approx(0.19)
    assert summary["matrix_lift_over_size_geo"] == pytest.approx(0.11)


def test_share_retained_is_none_rather_than_exploding_at_zero() -> None:
    """Source A's plain contribution is -0.0000; a ratio against it is not a number."""
    targets = sorted(aet.TARGET_TABLES)[:2]
    plain = {pillar: [0.0, 0.0] for pillar in PILLARS}
    geo = {pillar: [0.02, 0.02] for pillar in PILLARS}
    summary = aet.geo_control_summary(_score_frame(plain, geo, targets), targets)

    assert summary["by_pillar"]["A"]["share_retained"] is None


def test_bootstrap_pairs_pillars_within_a_replicate(monkeypatch) -> None:
    """A pillar-difference interval must collapse when the difference is constant.

    Every pillar here differs from every other by a fixed offset on every
    target, so a paired resample can only ever produce that same offset. An
    unpaired one would draw two independent baskets and manufacture a spread.
    """
    targets = sorted(aet.TARGET_TABLES)[:6]
    plain = {
        pillar: [0.05 * index + 0.01 * (ord(pillar) - 65) for index in range(6)]
        for pillar in PILLARS
    }
    scores = _score_frame(plain, plain, targets)

    monkeypatch.setattr(aet, "N_BOOTSTRAP", 200)
    result = aet.bootstrap_drop_one(scores, targets)

    # Pairs are emitted in sorted-letter order, so A leads and its offset is
    # the negative one.
    difference = result["naive"]["pairwise"]["A_minus_B"]
    assert difference["point"] == pytest.approx(-0.01)
    assert difference["low"] == pytest.approx(-0.01)
    assert difference["high"] == pytest.approx(-0.01)


def test_bootstrap_interval_brackets_the_point_estimate(monkeypatch) -> None:
    targets = sorted(aet.TARGET_TABLES)[:8]
    rng = np.random.default_rng(0)
    plain = {pillar: list(rng.normal(0.02, 0.01, 8)) for pillar in PILLARS}
    geo = {pillar: list(rng.normal(0.01, 0.01, 8)) for pillar in PILLARS}
    scores = _score_frame(plain, geo, targets)

    monkeypatch.setattr(aet, "N_BOOTSTRAP", 500)
    result = aet.bootstrap_drop_one(scores, targets)

    for scheme in ("naive", "table_clustered"):
        for pillar in PILLARS:
            block = result[scheme]["by_pillar"][pillar]["contribution"]
            assert block["low"] <= block["point"] <= block["high"]


def test_geo_minus_plain_is_the_paired_difference(monkeypatch) -> None:
    """The reported gap must be geo minus plain on one draw, not two means differenced."""
    targets = sorted(aet.TARGET_TABLES)[:5]
    rng = np.random.default_rng(1)
    plain = {pillar: list(rng.normal(0.03, 0.02, 5)) for pillar in PILLARS}
    geo = {pillar: [value - 0.004 for value in plain[pillar]] for pillar in PILLARS}
    scores = _score_frame(plain, geo, targets)

    monkeypatch.setattr(aet, "N_BOOTSTRAP", 300)
    result = aet.bootstrap_drop_one(scores, targets)

    for pillar in PILLARS:
        gap = result["table_clustered"]["by_pillar"][pillar]["geo_minus_plain"]
        assert gap["point"] == pytest.approx(-0.004)
        assert gap["low"] == pytest.approx(-0.004)
        assert gap["high"] == pytest.approx(-0.004)


def test_bootstrap_rejects_a_basket_some_pillar_did_not_score() -> None:
    """Pairing across different baskets is the failure this guards."""
    targets = sorted(aet.TARGET_TABLES)[:3]
    plain = {pillar: [0.01, 0.02, 0.03] for pillar in PILLARS}
    scores = _score_frame(plain, plain, targets)
    scores = scores[
        ~((scores["model"] == "size_emacro_drop_C") & (scores["target"] == targets[0]))
    ]

    with pytest.raises(ValueError, match="every drop model must score every basket target"):
        aet.bootstrap_drop_one(scores, targets)


def test_clustered_draw_keeps_a_table_together() -> None:
    """Five heating-fuel shares from b25040 must move in and out as one unit."""
    targets = [
        target for target, table in aet.TARGET_TABLES.items() if table == "b25040"
    ]
    assert len(targets) > 1, "b25040 should contribute several targets"

    rng = np.random.default_rng(7)
    drawn = aet._draw_target_positions(rng, targets, cluster_by_table=True)

    # One table, so every clustered draw is the whole table, every time.
    assert sorted(drawn.tolist()) == list(range(len(targets)))


def test_headline_basket_is_still_scored() -> None:
    """The five original targets must survive the basket widening."""
    for target in aet.HEADLINE_TARGETS:
        assert target in aet.TARGET_TABLES


def test_placebo_contributions_key_off_the_plain_family_only(two_table_scores) -> None:
    """The geo twins withhold the same letters; the noise floor must not read theirs."""
    scores, _ = two_table_scores
    contributions = {
        row.withheld_pillars: float(row.contribution_ablated)
        for row in scores.itertuples()
        if row.model in (f"size_emacro_drop_{pillar}" for pillar in PILLARS)
    }
    # Plain values, not the geo family's halved ones.
    assert contributions["A"] == pytest.approx(0.01)
    assert contributions["F"] == pytest.approx(0.06)


def test_degenerate_targets_are_excluded_from_every_headline_mean() -> None:
    """A gap between two worse-than-mean fits must not be averaged into a verdict."""
    assert "no_fuel_used_share" in aet.EXCLUDED_TARGETS
    for target, reason in aet.EXCLUDED_TARGETS.items():
        assert isinstance(reason, str) and len(reason) > 40, f"{target} needs a real reason"


def test_the_two_sweeps_share_one_exclusion_list() -> None:
    """The drop-one figure and the representation section draw the same basket."""
    import analyze_source_a_representation_marginal as marginal

    assert marginal.EXCLUDED_TARGETS == aet.EXCLUDED_TARGETS


def test_drop_one_summary_survives_a_csv_round_trip(tmp_path) -> None:
    """Empty strings read back from CSV as NaN, and NaN is truthy.

    `rebuild_external_target_stats.py` reads the sweep's own CSVs, so a summary
    that treated a missing reference as present would quietly report `size`,
    `emacro`, `grand_mean` and `size_geo` as models stating a contribution --
    only on the rebuild path, never on a fresh run.
    """
    rows = [
        {"target": "t1", "model": "size", "reference_model": "",
         "withheld_pillars": "", "contribution": np.nan, "contribution_ablated": np.nan},
        {"target": "t1", "model": "size_emacro_drop_A", "reference_model": "size_emacro",
         "withheld_pillars": "A", "contribution": 0.01, "contribution_ablated": 0.008},
    ]
    frame = pd.DataFrame(rows)

    in_memory = aet.drop_one_summary(frame)

    path = tmp_path / "scores.csv"
    frame.to_csv(path, index=False)
    round_tripped = aet.drop_one_summary(pd.read_csv(path))

    assert set(in_memory) == {"size_emacro_drop_A"}
    assert set(round_tripped) == set(in_memory), "CSV round trip changed which models qualify"
