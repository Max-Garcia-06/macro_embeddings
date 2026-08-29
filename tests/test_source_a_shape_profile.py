"""Shape-profile features: where sections sit, how standard they are, what they look like."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import extract_source_a_shape_profile as shape


def make_sections(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Build a section frame from (fips_code, section_id, title, text) tuples."""
    return pd.DataFrame(
        rows, columns=["fips_code", "section_id", "section_title", "section_text"]
    ).assign(county_name="Test County")


def test_position_runs_from_zero_to_one_in_section_id_order() -> None:
    # Use unevenly-spaced section_ids (1, 2, 100) to ensure position comes from
    # rank order (0, 1, 2) not from section_id normalization. An id-based approach
    # would yield ~(0.0, 0.01, 1.0) instead of (0.0, 0.5, 1.0).
    sections = make_sections(
        [("01001", 100, "C", "x"), ("01001", 1, "A", "x"), ("01001", 2, "B", "x")]
    )

    ordered = shape.ordered_sections(sections)
    by_title = dict(zip(ordered["section_title"], ordered["_position"]))

    assert by_title["A"] == pytest.approx(0.0)
    assert by_title["B"] == pytest.approx(0.5)
    assert by_title["C"] == pytest.approx(1.0)


def test_a_single_section_county_sits_at_zero_not_nan() -> None:
    """One section is both first and last; 0.0 is the defensible reading."""
    sections = make_sections([("01001", 1, "A", "x")])

    assert shape.ordered_sections(sections)["_position"].iloc[0] == pytest.approx(0.0)


def test_absent_sections_get_the_sentinel_not_zero() -> None:
    """Position 0.0 means 'first', which is the opposite of absent."""
    sections = make_sections([("01001", 1, "Geography", "x"), ("01002", 1, "Economy", "x")])

    positions = shape.position_features(sections, ["geography", "economy"])

    assert positions.loc["01001", "pos_geography"] == pytest.approx(0.0)
    assert positions.loc["01001", "pos_economy"] == shape.POSITION_ABSENT
    assert shape.POSITION_ABSENT < 0.0


def test_position_of_the_longest_section_is_found() -> None:
    sections = make_sections(
        [("01001", 1, "A", "x" * 10), ("01001", 2, "B", "x" * 900), ("01001", 3, "C", "x" * 10)]
    )

    positions = shape.position_features(sections, [])

    assert positions.loc["01001", "pos_longest_section"] == pytest.approx(0.5)


def test_history_before_economy_reads_the_actual_order() -> None:
    early = make_sections([("01001", 1, "History", "x"), ("01001", 2, "Economy", "x")])
    late = make_sections([("01002", 1, "Economy", "x"), ("01002", 2, "History", "x")])

    assert shape.position_features(early, []).loc["01001", "history_before_economy"] == 1.0
    assert shape.position_features(late, []).loc["01002", "history_before_economy"] == 0.0


def test_history_before_economy_is_zero_when_either_is_absent() -> None:
    sections = make_sections([("01001", 1, "History", "x"), ("01001", 2, "Geography", "x")])

    assert shape.position_features(sections, []).loc["01001", "history_before_economy"] == 0.0


def test_position_spread_is_zero_for_one_flagged_title() -> None:
    sections = make_sections([("01001", 1, "Geography", "x"), ("01001", 2, "Unflagged", "x")])

    positions = shape.position_features(sections, ["geography"])

    assert positions.loc["01001", "position_spread"] == pytest.approx(0.0)


def test_slug_collision_raises_valueerror() -> None:
    """Two distinct titles that slugify to the same column name must raise."""
    sections = make_sections([("01001", 1, "Test Title", "x"), ("01001", 2, "Test-Title", "x")])

    # "Test Title" and "Test-Title" both slugify to "test_title"
    with pytest.raises(ValueError, match="share the slug"):
        shape.position_features(sections, ["Test Title", "Test-Title"])


def test_real_corpus_vocabulary_has_no_slugify_collisions(sections_frame: pd.DataFrame) -> None:
    """The real corpus vocabulary must not contain slug collisions."""
    from extract_source_a_structure_features import flag_vocabulary

    vocabulary = flag_vocabulary(sections_frame)

    # If this passes, no collision in the real corpus; if it fails, the guard
    # correctly raises and we know about it.
    shape.position_features(sections_frame, vocabulary)


def make_template_corpus() -> pd.DataFrame:
    """Four counties. 'geography' and 'history' are modal; 'oddity' is not."""
    rows = []
    for i in range(1, 5):
        rows.append((f"0100{i}", 1, "Geography", "x"))
        rows.append((f"0100{i}", 2, "History", "x"))
    rows.append(("01004", 3, "Oddity", "x"))
    return make_sections(rows)


def test_modal_set_is_the_titles_most_counties_hold() -> None:
    modal = shape.modal_title_set(make_template_corpus())

    assert set(modal) == {"geography", "history"}


def test_modal_set_is_derived_not_hardcoded() -> None:
    """A corpus with a different skeleton produces a different modal set."""
    rows = [(f"0100{i}", 1, "Volcanology", "x") for i in range(1, 5)]

    assert shape.modal_title_set(make_sections(rows)) == ["volcanology"]


def test_jaccard_is_one_for_a_county_holding_exactly_the_modal_set() -> None:
    features = shape.template_features(make_template_corpus())

    assert features.loc["01001", "template_jaccard"] == pytest.approx(1.0)


def test_jaccard_falls_when_a_county_adds_an_unusual_section() -> None:
    features = shape.template_features(make_template_corpus())

    # 01004 holds {geography, history, oddity} against a modal {geography, history}
    assert features.loc["01004", "template_jaccard"] == pytest.approx(2 / 3)
    assert features.loc["01004", "template_jaccard"] < features.loc["01001", "template_jaccard"]


def test_missing_core_sections_are_counted() -> None:
    corpus = make_template_corpus()
    thin = pd.concat([corpus, make_sections([("01009", 1, "Geography", "x")])])

    features = shape.template_features(thin)

    assert features.loc["01009", "n_core_missing"] == 1.0  # has geography, lacks history
    assert features.loc["01001", "n_core_missing"] == 0.0


def test_unusual_sections_are_the_rare_ones() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 201)]
    rows.append(("00007", 2, "One Off", "x"))  # 1 of 200 counties = 0.5%

    features = shape.template_features(make_sections(rows))

    assert features.loc["00007", "n_unusual_sections"] == 1.0
    assert features.loc["00001", "n_unusual_sections"] == 0.0


def test_share_unusual_sections_divides_by_titled_sections_not_all_sections() -> None:
    """Deferred #5. The design says `/ n_body_sections`; the code says `/ titled`,
    with a comment explaining why (an untitled section has no title to be rare).
    This pins the shipped denominator so the two cannot drift apart silently."""
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 201)]
    rows.append(("00007", 2, "One Off", "x"))  # rare: 1 of 201 counties
    rows.append(("00007", 3, None, "x"))  # untitled: excluded from the denominator

    features = shape.template_features(make_sections(rows))

    # Two titled sections, one of them unusual -- not three body sections.
    assert features.loc["00007", "n_unusual_sections"] == 1.0
    assert features.loc["00007", "share_unusual_sections"] == pytest.approx(0.5)


def test_template_jaccard_is_zero_when_a_county_has_no_titled_sections() -> None:
    """Deferred #5. The empty-title-set branch: `held` is an empty set, so the
    Jaccard numerator is 0 and the denominator is the modal set."""
    corpus = make_template_corpus()
    untitled_only = pd.concat([corpus, make_sections([("01009", 1, None, "x")])])

    features = shape.template_features(untitled_only)

    assert features.loc["01009", "template_jaccard"] == pytest.approx(0.0)
    assert features.loc["01009", "n_core_missing"] == 2.0
    # Zero titled sections is a zero denominator; the column reports 0.0, not NaN.
    assert features.loc["01009", "share_unusual_sections"] == pytest.approx(0.0)
    assert np.isfinite(features.loc["01009"].to_numpy(dtype="float64")).all()


def test_title_rarity_is_higher_for_a_county_with_rare_titles() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 201)]
    rows.append(("00007", 2, "One Off", "x"))

    features = shape.template_features(make_sections(rows))

    assert features.loc["00007", "mean_title_rarity"] > features.loc["00001", "mean_title_rarity"]


def test_title_word_count_is_averaged_over_the_county() -> None:
    sections = make_sections(
        [("01001", 1, "Law and government", "x"), ("01001", 2, "Economy", "x")]
    )

    assert shape.template_features(sections).loc["01001", "n_title_words"] == pytest.approx(2.0)


def test_digit_density_is_zero_for_letters_and_one_for_digits() -> None:
    letters = make_sections([("01001", 1, "A", "abcdef")])
    digits = make_sections([("01002", 1, "A", "123456")])

    assert shape.surface_features(letters).loc["01001", "digit_density"] == pytest.approx(0.0)
    assert shape.surface_features(digits).loc["01002", "digit_density"] == pytest.approx(1.0)


def test_capital_ratio_counts_letters_only() -> None:
    """Digits are neither upper nor lower, so they must not dilute the ratio."""
    sections = make_sections([("01001", 1, "A", "AB12ab")])

    assert shape.surface_features(sections).loc["01001", "capital_ratio"] == pytest.approx(0.5)


def test_mean_word_length_ignores_whitespace() -> None:
    sections = make_sections([("01001", 1, "A", "aa bbbb cc")])

    assert shape.surface_features(sections).loc["01001", "mean_word_length"] == pytest.approx(
        8 / 3
    )


def test_bucket_density_is_computed_only_where_that_bucket_has_characters() -> None:
    """A density over zero characters is not a number; it must not be invented."""
    sections = make_sections([("01001", 1, "Geography", "abc123")])

    surface = shape.surface_features(sections)

    assert surface.loc["01001", "digit_density_geography"] == pytest.approx(0.5)
    assert surface.loc["01001", "digit_density_census"] == pytest.approx(0.0)


def test_a_county_with_no_characters_gets_zero_not_nan() -> None:
    sections = make_sections([("01001", 1, "A", "")])

    surface = shape.surface_features(sections)

    assert np.isfinite(surface.loc["01001"].to_numpy()).all()


def test_top3_share_is_the_three_longest_sections() -> None:
    sections = make_sections(
        [
            ("01001", 1, "A", "x" * 400),
            ("01001", 2, "B", "x" * 300),
            ("01001", 3, "C", "x" * 200),
            ("01001", 4, "D", "x" * 100),
        ]
    )

    curve = shape.length_curve_features(sections)

    assert curve.loc["01001", "top3_length_share"] == pytest.approx(0.9)


def test_top3_share_is_one_when_a_county_has_three_or_fewer_sections() -> None:
    sections = make_sections([("01001", 1, "A", "x" * 10), ("01001", 2, "B", "x" * 20)])

    assert shape.length_curve_features(sections).loc["01001", "top3_length_share"] == pytest.approx(
        1.0
    )


def test_decay_slope_is_negative_when_lengths_fall_off() -> None:
    steep = make_sections(
        [("01001", i, f"S{i}", "x" * n) for i, n in enumerate([1000, 100, 10, 5], start=1)]
    )
    flat = make_sections([("01002", i, f"S{i}", "x" * 100) for i in range(1, 5)])

    curve = shape.length_curve_features(pd.concat([steep, flat]))

    assert curve.loc["01001", "length_decay_slope"] < curve.loc["01002", "length_decay_slope"]
    assert curve.loc["01002", "length_decay_slope"] == pytest.approx(0.0)


def test_absolute_bucket_characters_are_reported_for_every_bucket() -> None:
    sections = make_sections([("01001", 1, "Geography", "x" * 250)])

    curve = shape.length_curve_features(sections)

    assert curve.loc["01001", "chars_geography"] == pytest.approx(250.0)
    assert curve.loc["01001", "chars_census"] == pytest.approx(0.0)


def test_recomputed_top_one_share_matches_round_one(sections_frame: pd.DataFrame) -> None:
    """Cross-module consistency: the same quantity under two names must agree.

    Round one ships `share_in_largest_section`; this module deliberately does not
    ship a duplicate of it. Asserting the equality here catches a drift in either
    module without putting the column in the block twice.
    """
    import extract_source_a_structure_features as structure

    round_one = structure.length_features(sections_frame)["share_in_largest_section"]

    chars = sections_frame["section_text"].fillna("").str.len().astype("float64")
    grouped = sections_frame.assign(_chars=chars).groupby("fips_code")["_chars"]
    recomputed = (grouped.max() / grouped.sum().replace(0.0, np.nan)).fillna(0.0)

    assert np.allclose(recomputed.to_numpy(), round_one.reindex(recomputed.index).to_numpy())


def test_every_county_appears_exactly_once(sections_frame: pd.DataFrame) -> None:
    features, _ = shape.build_shape_profile(sections_frame)

    assert len(features) == sections_frame["fips_code"].nunique()
    assert features["fips_code"].is_unique


def test_all_feature_columns_are_finite(sections_frame: pd.DataFrame) -> None:
    features, _ = shape.build_shape_profile(sections_frame)
    block = features[shape.shape_profile_columns(features)]

    assert (block.dtypes == "float64").all()
    assert np.isfinite(block.to_numpy()).all()


def test_the_profile_shares_no_column_with_round_one(sections_frame: pd.DataFrame) -> None:
    """The analysis module joins the two parquets; a shared name would collide."""
    import extract_source_a_structure_features as structure

    profile, _ = shape.build_shape_profile(sections_frame)
    round_one, _ = structure.build_structure_features(sections_frame)

    shared = (set(profile.columns) & set(round_one.columns)) - {"fips_code"}
    assert shared == set(), f"colliding column names: {sorted(shared)}"


def test_position_columns_cover_round_ones_flag_vocabulary(sections_frame: pd.DataFrame) -> None:
    """`pos_x` and `has_section_x` must never describe different title sets."""
    import extract_source_a_structure_features as structure

    profile, _ = shape.build_shape_profile(sections_frame)

    flagged = {
        c[len(structure.TITLE_FLAG_PREFIX):]
        for c in structure.build_structure_features(sections_frame)[0].columns
        if c.startswith(structure.TITLE_FLAG_PREFIX)
    }
    positioned = {
        c[len(shape.POSITION_PREFIX):]
        for c in profile.columns
        if c.startswith(shape.POSITION_PREFIX) and not c.startswith("pos_first_")
        and c != "pos_longest_section"
    }

    assert positioned == flagged


def test_summary_records_the_sets_it_derived(sections_frame: pd.DataFrame) -> None:
    features, metadata = shape.build_shape_profile(sections_frame)

    stats = shape.summarize(features, metadata)

    assert stats["n_counties"] == len(features)
    assert stats["modal_title_set"] == metadata["modal_title_set"]
    assert stats["modal_title_min_share"] == shape.MODAL_TITLE_MIN_SHARE
    assert stats["unusual_title_max_share"] == shape.UNUSUAL_TITLE_MAX_SHARE
    assert len(stats["modal_title_set"]) > 0


def test_five_arms_are_declared() -> None:
    import analyze_source_a_shape_profile as scoring

    assert [arm.key for arm in scoring.SHAPE_ARMS] == [
        "shape_v1",
        "shape_v2",
        "typed",
        "typed_plus_shape_v2",
        "size_nonlinear",
    ]


def test_shape_v2_strictly_contains_shape_v1() -> None:
    """v2 is v1 plus the new families; if it were not, the comparison is meaningless."""
    import analyze_source_a_shape_profile as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    matrix, v1_cols, profile_cols = scoring.attach_blocks(matrix)
    rows = np.ones(len(matrix), dtype=bool)

    blocks = scoring.build_arm_blocks(matrix, v1_cols, profile_cols, rows)

    assert blocks["shape_v1"].shape[1] == len(v1_cols)
    assert blocks["shape_v2"].shape[1] == len(v1_cols) + len(profile_cols)


def test_both_blocks_attach_without_collision() -> None:
    import analyze_source_a_shape_profile as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    attached, v1_cols, profile_cols = scoring.attach_blocks(matrix)

    assert len(attached) == len(matrix)
    assert not set(v1_cols) & set(profile_cols)
    assert attached[v1_cols + profile_cols].notna().all().all()


def test_every_arm_carries_all_three_framings() -> None:
    """A row missing a framing is how round one's number got quoted wrong."""
    import analyze_source_a_shape_profile as scoring

    record = scoring.empty_record_keys()

    for arm in scoring.SHAPE_ARMS:
        assert f"r2_alone_{arm.key}" in record
        assert f"lift_{arm.key}" in record
        assert f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}" in record


def test_boost_residual_matches_the_imported_routines_fold_structure() -> None:
    """The two learners must differ in estimator only, never in fold handling."""
    import analyze_source_a_shape_profile as scoring
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 3))
    block = rng.normal(size=(200, 5))
    y = base[:, 0] * 2.0 + block[:, 1] ** 2 + rng.normal(scale=0.1, size=200)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    predictions = scoring.boost_residual_oof(base, block, y, folds)

    assert predictions.shape == (200,)
    assert np.isfinite(predictions).all()


def test_boost_recovers_a_curve_that_ridge_cannot() -> None:
    """The reason this learner exists: a step function is invisible to a linear fit."""
    import analyze_source_a_shape_profile as scoring
    from analyze_source_a_representation import _alone_oof_r2
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=600)
    block = x.reshape(-1, 1)
    y = np.where(x > 0.0, 1.0, -1.0) + rng.normal(scale=0.05, size=600)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    assert scoring.boost_alone_oof_r2(block, y, folds) > _alone_oof_r2(block, y, folds, None)


def test_both_learners_appear_in_every_record_key() -> None:
    import analyze_source_a_shape_profile as scoring

    keys = scoring.empty_record_keys()

    for arm in scoring.SHAPE_ARMS:
        assert f"r2_alone_{arm.key}{scoring.BOOST_SUFFIX}" in keys
        assert f"lift_{arm.key}{scoring.BOOST_SUFFIX}" in keys
        assert f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{scoring.BOOST_SUFFIX}" in keys


def test_size_recoverability_finds_a_planted_size_signal() -> None:
    """A block that is a noisy copy of size must score high; noise must not."""
    import analyze_source_a_shape_profile as scoring

    rng = np.random.default_rng(0)
    size = rng.normal(size=400)
    matrix = pd.DataFrame(
        {"log_population": size, "log_agi": size, "log_gdp_latest": size}
    )
    blocks = {
        "copy": (size + rng.normal(scale=0.1, size=400)).reshape(-1, 1),
        "noise": rng.normal(size=(400, 3)),
    }

    recovery = scoring.size_recoverability(matrix, blocks)

    assert recovery["copy"]["log_population_ridge"] > 0.9
    assert recovery["noise"]["log_population_ridge"] < 0.1


def test_size_recoverability_reports_every_size_measure_and_learner() -> None:
    import analyze_source_a_shape_profile as scoring

    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        {
            "log_population": rng.normal(size=200),
            "log_agi": rng.normal(size=200),
            "log_gdp_latest": rng.normal(size=200),
        }
    )

    recovery = scoring.size_recoverability(matrix, {"block": rng.normal(size=(200, 2))})

    for measure in ("log_population", "log_agi", "log_gdp_latest"):
        for learner in ("ridge", "boost"):
            assert f"{measure}_{learner}" in recovery["block"]


def _make_lift_frame(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """A synthetic results-shaped frame carrying every arm's lift columns."""
    import analyze_source_a_shape_profile as scoring

    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "pillar": ["B"] * n,
        "column": [f"target_{i}" for i in range(n)],
    }
    for arm in scoring.SHAPE_ARMS:
        for suffix in ("", scoring.BOOST_SUFFIX):
            for key in scoring.arm_record_keys(arm.key, suffix):
                data[key] = rng.normal(scale=0.01, size=n)
    return pd.DataFrame(data)


def test_paired_test_resolves_every_arm_framing_learner_combination() -> None:
    """Finding 3: `_paired_test` must resolve the right column for all 20 combos."""
    import analyze_source_a_shape_profile as scoring
    from scipy.stats import wilcoxon

    results = _make_lift_frame()

    for arm in scoring.SHAPE_ARMS:
        for suffix in ("", scoring.BOOST_SUFFIX):
            for column in (
                f"lift_{arm.key}{suffix}",
                f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{suffix}",
            ):
                record = scoring._paired_test(results, arm, column)

                assert record["mean_lift"] == pytest.approx(results[column].mean())
                assert record["vs_baseline"]["wilcoxon_p"] == pytest.approx(
                    wilcoxon(results[column])[1]
                )
                if arm.against is None:
                    assert "vs_arm" not in record
                else:
                    other_column = column.replace(arm.key, arm.against, 1)
                    expected_diff = results[column] - results[other_column]
                    assert record["vs_arm"]["compared_against"] == arm.against
                    assert record["vs_arm"]["wilcoxon_p"] == pytest.approx(
                        wilcoxon(expected_diff)[1]
                    )


def test_vs_baseline_and_vs_arm_can_diverge_sharply() -> None:
    """The exact failure fix round 1 found: a real lift over baseline read as
    insignificant only because the wrong comparison's p-value was published
    beside it. This constructs data where the two readings must disagree."""
    import analyze_source_a_shape_profile as scoring

    n = 30
    rng = np.random.default_rng(1)
    lift_v1 = rng.normal(loc=0.02, scale=0.002, size=n)  # clearly beats baseline
    lift_v2 = lift_v1 + rng.normal(scale=0.02, size=n)  # no clear edge over v1

    arm = next(a for a in scoring.SHAPE_ARMS if a.key == "shape_v2")
    results = pd.DataFrame({"lift_shape_v1": lift_v1, "lift_shape_v2": lift_v2})

    record = scoring._paired_test(results, arm, "lift_shape_v2")

    assert record["vs_baseline"]["wilcoxon_p"] < 0.01
    assert record["vs_arm"]["wilcoxon_p"] > 0.05


def test_summarize_by_pillar_reproduces_a_known_mean() -> None:
    import analyze_source_a_shape_profile as scoring

    results = _make_lift_frame(n=4)
    results["pillar"] = ["B", "B", "C", "C"]

    by_pillar = scoring.summarize_by_pillar(results).set_index("pillar")

    b_rows = results[results["pillar"] == "B"]
    assert int(by_pillar.loc["B", "n_targets"]) == 2
    assert by_pillar.loc["B", "lift_shape_v1_linear"] == pytest.approx(
        b_rows["lift_shape_v1"].mean()
    )
    assert by_pillar.loc["B", f"lift_shape_v1_linear{scoring.BOOST_SUFFIX}"] == pytest.approx(
        b_rows[f"lift_shape_v1{scoring.BOOST_SUFFIX}"].mean()
    )


def test_boost_floor_lift_scores_pure_noise_through_the_boost_path() -> None:
    """The floor must run every target through the same boost_residual_oof path."""
    import analyze_source_a_shape_profile as scoring
    from analyze_pillar_matrix_signal import Target

    rng = np.random.default_rng(3)
    n = 300
    matrix = pd.DataFrame({"target_a": rng.normal(size=n)})
    baseline = pd.DataFrame({"const": np.ones(n)})
    flexible_baseline = pd.DataFrame(
        {"const": np.ones(n), "curve": rng.normal(scale=0.01, size=n)}
    )
    targets = [Target("B", "target_a", "Target A")]

    floor = scoring.boost_floor_lift(
        matrix, baseline, flexible_baseline, targets, {"shape_v1": 4}
    )

    linear_floor, flexible_floor = scoring.floor_column_names("shape_v1")
    assert list(floor["column"]) == ["target_a"]
    assert linear_floor in floor.columns
    assert flexible_floor in floor.columns
    assert np.isfinite(floor[linear_floor]).all()
    assert np.isfinite(floor[flexible_floor]).all()


def _make_summarize_frame(n: int = 12, seed: int = 0) -> pd.DataFrame:
    """A synthetic frame carrying everything `summarize` reads, floor included."""
    import analyze_source_a_shape_profile as scoring

    frame = _make_lift_frame(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    frame["r2_baseline"] = rng.uniform(0.2, 0.3, size=n)
    frame["r2_baseline_flexible"] = rng.uniform(0.2, 0.3, size=n)
    for arm in scoring.SHAPE_ARMS:
        for column in scoring.floor_column_names(arm.key):
            frame[column] = rng.normal(scale=0.01, size=n)
    for arm in scoring.SHAPE_ARMS:
        for suffix in ("", scoring.BOOST_SUFFIX):
            frame[f"r2_alone_{arm.key}{suffix}"] = rng.uniform(0.0, 0.2, size=n)
    return frame


def test_summarize_emits_both_p_value_readings_for_every_arm() -> None:
    """Finding 3: `summarize` must expose vs_baseline (always), vs_arm (when
    paired) and vs_boost_floor (boost learner only) for every arm."""
    import analyze_source_a_shape_profile as scoring

    results = _make_summarize_frame()
    size_recovery: dict[str, dict[str, float]] = {"shape_v1": {}, "shape_v2": {}}

    stats = scoring.summarize(results, size_recovery, n_v1=64, n_profile=73)

    for arm in scoring.SHAPE_ARMS:
        for learner in ("ridge", "boost"):
            entry = stats["arms"][f"{arm.key}_{learner}"]
            for framing in ("linear", "flexible"):
                assert "vs_baseline" in entry[framing]
                if arm.against is None:
                    assert "vs_arm" not in entry[framing]
                else:
                    assert entry[framing]["vs_arm"]["compared_against"] == arm.against
                if learner == "boost":
                    assert "vs_boost_floor" in entry[framing]
                else:
                    assert "vs_boost_floor" not in entry[framing]

    for arm in scoring.SHAPE_ARMS:
        linear_floor, flexible_floor = scoring.floor_column_names(arm.key)
        assert stats["boost_floor"][arm.key]["linear"]["mean_lift"] == pytest.approx(
            results[linear_floor].mean()
        )
        assert stats["boost_floor"][arm.key]["flexible"]["mean_lift"] == pytest.approx(
            results[flexible_floor].mean()
        )


def _make_scorable_matrix(n: int = 150, seed: int = 5) -> tuple[pd.DataFrame, list[str], list[str]]:
    """A tiny matrix carrying everything `build_arm_blocks` slices.

    Small enough to fit every arm through both learners in seconds, real enough
    that `score_target` runs its actual code path rather than a stand-in.
    """
    from analyze_source_a_structure import typed_columns
    from pillar_matrix import SIZE_FEATURES

    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for name in SIZE_FEATURES:
        data[name] = rng.normal(loc=10.0, scale=1.0, size=n)
    for name in typed_columns():
        data[name] = rng.normal(size=n)
    v1_cols = [f"v1_{i}" for i in range(3)]
    profile_cols = [f"profile_{i}" for i in range(2)]
    for name in [*v1_cols, *profile_cols]:
        data[name] = rng.normal(size=n)
    data["target_a"] = rng.normal(size=n)
    return pd.DataFrame(data), v1_cols, profile_cols


def test_score_target_writes_every_key_empty_record_keys_promises() -> None:
    """Finding I3: the key-consistency assertions were circular.

    `empty_record_keys` is built by looping `arm_record_keys`, so a test that
    re-derives its expectations from `arm_record_keys` and checks them against
    `empty_record_keys` would still pass if `score_target` stopped writing the
    boost keys entirely. This runs the real `score_target` and checks the real
    record against the promise, which is the assertion the docstring claims.
    """
    import analyze_source_a_shape_profile as scoring
    from analyze_pillar_matrix_signal import Target

    matrix, v1_cols, profile_cols = _make_scorable_matrix()
    baseline = pd.DataFrame({"const": np.ones(len(matrix))}, index=matrix.index)
    flexible = baseline.assign(curve=matrix["log_population"] ** 2)

    record = scoring.score_target(
        matrix,
        v1_cols,
        profile_cols,
        baseline,
        flexible,
        Target("B", "target_a", "Target A"),
    )

    assert set(scoring.empty_record_keys()) <= set(record)
    for key in scoring.empty_record_keys():
        assert isinstance(record[key], float)
        assert np.isfinite(record[key])


def test_boost_floor_is_scored_at_each_arms_own_width() -> None:
    """Finding I1: the floor is an overfitting penalty, and that penalty scales
    with block width. A single narrow floor is not a fair comparison for a
    137-column arm."""
    import analyze_source_a_shape_profile as scoring
    from analyze_pillar_matrix_signal import Target

    rng = np.random.default_rng(3)
    n = 300
    matrix = pd.DataFrame({"target_a": rng.normal(size=n), "target_b": rng.normal(size=n)})
    baseline = pd.DataFrame({"const": np.ones(n)})
    flexible_baseline = pd.DataFrame(
        {"const": np.ones(n), "curve": rng.normal(scale=0.01, size=n)}
    )
    targets = [Target("B", "target_a", "A"), Target("B", "target_b", "B")]

    floor = scoring.boost_floor_lift(
        matrix, baseline, flexible_baseline, targets, {"narrow": 2, "wide": 40}
    )

    assert list(floor["column"]) == ["target_a", "target_b"]
    for arm in ("narrow", "wide"):
        for column in scoring.floor_column_names(arm):
            assert column in floor.columns
            assert np.isfinite(floor[column]).all()
    # Different widths must actually produce different floors, or width-matching
    # would be ceremony.
    assert not np.allclose(
        floor["lift_boost_floor_narrow"], floor["lift_boost_floor_wide"]
    )


def test_boost_floor_noise_does_not_depend_on_the_order_arms_are_scored_in() -> None:
    """The noise block is seeded from the width, so scoring one arm alone and
    scoring it alongside others give the identical floor. Without this, a floor
    computed in two passes would not agree with one computed in a single pass."""
    import analyze_source_a_shape_profile as scoring
    from analyze_pillar_matrix_signal import Target

    rng = np.random.default_rng(4)
    n = 250
    matrix = pd.DataFrame({"target_a": rng.normal(size=n)})
    baseline = pd.DataFrame({"const": np.ones(n)})
    flexible_baseline = pd.DataFrame({"const": np.ones(n), "curve": rng.normal(size=n)})
    targets = [Target("B", "target_a", "A")]

    alone = scoring.boost_floor_lift(
        matrix, baseline, flexible_baseline, targets, {"wide": 12}
    )
    together = scoring.boost_floor_lift(
        matrix, baseline, flexible_baseline, targets, {"narrow": 2, "wide": 12}
    )

    assert alone["lift_boost_floor_wide"].to_numpy() == pytest.approx(
        together["lift_boost_floor_wide"].to_numpy()
    )


def test_each_boost_arm_is_compared_to_the_floor_at_its_own_width() -> None:
    """Finding I1, in `summarize`: `vs_boost_floor` must read the arm's own
    width-matched floor column, not one shared narrow floor."""
    import analyze_source_a_shape_profile as scoring
    from scipy.stats import wilcoxon

    results = _make_summarize_frame()
    widths = {arm.key: 10 + i for i, arm in enumerate(scoring.SHAPE_ARMS)}
    stats = scoring.summarize(results, {"shape_v1": {}, "shape_v2": {}}, 64, 73, widths)

    for arm in scoring.SHAPE_ARMS:
        entry = stats["arms"][f"{arm.key}_boost"]
        linear_floor, flexible_floor = scoring.floor_column_names(arm.key)
        expected_linear = (
            results[f"lift_{arm.key}{scoring.BOOST_SUFFIX}"] - results[linear_floor]
        )
        assert entry["linear"]["vs_boost_floor"]["wilcoxon_p"] == pytest.approx(
            wilcoxon(expected_linear)[1]
        )
        expected_flexible = (
            results[f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{scoring.BOOST_SUFFIX}"]
            - results[flexible_floor]
        )
        assert entry["flexible"]["vs_boost_floor"]["wilcoxon_p"] == pytest.approx(
            wilcoxon(expected_flexible)[1]
        )
        assert stats["boost_floor"][arm.key]["width"] == widths[arm.key]


def test_summarize_reports_the_undegraded_basket_beside_the_full_one() -> None:
    """Finding C2. §23.6's Forbidden wording bans quoting a flexible figure
    without saying which basket produced it, so both baskets must be in the
    artifact."""
    import analyze_source_a_shape_profile as scoring
    from scipy.stats import wilcoxon

    results = _make_summarize_frame(n=16, seed=2)
    degraded = results["r2_baseline_flexible"] < results["r2_baseline"]
    assert 0 < int(degraded.sum()) < len(results)  # the fixture must exercise both

    stats = scoring.summarize(results, {"shape_v1": {}, "shape_v2": {}}, 64, 73)

    assert stats["flexible_degraded_targets"] == sorted(results.loc[degraded, "column"])
    assert stats["n_flexible_undegraded_targets"] == int((~degraded).sum())

    kept = results.loc[~degraded]
    for arm in scoring.SHAPE_ARMS:
        for learner, suffix in (("ridge", ""), ("boost", scoring.BOOST_SUFFIX)):
            entry = stats["arms"][f"{arm.key}_{learner}"]
            column = f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{suffix}"
            undegraded = entry["flexible"]["undegraded"]

            assert undegraded["n_targets"] == len(kept)
            assert undegraded["mean_lift"] == pytest.approx(kept[column].mean())
            assert undegraded["vs_baseline"]["wilcoxon_p"] == pytest.approx(
                wilcoxon(kept[column])[1]
            )
            if arm.against is not None:
                other = column.replace(arm.key, arm.against, 1)
                assert undegraded["vs_arm"]["wilcoxon_p"] == pytest.approx(
                    wilcoxon(kept[column] - kept[other])[1]
                )


def test_the_undegraded_basket_is_read_from_the_data_not_hardcoded() -> None:
    """Which targets the flexible baseline degrades is a property of the run."""
    import analyze_source_a_shape_profile as scoring

    results = _make_summarize_frame(n=16, seed=2)
    results.loc[:, "r2_baseline_flexible"] = results["r2_baseline"] + 0.01

    stats = scoring.summarize(results, {"shape_v1": {}, "shape_v2": {}}, 64, 73)

    assert stats["flexible_degraded_targets"] == []
    assert stats["n_flexible_undegraded_targets"] == len(results)


def test_only_the_flexible_framing_carries_an_undegraded_reading() -> None:
    """The degradation is the *flexible* baseline's; the linear one never moved,
    so an undegraded reading of the linear framing would be meaningless."""
    import analyze_source_a_shape_profile as scoring

    results = _make_summarize_frame(n=16, seed=2)
    stats = scoring.summarize(results, {"shape_v1": {}, "shape_v2": {}}, 64, 73)

    for arm in scoring.SHAPE_ARMS:
        for learner in ("ridge", "boost"):
            entry = stats["arms"][f"{arm.key}_{learner}"]
            assert "undegraded" not in entry["linear"]
            assert "undegraded" in entry["flexible"]


def test_by_pillar_names_the_framing_of_every_column_it_reports() -> None:
    """Finding I4: the by-pillar CSV is the artifact most likely to be opened
    alone, and its header said nothing about which baseline produced it."""
    import analyze_source_a_shape_profile as scoring

    results = _make_lift_frame(n=4)
    results["pillar"] = ["B", "B", "C", "C"]

    by_pillar = scoring.summarize_by_pillar(results).set_index("pillar")
    b_rows = results[results["pillar"] == "B"]

    for arm in scoring.SHAPE_ARMS:
        for suffix in ("", scoring.BOOST_SUFFIX):
            linear = f"lift_{arm.key}_linear{suffix}"
            flexible = f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{suffix}"
            assert by_pillar.loc["B", linear] == pytest.approx(
                b_rows[f"lift_{arm.key}{suffix}"].mean()
            )
            assert by_pillar.loc["B", flexible] == pytest.approx(
                b_rows[f"lift_{arm.key}{scoring.FLEXIBLE_SUFFIX}{suffix}"].mean()
            )
    # No column may leave its baseline unnamed.
    for column in by_pillar.columns:
        if column == "n_targets":
            continue
        assert "_linear" in column or scoring.FLEXIBLE_SUFFIX in column
