"""Structural features derived from Wikipedia section titles and lengths."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import extract_source_a_structure_features as structure


def make_sections(rows: list[tuple[str, int, str, str]]) -> pd.DataFrame:
    """Build a section frame from (fips_code, section_id, title, text) tuples."""
    return pd.DataFrame(
        rows, columns=["fips_code", "section_id", "section_title", "section_text"]
    ).assign(county_name="Test County")


def test_titles_are_stripped_and_casefolded() -> None:
    sections = make_sections([("01001", 1, "  Demographics ", "x"), ("01001", 2, "ECONOMY", "y")])

    assert list(structure.normalize_titles(sections)) == ["demographics", "economy"]


def test_counts_cover_sections_titles_and_blanks() -> None:
    sections = make_sections(
        [
            ("01001", 1, "History", "a"),
            ("01001", 2, "History", "b"),
            ("01001", 3, "   ", "c"),
        ]
    )

    counts = structure.count_features(sections)

    assert counts.loc["01001", "n_body_sections"] == 3
    assert counts.loc["01001", "n_distinct_titles"] == 2  # "history" and ""
    assert counts.loc["01001", "n_untitled_sections"] == 1


def test_id_gaps_are_zero_when_ids_are_contiguous() -> None:
    sections = make_sections([("01001", i, f"S{i}", "x") for i in range(1, 6)])

    assert structure.count_features(sections).loc["01001", "n_id_gaps"] == 0


def test_id_gaps_count_skipped_parsoid_ids() -> None:
    """Parsoid numbers nested sections it does not emit; the gap is the signal."""
    sections = make_sections([("01001", 1, "A", "x"), ("01001", 2, "B", "y"), ("01001", 9, "C", "z")])

    assert structure.count_features(sections).loc["01001", "n_id_gaps"] == 6


def test_length_summary_uses_character_counts() -> None:
    sections = make_sections([("01001", 1, "A", "a" * 100), ("01001", 2, "B", "b" * 300)])

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "total_body_chars"] == 400
    assert lengths.loc["01001", "mean_section_chars"] == 200
    assert lengths.loc["01001", "max_section_chars"] == 300
    assert lengths.loc["01001", "share_in_largest_section"] == pytest.approx(0.75)


def test_stub_threshold_splits_at_200_characters() -> None:
    sections = make_sections(
        [("01001", 1, "A", "a" * 199), ("01001", 2, "B", "b" * 200), ("01001", 3, "C", "c" * 400)]
    )

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "n_stub_sections"] == 1
    assert lengths.loc["01001", "share_stub_sections"] == pytest.approx(1 / 3)


def test_single_section_county_has_zero_spread_not_nan() -> None:
    """A one-section county has an undefined sample sd; it must not reach the model as NaN."""
    sections = make_sections([("01001", 1, "A", "a" * 500)])

    lengths = structure.length_features(sections)

    assert lengths.loc["01001", "sd_section_chars"] == 0.0
    assert lengths.loc["01001", "section_length_gini"] == 0.0


def test_gini_is_zero_for_equal_sections() -> None:
    assert structure.gini(np.array([300.0, 300.0, 300.0])) == pytest.approx(0.0)


def test_gini_rises_when_one_section_dominates() -> None:
    even = structure.gini(np.array([100.0, 100.0, 100.0, 100.0]))
    lopsided = structure.gini(np.array([10.0, 10.0, 10.0, 5000.0]))

    assert lopsided > even
    assert lopsided < 1.0


def test_vocabulary_keeps_titles_above_the_share_floor() -> None:
    """Four counties: 'geography' is in three, 'quirk' in one. The floor is 5%."""
    rows = [(f"0100{i}", 1, "Geography", "x") for i in range(1, 4)]
    rows.append(("01004", 1, "Quirk", "x"))
    rows.append(("01004", 2, "Geography", "x"))

    vocabulary = structure.flag_vocabulary(make_sections(rows))

    assert "geography" in vocabulary
    assert "quirk" in vocabulary  # 1 of 4 counties = 25%, above the floor


def test_vocabulary_drops_a_title_below_the_share_floor() -> None:
    rows = [(f"{i:05d}", 1, "Geography", "x") for i in range(1, 41)]
    rows.append(("00007", 2, "One Off", "x"))  # 1 of 40 counties = 2.5%

    vocabulary = structure.flag_vocabulary(make_sections(rows))

    assert "geography" in vocabulary
    assert "one off" not in vocabulary


def test_vocabulary_is_derived_not_hardcoded() -> None:
    """A corpus with a different title distribution produces different flags."""
    rows = [(f"0100{i}", 1, "Volcanology", "x") for i in range(1, 5)]

    assert structure.flag_vocabulary(make_sections(rows)) == ["volcanology"]


def test_vocabulary_is_ordered_by_county_count() -> None:
    rows = [(f"0100{i}", 1, "Geography", "x") for i in range(1, 5)]
    rows += [(f"0100{i}", 2, "Economy", "x") for i in range(1, 3)]

    assert structure.flag_vocabulary(make_sections(rows)) == ["geography", "economy"]


def test_untitled_sections_do_not_enter_the_vocabulary() -> None:
    rows = [(f"0100{i}", 1, "   ", "x") for i in range(1, 5)]

    assert structure.flag_vocabulary(make_sections(rows)) == []


def test_slugify_produces_a_valid_column_name() -> None:
    assert structure.slugify("2020 census") == "2020_census"
    assert structure.slugify("law and government") == "law_and_government"
    assert structure.slugify("census-designated places") == "census_designated_places"


def test_slugify_survives_a_title_with_no_usable_characters() -> None:
    """2,009 sections are untitled; slugification must not produce an empty column name."""
    assert structure.slugify("") == "untitled"
    assert structure.slugify("---") == "untitled"


def test_two_titles_slugifying_the_same_way_raise_instead_of_overwriting() -> None:
    """A silent overwrite would report N vocabulary entries against N-1 flag columns."""
    rows = [("01001", 1, "2020 census", "x"), ("01001", 2, "2020-census", "y")]

    with pytest.raises(ValueError, match="slug"):
        structure.title_flag_features(make_sections(rows), ["2020 census", "2020-census"])


def test_the_real_corpus_vocabulary_has_no_slug_collisions(sections_frame: pd.DataFrame) -> None:
    """The guard above must not be firing on the corpus this round actually scored."""
    vocabulary = structure.flag_vocabulary(sections_frame)

    assert len({structure.slugify(title) for title in vocabulary}) == len(vocabulary)


def test_flags_are_binary_per_county() -> None:
    rows = [
        ("01001", 1, "Geography", "x"),
        ("01001", 2, "Geography", "x"),  # twice in one county is still one flag
        ("01002", 1, "Economy", "x"),
    ]

    flags = structure.title_flag_features(make_sections(rows), ["geography", "economy"])

    assert flags.loc["01001", "has_section_geography"] == 1.0
    assert flags.loc["01001", "has_section_economy"] == 0.0
    assert flags.loc["01002", "has_section_geography"] == 0.0
    assert set(flags["has_section_geography"].unique()) <= {0.0, 1.0}


def test_bucket_shares_sum_to_one_for_every_county(sections_frame: pd.DataFrame) -> None:
    shares = structure.bucket_share_features(sections_frame)

    assert len(shares) == sections_frame["fips_code"].nunique()
    assert np.allclose(shares.sum(axis=1).to_numpy(), 1.0)


def test_census_wins_the_population_ranking_collision() -> None:
    """'population ranking' matches both the census and list patterns."""
    sections = make_sections([("01001", 1, "Population ranking", "x" * 100)])

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_census"] == pytest.approx(1.0)
    assert shares.loc["01001", "share_chars_lists"] == pytest.approx(0.0)


def test_transportation_is_a_highway_not_an_economy_section() -> None:
    sections = make_sections([("01001", 1, "Transportation", "x" * 100)])

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_highways"] == pytest.approx(1.0)
    assert shares.loc["01001", "share_chars_economy"] == pytest.approx(0.0)


def test_highways_are_their_own_bucket_not_folded_into_lists() -> None:
    sections = make_sections(
        [("01001", 1, "Major highways", "x" * 100), ("01001", 2, "Communities", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_highways"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_lists"] == pytest.approx(0.5)


def test_shares_are_weighted_by_characters_not_section_count() -> None:
    sections = make_sections(
        [("01001", 1, "Demographics", "x" * 900), ("01001", 2, "Economy", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_census"] == pytest.approx(0.9)
    assert shares.loc["01001", "share_chars_economy"] == pytest.approx(0.1)


def test_unmatched_titles_fall_to_other() -> None:
    sections = make_sections([("01001", 1, "Volcanology", "x" * 100)])

    assert structure.bucket_share_features(sections).loc["01001", "share_chars_other"] == pytest.approx(1.0)


def test_geography_and_government_are_split_out_of_other() -> None:
    sections = make_sections(
        [("01001", 1, "Geography", "x" * 100), ("01001", 2, "Government", "y" * 100)]
    )

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001", "share_chars_geography"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_government"] == pytest.approx(0.5)
    assert shares.loc["01001", "share_chars_other"] == pytest.approx(0.0)


def test_other_is_not_the_largest_bucket_in_the_real_corpus(sections_frame: pd.DataFrame) -> None:
    """Splitting geography and government out of `other` is the point of doing it."""
    corpus_share = structure.bucket_share_features(sections_frame).mean()

    assert corpus_share["share_chars_other"] < corpus_share.drop("share_chars_other").max()


def test_a_county_with_no_characters_still_sums_to_one() -> None:
    """The invariant is unconditional, not a property of the current corpus."""
    sections = make_sections([("01001", 1, "Geography", ""), ("01001", 2, "Economy", "")])

    shares = structure.bucket_share_features(sections)

    assert shares.loc["01001"].sum() == pytest.approx(1.0)
    assert shares.loc["01001", "share_chars_other"] == pytest.approx(1.0)


def test_every_county_appears_exactly_once(sections_frame: pd.DataFrame) -> None:
    features, _ = structure.build_structure_features(sections_frame)

    assert len(features) == sections_frame["fips_code"].nunique()
    assert features["fips_code"].is_unique
    assert set(features["fips_code"]) == set(sections_frame["fips_code"])


def test_feature_columns_are_numeric_and_finite(sections_frame: pd.DataFrame) -> None:
    features, _ = structure.build_structure_features(sections_frame)
    block = features[structure.structure_feature_columns(features)]

    assert (block.dtypes == "float64").all()
    assert np.isfinite(block.to_numpy()).all(), "imputation must not be papering over NaNs here"


def test_the_block_carries_counts_lengths_flags_and_shares(sections_frame: pd.DataFrame) -> None:
    features, vocabulary = structure.build_structure_features(sections_frame)
    columns = set(features.columns)

    assert {"n_body_sections", "n_id_gaps", "total_body_chars", "section_length_gini"} <= columns
    assert f"{structure.TITLE_FLAG_PREFIX}demographics" in columns
    assert all(f"{structure.BUCKET_SHARE_PREFIX}{key}" in columns for key in structure.BUCKET_KEYS)
    assert len(vocabulary) > 10


def test_no_section_text_survives_into_the_block(sections_frame: pd.DataFrame) -> None:
    """The premise of the round: shape only, never content."""
    features, _ = structure.build_structure_features(sections_frame)

    assert "section_text" not in features.columns
    non_numeric = {
        column for column in features.columns if not pd.api.types.is_numeric_dtype(features[column])
    }
    assert non_numeric == {"fips_code"}, f"unexpected text column(s): {non_numeric}"


def test_structure_parquet_cannot_reach_the_pillar_matrix() -> None:
    """A structural block that leaked into the matrix would predict itself."""
    import pillar_matrix

    source = pathlib.Path(pillar_matrix.__file__).read_text()

    assert "source_a_structure_features" not in source


def test_summary_records_the_vocabulary_it_chose(sections_frame: pd.DataFrame) -> None:
    features, vocabulary = structure.build_structure_features(sections_frame)

    stats = structure.summarize(features, vocabulary)

    assert stats["n_counties"] == len(features)
    assert stats["title_flag_vocabulary"] == vocabulary
    assert stats["title_flag_min_share"] == structure.TITLE_FLAG_MIN_SHARE
    assert stats["stub_char_threshold"] == structure.STUB_CHAR_THRESHOLD


def test_four_scored_arms_are_declared() -> None:
    """The unlisted fifth arm is the baseline, which is the comparison rather than a block."""
    import analyze_source_a_structure as scoring

    assert [arm.key for arm in scoring.ARMS] == [
        "structure",
        "typed",
        "typed_plus_structure",
        "size_nonlinear",
    ]


def test_the_null_arm_is_labelled_as_a_null_control() -> None:
    """A reader skimming the arms table must not mistake the calibration arm for a finding."""
    import analyze_source_a_structure as scoring

    null_arm = next(arm for arm in scoring.ARMS if arm.key == "size_nonlinear")

    assert null_arm.against is None
    assert "null control" in null_arm.label.lower()


def test_null_block_is_built_only_from_the_baseline_size_columns() -> None:
    """It must add no information: every column is a function of the three size columns."""
    import analyze_source_a_structure as scoring
    from pillar_matrix import SIZE_FEATURES

    frame = pd.DataFrame(
        {
            "fips_code": ["01001", "01003"],
            "log_population": [10.0, 11.0],
            "log_agi": [13.0, 14.0],
            "log_gdp_latest": [15.0, 16.0],
            "irrelevant": [999.0, -999.0],
        }
    )

    block = scoring.size_nonlinear_block(frame)

    # Three squares, three cubes, three pairwise products.
    assert len(block.columns) == 9
    assert block.loc[0, "log_population_sq"] == pytest.approx(100.0)
    assert block.loc[1, "log_agi_cube"] == pytest.approx(14.0**3)
    assert block.loc[0, "log_population_x_log_gdp_latest"] == pytest.approx(150.0)
    assert all(
        any(size in column for size in SIZE_FEATURES) for column in block.columns
    ), "every null column must be named for the size columns it came from"


def test_null_block_columns_never_collide_with_the_matrix() -> None:
    """The null block is concatenated beside real columns; a collision would score the wrong thing."""
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()

    block = scoring.size_nonlinear_block(matrix)

    assert not set(block.columns) & set(matrix.columns)


def test_typed_block_is_the_shipped_29_columns() -> None:
    """Set equality, not width: two blocks of 29 different columns would pass a length check."""
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    _, blocks = build_matrix()

    assert set(scoring.typed_columns()) == set(blocks["A"])
    assert len(scoring.typed_columns()) == 29


def test_structure_attaches_to_every_matrix_row() -> None:
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    attached, columns = scoring.attach_structure(matrix)

    assert len(attached) == len(matrix)
    assert attached[columns].notna().all().all()
    assert "n_body_sections" in columns


def test_structure_columns_do_not_collide_with_matrix_columns() -> None:
    """A collision would silently rename a column to `_x`/`_y` and score the wrong block."""
    import analyze_source_a_structure as scoring
    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    _, columns = scoring.attach_structure(matrix)

    assert not set(columns) & set(matrix.columns)


def test_attach_structure_rejects_a_duplicated_county(monkeypatch: pytest.MonkeyPatch) -> None:
    """A duplicated fips_code would put copies of one county in both train and test folds."""
    import analyze_source_a_structure as scoring

    matrix = pd.DataFrame({"fips_code": ["01001", "01003"], "log_population": [10.0, 11.0]})
    duplicated = pd.DataFrame(
        {"fips_code": ["01001", "01001", "01003"], "n_body_sections": [10.0, 10.0, 12.0]}
    )
    monkeypatch.setattr(scoring.pd, "read_parquet", lambda _path: duplicated)

    with pytest.raises(ValueError, match="one row per county"):
        scoring.attach_structure(matrix)


def test_attach_structure_keeps_every_matrix_row_on_a_clean_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not fire on the frame it is meant to allow."""
    import analyze_source_a_structure as scoring

    matrix = pd.DataFrame({"fips_code": ["01001", "01003"], "log_population": [10.0, 11.0]})
    clean = pd.DataFrame({"fips_code": ["01001", "01003"], "n_body_sections": [10.0, 12.0]})
    monkeypatch.setattr(scoring.pd, "read_parquet", lambda _path: clean)

    attached, columns = scoring.attach_structure(matrix)

    assert len(attached) == 2
    assert columns == ["n_body_sections"]


def test_arms_share_folds_and_rows() -> None:
    """Paired comparison is the headline statistic; unpaired folds would void it."""
    from sklearn.model_selection import KFold

    import analyze_source_a_structure as scoring

    assert scoring.N_FOLDS == 5
    assert scoring.RANDOM_SEED == 42

    # Shared folds: the splitter is deterministic, so every arm's loop over it
    # sees byte-identical train/test indices.
    design = np.zeros((200, 3))
    splits = [
        [
            (train.tolist(), test.tolist())
            for train, test in KFold(
                n_splits=scoring.N_FOLDS, shuffle=True, random_state=scoring.RANDOM_SEED
            ).split(design)
        ]
        for _ in range(2)
    ]
    assert splits[0] == splits[1]
    assert len(splits[0]) == scoring.N_FOLDS

    # Shared rows: every arm's block is built from the same row mask, so the
    # per-target differences are paired rather than computed on different panels.
    matrix = pd.DataFrame(
        {
            "fips_code": ["01001", "01003", "01005", "01007"],
            "log_population": [10.0, 11.0, 12.0, 13.0],
            "log_agi": [13.0, 14.0, 15.0, 16.0],
            "log_gdp_latest": [15.0, 16.0, 17.0, 18.0],
            "n_body_sections": [10.0, 12.0, 14.0, 16.0],
        }
    )
    for column in scoring.typed_columns():
        matrix[column] = 1.0
    rows = np.array([True, True, False, True])

    blocks = scoring.build_arm_blocks(matrix, ["n_body_sections"], rows)

    assert {arm.key for arm in scoring.ARMS} == set(blocks)
    assert {block.shape[0] for block in blocks.values()} == {int(rows.sum())}
