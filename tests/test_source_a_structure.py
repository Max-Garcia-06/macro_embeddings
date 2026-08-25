"""Structural features derived from Wikipedia section titles and lengths."""
from __future__ import annotations

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
