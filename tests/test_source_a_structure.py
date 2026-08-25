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
