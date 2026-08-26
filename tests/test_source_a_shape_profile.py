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
    sections = make_sections(
        [("01001", 5, "C", "x"), ("01001", 1, "A", "x"), ("01001", 3, "B", "x")]
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
