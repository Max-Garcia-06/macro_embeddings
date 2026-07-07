"""Tests for text_cleaning: characterization of existing behavior + new patterns."""

from text_cleaning import (
    clean_intro_text,
    strip_boilerplate_phrasing,
    strip_self_reference,
)

# Real corpus text (Lincoln County, Kansas — source_a_embeddings.parquet).
LINCOLN_KS_INTRO = (
    "County in Kansas, United States County in Kansas Lincoln County is a "
    "county located in the U.S. state of Kansas . Its county seat and largest "
    "city is Lincoln Center . As of the 2020 census , the county population "
    "was 2,939. The county was named after Abraham Lincoln , the 16th "
    "president of the United States."
)


class TestExistingBehavior:
    def test_self_reference_removes_name_state_and_breadcrumb(self) -> None:
        out = strip_self_reference(LINCOLN_KS_INTRO, "Lincoln County, Kansas")
        assert "Lincoln County" not in out
        assert "Kansas" not in out
        assert not out.startswith("County in")  # breadcrumb before first mention dropped
        assert "Lincoln Center" in out  # unrelated proper noun kept

    def test_census_clause_stripped_value_kept(self) -> None:
        out = strip_boilerplate_phrasing(
            "As of the 2020 census , the county population was 2,939."
        )
        assert "As of the 2020 census" not in out
        assert "2,939" in out

    def test_county_seat_clause_stripped(self) -> None:
        out = strip_boilerplate_phrasing("Its county seat and largest city is Lincoln Center .")
        assert "county seat" not in out
        assert "Lincoln Center" in out

    def test_clean_intro_text_returns_empty_string_when_no_content(self) -> None:
        html = '<section data-mw-section-id="0"><table><tr><td>x</td></tr></table></section>'
        assert clean_intro_text(html) == ""
