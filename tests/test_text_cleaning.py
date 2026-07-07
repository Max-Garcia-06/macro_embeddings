"""Tests for text_cleaning: characterization of existing behavior + new patterns."""

from text_cleaning import (
    clean_for_embedding,
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


class TestNewBoilerplatePatterns:
    def test_eponym_clause_removed_including_name(self) -> None:
        text = (
            "The county was named after Abraham Lincoln , the 16th president "
            "of the United States."
        )
        out = strip_boilerplate_phrasing(text)
        assert "Abraham Lincoln" not in out
        assert "named" not in out

    def test_eponym_clause_removed_is_named_for_variant(self) -> None:
        text = "The county is named for Abraham Lincoln , 16th president of the United States ."
        out = strip_boilerplate_phrasing(text)
        assert "Abraham Lincoln" not in out

    def test_metro_area_sentence_removed_entirely(self) -> None:
        text = (
            "comprises the Jamestown micropolitan statistical area . "
            "The population was 21,593."
        )
        out = strip_boilerplate_phrasing(text)
        assert "micropolitan" not in out
        assert "Jamestown" not in out
        assert "21,593" in out

    def test_metropolitan_area_sentence_removed(self) -> None:
        text = "is included in the Montgomery metropolitan area ."
        assert strip_boilerplate_phrasing(text) == ""

    def test_formation_connective_removed_date_kept(self) -> None:
        text = "The county was established on May 9, 1838, and named for Benjamin Franklin ."
        out = strip_boilerplate_phrasing(text)
        assert "established" not in out
        assert "1838" in out
        assert "Benjamin Franklin" not in out


class TestCleanForEmbedding:
    def test_applies_both_strip_stages(self) -> None:
        out = clean_for_embedding(LINCOLN_KS_INTRO, "Lincoln County, Kansas")
        assert "Lincoln County" not in out
        assert "Abraham Lincoln" not in out
        assert "2,939" in out

    def test_falls_back_when_stripping_empties_text(self) -> None:
        # Text that is *entirely* boilerplate: full stripping leaves nothing.
        text = "Foo County is a county located in the U.S. state of Kansas ."
        out = clean_for_embedding(text, "Foo County, Kansas")
        assert out != ""

    def test_falls_back_to_raw_when_everything_empties(self) -> None:
        # County name == entire text: self-reference stripping empties it too.
        out = clean_for_embedding("Foo County", "Foo County, Kansas")
        assert out == "Foo County"
