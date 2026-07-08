"""Tests for the corpus-frequency sentence filter."""

from boilerplate_frequency import (
    drop_common_sentences,
    find_common_templates,
    find_dropped_sentences,
    mask_sentence_template,
    split_sentences,
)


def test_split_sentences() -> None:
    assert split_sentences("First one. Second one. Third") == [
        "First one.",
        "Second one.",
        "Third",
    ]


def test_mask_numbers_and_proper_noun_runs() -> None:
    masked = mask_sentence_template("The population was 2,939 in Lincoln Center .")
    assert masked == "<name> population was <num> in <name> ."


def test_same_shape_different_values_share_template() -> None:
    a = mask_sentence_template("the population was 2,939.")
    b = mask_sentence_template("the population was 50,395.")
    assert a == b


def test_find_common_templates_by_county_fraction() -> None:
    # NOTE: the "unique" sentences must differ in *shape*, not just values —
    # masking collapses numbers/names, so vary the word count per text.
    common = "the population was 100."
    texts = [f"{common} {'alpha ' * (i + 1)}omega." for i in range(10)]
    templates = find_common_templates(texts, min_fraction=0.5)
    assert mask_sentence_template(common) in templates
    assert mask_sentence_template("alpha alpha alpha omega.") not in templates


def test_repeats_within_one_county_count_once() -> None:
    # One county repeating a sentence 10 times must not make it "common".
    texts = ["same thing here. " * 10] + [f"different {i} alpha beta." for i in range(9)]
    templates = find_common_templates(texts, min_fraction=0.5)
    assert mask_sentence_template("same thing here.") not in templates


def test_drop_common_sentences_keeps_rare_ones() -> None:
    templates = {mask_sentence_template("the population was 100.")}
    out = drop_common_sentences(
        "the population was 2,939. It hosts the state's only alligator farm.", templates
    )
    assert "alligator farm" in out
    assert "2,939" not in out


def test_drop_common_sentences_falls_back_when_all_dropped() -> None:
    text = "the population was 2,939."
    templates = {mask_sentence_template(text)}
    assert drop_common_sentences(text, templates) == text


def test_find_dropped_sentences_returns_the_common_ones() -> None:
    templates = {mask_sentence_template("the population was 100.")}
    out = find_dropped_sentences(
        "the population was 2,939. It hosts the state's only alligator farm.",
        templates,
    )
    assert out == ["the population was 2,939."]


def test_find_dropped_sentences_empty_when_nothing_common() -> None:
    templates = {mask_sentence_template("the population was 100.")}
    out = find_dropped_sentences("It hosts the state's only alligator farm.", templates)
    assert out == []


def test_find_dropped_sentences_is_the_complement_of_drop_common_sentences() -> None:
    text = "the population was 2,939. It hosts the state's only alligator farm."
    templates = {mask_sentence_template("the population was 100.")}
    kept = drop_common_sentences(text, templates)
    dropped = find_dropped_sentences(text, templates)
    assert set(split_sentences(kept)) | set(dropped) == set(split_sentences(text))
    assert set(split_sentences(kept)) & set(dropped) == set()
