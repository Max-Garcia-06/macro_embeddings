"""Tests for llm_boilerplate_review: prompt building, response parsing,
the Gemma client, caching, and review orchestration."""

import json

import pytest

from llm_boilerplate_review import build_review_prompt, parse_review_response


class TestBuildReviewPrompt:
    def test_includes_kept_text_and_numbered_sentences(self) -> None:
        prompt = build_review_prompt(
            "It hosts the state's only alligator farm.",
            ["the population was 2,939.", "It was a stop on the old rail line."],
        )
        assert "alligator farm" in prompt
        assert "0. the population was 2,939." in prompt
        assert "1. It was a stop on the old rail line." in prompt


class TestParseReviewResponse:
    def test_parses_well_formed_response(self) -> None:
        raw = json.dumps({"0": True, "1": False})
        assert parse_review_response(raw, 2) == [True, False]

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(ValueError):
            parse_review_response("not json", 1)

    def test_raises_on_missing_sentence_index(self) -> None:
        raw = json.dumps({"0": True})
        with pytest.raises(ValueError):
            parse_review_response(raw, 2)

    def test_raises_on_non_boolean_verdict(self) -> None:
        raw = json.dumps({"0": "yes"})
        with pytest.raises(ValueError):
            parse_review_response(raw, 1)

    def test_raises_on_non_object_json(self) -> None:
        with pytest.raises(ValueError):
            parse_review_response("[true, false]", 2)
