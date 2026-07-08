"""Tests for llm_boilerplate_review: prompt building, response parsing,
the Gemma client, caching, and review orchestration."""

import json
from unittest.mock import Mock, patch

import pytest

from llm_boilerplate_review import (
    DEFAULT_GEMMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    GemmaClient,
    build_review_prompt,
    parse_review_response,
)


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


class TestGemmaClient:
    @patch("llm_boilerplate_review.requests.post")
    def test_generate_posts_pinned_temperature_and_model(self, mock_post: Mock) -> None:
        mock_post.return_value = Mock(json=lambda: {"response": '{"0": true}'})
        mock_post.return_value.raise_for_status = lambda: None

        client = GemmaClient()
        result = client.generate("some prompt")

        assert result == '{"0": true}'
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == DEFAULT_GEMMA_MODEL
        assert kwargs["json"]["prompt"] == "some prompt"
        assert kwargs["json"]["options"]["temperature"] == 0
        assert mock_post.call_args[0][0] == f"{DEFAULT_OLLAMA_HOST}/api/generate"

    @patch("llm_boilerplate_review.requests.post")
    def test_generate_raises_on_http_error(self, mock_post: Mock) -> None:
        import requests

        mock_post.return_value = Mock()
        mock_post.return_value.raise_for_status.side_effect = requests.HTTPError("500")

        client = GemmaClient()
        with pytest.raises(requests.RequestException):
            client.generate("some prompt")
