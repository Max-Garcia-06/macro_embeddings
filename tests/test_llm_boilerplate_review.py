"""Tests for llm_boilerplate_review: prompt building, response parsing,
the Gemma client, caching, and review orchestration."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from llm_boilerplate_review import (
    DEFAULT_GEMMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    GemmaClient,
    append_cache_entry,
    build_review_prompt,
    cache_key,
    load_cache,
    parse_review_response,
    review_dropped_sentences,
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


class FakeGemmaClient:
    """Test double: returns a canned response or raises a canned exception."""

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestCache:
    def test_cache_key_is_stable_and_input_sensitive(self) -> None:
        a = cache_key("kept", "dropped")
        b = cache_key("kept", "dropped")
        c = cache_key("kept", "different")
        assert a == b
        assert a != c

    def test_load_cache_empty_when_missing(self, tmp_path: Path) -> None:
        assert load_cache(tmp_path / "missing.jsonl") == {}

    def test_append_then_load_round_trips(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        append_cache_entry(cache_path, "abc", True)
        append_cache_entry(cache_path, "def", False)
        assert load_cache(cache_path) == {"abc": True, "def": False}


class TestReviewDroppedSentences:
    def test_returns_empty_list_for_no_dropped_sentences(self, tmp_path: Path) -> None:
        client = FakeGemmaClient(response="{}")
        out = review_dropped_sentences("kept text", [], client, tmp_path / "cache.jsonl")
        assert out == []
        assert client.calls == []

    def test_restores_sentence_gemma_marks_true(self, tmp_path: Path) -> None:
        client = FakeGemmaClient(response='{"0": true, "1": false}')
        out = review_dropped_sentences(
            "kept text", ["restore me.", "leave me dropped."], client, tmp_path / "cache.jsonl"
        )
        assert out == ["restore me."]

    def test_falls_back_to_empty_on_malformed_response(self, tmp_path: Path) -> None:
        client = FakeGemmaClient(response="not json")
        out = review_dropped_sentences(
            "kept text", ["some sentence."], client, tmp_path / "cache.jsonl"
        )
        assert out == []

    def test_falls_back_to_empty_on_transport_failure(self, tmp_path: Path) -> None:
        import requests

        client = FakeGemmaClient(response=requests.RequestException("network down"))
        out = review_dropped_sentences(
            "kept text", ["some sentence."], client, tmp_path / "cache.jsonl"
        )
        assert out == []

    def test_cached_verdict_skips_a_second_client_call(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        client = FakeGemmaClient(response='{"0": true}')
        first = review_dropped_sentences("kept text", ["restore me."], client, cache_path)
        second = review_dropped_sentences("kept text", ["restore me."], client, cache_path)
        assert first == second == ["restore me."]
        assert len(client.calls) == 1

    def test_failed_verdict_is_not_persisted_to_cache_file(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        failing_client = FakeGemmaClient(response="not json")
        review_dropped_sentences("kept text", ["some sentence."], failing_client, cache_path)
        assert load_cache(cache_path) == {}

    def test_retries_gemma_after_a_previous_failure(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "cache.jsonl"
        failing_client = FakeGemmaClient(response="not json")
        review_dropped_sentences("kept text", ["some sentence."], failing_client, cache_path)

        working_client = FakeGemmaClient(response='{"0": true}')
        out = review_dropped_sentences("kept text", ["some sentence."], working_client, cache_path)
        assert out == ["some sentence."]
        assert len(working_client.calls) == 1
