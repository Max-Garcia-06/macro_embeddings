# v5 LLM-Assisted Boilerplate-Drop Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `v5` cleaning variant that asks a local Gemma model to review, per county, every sentence v3's corpus-frequency filter would drop, and restores any sentence Gemma judges to carry real county-specific content — fixing the Stutsman/Providence-style over-stripping failure that v4's length-floor heuristic could not.

**Architecture:** A new pure module (`llm_boilerplate_review.py`) provides prompt building, response parsing, a thin local-Ollama client (`GemmaClient`), an on-disk JSONL verdict cache, and a `review_dropped_sentences` orchestration function. `reembed_source_a.py:build_embedding_texts` gains a `"v5"` branch that computes v3's kept/dropped split as today, calls `review_dropped_sentences` per county, and reconstructs the final text preserving original sentence order. Design reference: `docs/rounds/2026-07-08-llm-boilerplate-review/DESIGN.md`.

**Tech Stack:** Python ≥3.12, uv, `requests` (already a dependency — used for the Ollama HTTP API, no new runtime dependency), pandas/pyarrow, pytest.

## Global Constraints

- Python `>=3.12` (from `pyproject.toml`); all commands run via `uv run`.
- No new runtime dependency: `requests>=2.34.2` is already in `pyproject.toml` and is sufficient to call Ollama's local HTTP API (`POST /api/generate`).
- **Operational prerequisite (not a code dependency):** Ollama must be installed and running locally with the target model pulled, e.g. `ollama pull gemma2:9b`, before Task 6 (the real corpus run) can execute. Tasks 1–5 use a fake client and need neither Ollama nor a GPU/network call.
- `RANDOM_SEED = 42` and `N_PERMUTATIONS = 499` are unchanged (Task 6 reuses `evaluate_source_a_variants.py` as-is).
- **No Wikimedia API calls.** v5 re-cleans and re-embeds from the `raw_intro_text` column already stored in `source_a_embeddings.parquet`, exactly like v2/v3/v4.
- Parquet schema is unchanged from v2–v4 (`county_name`, `fips_code`, `raw_intro_text`, `embedding_text`, `embedding`).
- Conventional Commits; end every commit message with the trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- All functions have type hints and Google-style docstrings; module-level `logger = logging.getLogger(__name__)`; no `print()` in library code; no bare `except:` — catch `(requests.RequestException, ValueError)` specifically where Gemma calls can fail.
- Do not re-open closed directions: v4 remains rejected (do not resurrect the length-floor approach); the real historical baseline for gating is v3's §12 numbers, not a zero-cleaning `raw` reconstruction (findings doc's v4 methodology caveat).

## Decision Gate (pre-registered, used in Task 7)

Reused from `evaluate_source_a_variants.py` against the real historical baseline (`tracked_pair_mean=0.82926`, `pairwise_similarity_std=0.06251`, `mantel_r=-0.12171`):

1. `tracked_pair_mean` drops by **≥0.03** vs. real baseline.
2. `pairwise_similarity_std` is **≥** real baseline.
3. Mantel `r < 0`, `p < 0.05`.
4. Worst `top_far_similar_pairs` entry **≤** v3's worst (0.9607).
5. **New this round:** Stutsman ND ↔ Providence RI tracked-pair similarity **decreases** vs. v3's 0.8327 (not merely "does not increase").

If any criterion fails, v5 is rejected and the baseline remains v3, recorded in `analysis-output/source-a-findings.md` per this experiment line's standing rule.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `boilerplate_frequency.py` | Modify | Add `find_dropped_sentences`, the companion query to `drop_common_sentences` that Task 5 needs to know what to send Gemma. |
| `llm_boilerplate_review.py` | Create | Prompt building, response parsing, `GemmaClient` (local Ollama HTTP wrapper), on-disk verdict cache, `review_dropped_sentences` orchestration. No import-time network or model side effects. |
| `reembed_source_a.py` | Modify | Add `"v5"` to `build_embedding_texts` and the CLI's `--variant` choices; order-preserving reconstruction helper. |
| `tests/test_boilerplate_frequency.py` | Modify | Tests for `find_dropped_sentences`. |
| `tests/test_llm_boilerplate_review.py` | Create | Tests for prompt building, response parsing, `GemmaClient` (mocked HTTP), cache, and `review_dropped_sentences`. |
| `tests/test_reembed_source_a.py` | Modify | v5 integration tests: zero-drop regression, restoration with order preservation, fallback on Gemma failure. |
| `analysis-output/source-a-findings.md` | Modify (Task 7) | Round writeup recording the v5 gate outcome. |

**Interfaces between new/modified functions (single source of truth):**

```python
# boilerplate_frequency.py (Task 1 adds this one)
def find_dropped_sentences(text: str, common_templates: set[str]) -> list[str]

# llm_boilerplate_review.py
DEFAULT_GEMMA_MODEL: str = "gemma2:9b"
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
REQUEST_TIMEOUT_SECONDS: int = 120
DEFAULT_CACHE_PATH: Path

def build_review_prompt(kept_text: str, dropped_sentences: list[str]) -> str
def parse_review_response(raw: str, n_sentences: int) -> list[bool]        # raises ValueError on malformed/mismatched output

class GemmaClient:
    def __init__(self, model: str = DEFAULT_GEMMA_MODEL, host: str = DEFAULT_OLLAMA_HOST) -> None
    def generate(self, prompt: str) -> str                                  # raises requests.RequestException on transport failure

def cache_key(kept_text: str, sentence: str) -> str
def load_cache(cache_path: Path) -> dict[str, bool]
def append_cache_entry(cache_path: Path, key: str, verdict: bool) -> None
def review_dropped_sentences(
    kept_text: str,
    dropped_sentences: list[str],
    client: GemmaClient,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[str]                                                               # never raises; falls back to [] (nothing restored) on failure

# reembed_source_a.py
def build_embedding_texts(
    df: pd.DataFrame,
    variant: str,
    gemma_client: GemmaClient | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[str]                                                               # variant now also accepts "v5"
```

---

### Task 1: `find_dropped_sentences` in `boilerplate_frequency.py`

**Files:**
- Modify: `boilerplate_frequency.py`
- Test: `tests/test_boilerplate_frequency.py`

**Interfaces:**
- Consumes: `split_sentences`, `mask_sentence_template` (existing, unchanged).
- Produces: `find_dropped_sentences(text: str, common_templates: set[str]) -> list[str]` — the inverse query of `drop_common_sentences`: sentences (original wording, original order) whose masked template *is* in `common_templates`. Task 5 needs this to know which sentences to send Gemma for review.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_boilerplate_frequency.py`)

```python
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
```

Add `find_dropped_sentences` to the file's `from boilerplate_frequency import (...)` block.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_boilerplate_frequency.py -v -k find_dropped_sentences
```

Expected: FAIL with `ImportError: cannot import name 'find_dropped_sentences'`.

- [ ] **Step 3: Implement**

Append to `boilerplate_frequency.py`, directly after `drop_common_sentences`:

```python
def find_dropped_sentences(text: str, common_templates: set[str]) -> list[str]:
    """Find the sentences drop_common_sentences would remove from text.

    The complement of drop_common_sentences: instead of the kept text, this
    returns the sentences (original wording, original order) whose masked
    template is common. Used to build the candidate list an LLM reviewer
    checks for wrongly-dropped county-specific content.

    Args:
        text: Cleaned intro text.
        common_templates: Output of find_common_templates.

    Returns:
        Sentences whose masked template is in common_templates, in their
        original order and wording.
    """
    return [s for s in split_sentences(text) if mask_sentence_template(s) in common_templates]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_boilerplate_frequency.py -v
```

Expected: all tests PASS (existing 7 + 3 new = 10).

- [ ] **Step 5: Commit**

```bash
git add boilerplate_frequency.py tests/test_boilerplate_frequency.py
git commit -m "feat(source-a): add find_dropped_sentences to boilerplate_frequency"
```

---

### Task 2: Prompt building and response parsing (`llm_boilerplate_review.py`)

**Files:**
- Create: `llm_boilerplate_review.py`
- Test: `tests/test_llm_boilerplate_review.py`

**Interfaces:**
- Consumes: nothing (pure module, no network, no model).
- Produces: `build_review_prompt(kept_text: str, dropped_sentences: list[str]) -> str`, `parse_review_response(raw: str, n_sentences: int) -> list[bool]`.

This task covers only the two pure functions — no HTTP client yet (Task 3) and no caching yet (Task 4). `parse_review_response` must reject anything that isn't a JSON object with a boolean verdict for every sentence index `0..n_sentences-1`, since Task 4's fallback logic depends on `ValueError` being the single failure signal to catch.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_boilerplate_review.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'llm_boilerplate_review'`.

- [ ] **Step 3: Implement**

Create `llm_boilerplate_review.py`:

```python
"""LLM-assisted review of boilerplate_frequency's dropped sentences.

For counties where the corpus-frequency filter (boilerplate_frequency.py)
drops a sentence because its masked template recurs across many counties,
this module asks a local Gemma model whether that specific sentence still
carries county-specific information the kept text doesn't already have. If
so, the sentence is restored. This targets the over-stripping failure mode
documented in analysis-output/source-a-findings.md section 12 (the
Stutsman/Providence tracked pair), which a pure length-floor heuristic
(the rejected v4 variant) could not fix.

No import-time network or model access -- prompt building and response
parsing are pure functions; GemmaClient (Task 3) and caching (Task 4) are
only invoked when review_dropped_sentences is actually called.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def build_review_prompt(kept_text: str, dropped_sentences: list[str]) -> str:
    """Build the prompt asking Gemma to review one county's dropped sentences.

    Args:
        kept_text: The county's v3 (frequency-filtered) kept text, given as
            context so the model can judge whether a dropped sentence adds
            anything not already present.
        dropped_sentences: Sentences the frequency filter would drop for
            this county, in original wording and order.

    Returns:
        The full prompt string.
    """
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(dropped_sentences))
    return (
        "You are reviewing a Wikipedia county article's introductory text "
        "that has already had generic template sentences removed.\n\n"
        "KEPT TEXT (already retained for this county):\n"
        f"{kept_text}\n\n"
        "CANDIDATE SENTENCES (flagged as boilerplate because their shape "
        "recurs across many counties, but shape alone can be wrong for a "
        "specific county):\n"
        f"{numbered}\n\n"
        "For each numbered sentence, decide: does it add county-specific "
        "information not already present in the kept text? Respond with "
        "ONLY a JSON object mapping each sentence's number (as a string) to "
        "true (restore it) or false (leave it dropped). Example for 2 "
        'sentences: {"0": true, "1": false}'
    )


def parse_review_response(raw: str, n_sentences: int) -> list[bool]:
    """Parse and validate Gemma's JSON verdict response.

    Args:
        raw: The model's raw text response.
        n_sentences: Expected number of verdicts (sentence indices 0..n-1).

    Returns:
        One bool per sentence index, in order (True = restore).

    Raises:
        ValueError: If the response isn't valid JSON, isn't an object, is
            missing a verdict for any index, or has a non-boolean verdict.
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemma response was not valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Gemma response was not a JSON object: {raw!r}")

    verdicts: list[bool] = []
    for i in range(n_sentences):
        key = str(i)
        if key not in parsed:
            raise ValueError(f"Gemma response missing verdict for sentence {i}: {raw!r}")
        value = parsed[key]
        if not isinstance(value, bool):
            raise ValueError(
                f"Gemma response verdict for sentence {i} was not boolean: {value!r}"
            )
        verdicts.append(value)
    return verdicts
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_boilerplate_review.py tests/test_llm_boilerplate_review.py
git commit -m "feat(source-a): add Gemma review prompt building and response parsing"
```

---

### Task 3: `GemmaClient` (local Ollama HTTP wrapper)

**Files:**
- Modify: `llm_boilerplate_review.py`
- Test: `tests/test_llm_boilerplate_review.py`

**Interfaces:**
- Consumes: nothing from Task 2 directly (independent class), but is what Task 4's `review_dropped_sentences` will call.
- Produces: `GemmaClient` class with `generate(self, prompt: str) -> str`; constants `DEFAULT_GEMMA_MODEL`, `DEFAULT_OLLAMA_HOST`, `REQUEST_TIMEOUT_SECONDS`.

Pinned for determinism: fixed model tag (never `:latest`), `temperature=0`. Uses `requests` (already a project dependency) against Ollama's local `/api/generate` endpoint — no new runtime dependency.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_llm_boilerplate_review.py`)

```python
from unittest.mock import Mock, patch

from llm_boilerplate_review import DEFAULT_GEMMA_MODEL, DEFAULT_OLLAMA_HOST, GemmaClient


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v -k GemmaClient
```

Expected: FAIL with `ImportError: cannot import name 'GemmaClient'`.

- [ ] **Step 3: Implement**

Append to `llm_boilerplate_review.py` (add `import requests` to the top-level imports):

```python
DEFAULT_GEMMA_MODEL: str = "gemma2:9b"
DEFAULT_OLLAMA_HOST: str = "http://localhost:11434"
REQUEST_TIMEOUT_SECONDS: int = 120


class GemmaClient:
    """Thin wrapper around a local Ollama /api/generate call, pinned for determinism."""

    def __init__(self, model: str = DEFAULT_GEMMA_MODEL, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        """Send a prompt to the local Gemma model and return its raw text response.

        Args:
            prompt: Full prompt text.

        Returns:
            The model's raw response string.

        Raises:
            requests.RequestException: On any transport or HTTP-status failure.
        """
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["response"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v
```

Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_boilerplate_review.py tests/test_llm_boilerplate_review.py
git commit -m "feat(source-a): add GemmaClient local Ollama wrapper"
```

---

### Task 4: Verdict cache and `review_dropped_sentences` orchestration

**Files:**
- Modify: `llm_boilerplate_review.py`
- Test: `tests/test_llm_boilerplate_review.py`

**Interfaces:**
- Consumes: `build_review_prompt`, `parse_review_response` (Task 2), `GemmaClient` (Task 3).
- Produces: `DEFAULT_CACHE_PATH: Path`, `cache_key(kept_text: str, sentence: str) -> str`, `load_cache(cache_path: Path) -> dict[str, bool]`, `append_cache_entry(cache_path: Path, key: str, verdict: bool) -> None`, `review_dropped_sentences(kept_text: str, dropped_sentences: list[str], client: GemmaClient, cache_path: Path = DEFAULT_CACHE_PATH) -> list[str]`. This is what Task 5 calls per county.

Caching is keyed on `(kept_text, sentence)` so a change upstream (e.g. a future v2 regex fix) correctly invalidates old verdicts. On any Gemma failure (`requests.RequestException` or `ValueError` from a malformed response), `review_dropped_sentences` logs a warning and returns `[]` for the uncached sentences in that call — the county falls back to plain v3 behavior (nothing restored), never raises, and never blocks the run. This mirrors this project's established per-item failure logging (see `ingest_source_a.py`'s per-county failure handling, noted in `README.md`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_llm_boilerplate_review.py`)

```python
from pathlib import Path

from llm_boilerplate_review import (
    append_cache_entry,
    cache_key,
    load_cache,
    review_dropped_sentences,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v -k "Cache or ReviewDroppedSentences"
```

Expected: FAIL with `ImportError` (names not yet defined).

- [ ] **Step 3: Implement**

Append to `llm_boilerplate_review.py` (add `import hashlib` and `from pathlib import Path` to the top-level imports):

```python
DEFAULT_CACHE_PATH: Path = (
    Path(__file__).resolve().parent / "analysis-output" / "llm_review_cache.jsonl"
)


def cache_key(kept_text: str, sentence: str) -> str:
    """Build a stable cache key for one (kept_text, dropped_sentence) pair.

    Args:
        kept_text: The county's kept (v3) text, used as review context.
        sentence: The specific dropped sentence being judged.

    Returns:
        A hex-digest cache key.
    """
    digest = hashlib.sha256()
    digest.update(kept_text.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(sentence.encode("utf-8"))
    return digest.hexdigest()


def load_cache(cache_path: Path) -> dict[str, bool]:
    """Load all cached verdicts from a JSONL cache file.

    Args:
        cache_path: Path to the cache file; need not exist yet.

    Returns:
        Mapping of cache_key -> verdict. Empty if the file doesn't exist.
    """
    if not cache_path.exists():
        return {}
    cache: dict[str, bool] = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        cache[entry["key"]] = entry["verdict"]
    return cache


def append_cache_entry(cache_path: Path, key: str, verdict: bool) -> None:
    """Append one verdict to the JSONL cache file, creating it if needed.

    Args:
        cache_path: Path to the cache file.
        key: Cache key, as produced by cache_key.
        verdict: True (restore) or False (leave dropped).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({"key": key, "verdict": verdict}) + "\n")


def review_dropped_sentences(
    kept_text: str,
    dropped_sentences: list[str],
    client: GemmaClient,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[str]:
    """Ask Gemma which of a county's dropped sentences to restore.

    Looks up each (kept_text, sentence) pair in the on-disk cache first;
    only calls the client for cache misses, in a single batched call
    covering all misses for this county. On any client or parse failure,
    logs a warning and treats every uncached sentence as "leave dropped"
    (the county falls back to plain v3 behavior) rather than raising.

    Args:
        kept_text: The county's v3 (frequency-filtered) kept text.
        dropped_sentences: Sentences the frequency filter would drop for
            this county, in original wording and order.
        client: GemmaClient (or compatible test double) to query on a
            cache miss.
        cache_path: JSONL verdict cache path.

    Returns:
        The subset of dropped_sentences to restore, in original order.
    """
    if not dropped_sentences:
        return []

    cache = load_cache(cache_path)
    keys = [cache_key(kept_text, s) for s in dropped_sentences]
    uncached_indices = [i for i, k in enumerate(keys) if k not in cache]

    if uncached_indices:
        uncached_sentences = [dropped_sentences[i] for i in uncached_indices]
        try:
            prompt = build_review_prompt(kept_text, uncached_sentences)
            raw = client.generate(prompt)
            verdicts = parse_review_response(raw, len(uncached_sentences))
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Gemma review failed (%s); leaving drops as-is", exc)
            verdicts = [False] * len(uncached_sentences)
        for idx, verdict in zip(uncached_indices, verdicts):
            cache[keys[idx]] = verdict
            append_cache_entry(cache_path, keys[idx], verdict)

    return [s for s, k in zip(dropped_sentences, keys) if cache[k]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_boilerplate_review.py -v
```

Expected: 16 PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_boilerplate_review.py tests/test_llm_boilerplate_review.py
git commit -m "feat(source-a): add verdict cache and review_dropped_sentences orchestration"
```

---

### Task 5: Integrate `v5` into `reembed_source_a.py`

**Files:**
- Modify: `reembed_source_a.py`
- Test: `tests/test_reembed_source_a.py`

**Interfaces:**
- Consumes: `find_dropped_sentences` (Task 1), `GemmaClient`, `review_dropped_sentences`, `DEFAULT_CACHE_PATH` (Tasks 3–4).
- Produces: `build_embedding_texts(df, variant, gemma_client=None, cache_path=DEFAULT_CACHE_PATH) -> list[str]` now accepting `variant="v5"`; private helper `_restore_ordered(text: str, common_templates: set[str], restored: set[str]) -> str`.

`_restore_ordered` reconstructs a county's final text by walking the *original* v2 sentence order and keeping a sentence if either its template isn't common, or it's in the Gemma-restored set — this is what preserves natural sentence order instead of appending restorations at the end. If nothing survives (all-boilerplate stub, restoration didn't save it), it falls back to the original text, matching `drop_common_sentences`'s own fallback.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_reembed_source_a.py`)

```python
class FakeGemmaClient:
    def __init__(self, response: str | Exception) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_v5_matches_v3_when_no_sentences_dropped(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [
                "The county maintains one of the state's oldest grain elevators.",
                "Farming is common in this part of the state.",
            ],
        }
    )
    v3_texts = build_embedding_texts(df, "v3")
    v5_texts = build_embedding_texts(
        df, "v5", gemma_client=FakeGemmaClient(response="{}"), cache_path=tmp_path / "cache.jsonl"
    )
    assert v5_texts == v3_texts


def test_v5_restores_sentence_gemma_marks_non_boilerplate(tmp_path) -> None:
    common_sentence = (
        "Many local histories mention faraway explorers visiting "
        "centuries ago for trade."
    )
    unique_a = "The county maintains one of the state's oldest grain elevators."
    unique_b = "Farming is common here."
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [f"{common_sentence} {unique_a}", f"{common_sentence} {unique_b}"],
        }
    )
    # Confirm the frequency filter does drop the common sentence under v3.
    v3_texts = build_embedding_texts(df, "v3")
    assert common_sentence not in v3_texts[0]

    client = FakeGemmaClient(response='{"0": true}')
    v5_texts = build_embedding_texts(
        df, "v5", gemma_client=client, cache_path=tmp_path / "cache.jsonl"
    )

    # Restored, and in original order (common sentence first, as in raw text).
    assert v5_texts[0] == f"{common_sentence} {unique_a}"
    assert v5_texts[1] == f"{common_sentence} {unique_b}"


def test_v5_falls_back_to_v3_when_gemma_call_fails(tmp_path) -> None:
    common_sentence = (
        "Many local histories mention faraway explorers visiting "
        "centuries ago for trade."
    )
    unique_a = "The county maintains one of the state's oldest grain elevators."
    unique_b = "Farming is common here."
    df = pd.DataFrame(
        {
            "county_name": ["Millbrook County, Astoria", "Thistle County, Astoria"],
            "raw_intro_text": [f"{common_sentence} {unique_a}", f"{common_sentence} {unique_b}"],
        }
    )
    v3_texts = build_embedding_texts(df, "v3")
    v5_texts = build_embedding_texts(
        df,
        "v5",
        gemma_client=FakeGemmaClient(response="not json"),
        cache_path=tmp_path / "cache.jsonl",
    )
    assert v5_texts == v3_texts
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_reembed_source_a.py -v -k v5
```

Expected: FAIL — `build_embedding_texts` raises `ValueError: Unknown variant: 'v5'`.

- [ ] **Step 3: Implement**

In `reembed_source_a.py`, update the imports:

```python
from pathlib import Path

from boilerplate_frequency import (
    DEFAULT_MIN_COUNTY_FRACTION,
    drop_common_sentences,
    find_common_templates,
    find_dropped_sentences,
    mask_sentence_template,
    split_sentences,
)
from llm_boilerplate_review import DEFAULT_CACHE_PATH, GemmaClient, review_dropped_sentences
```

Add the reconstruction helper directly above `build_embedding_texts`:

```python
def _restore_ordered(text: str, common_templates: set[str], restored: set[str]) -> str:
    """Rebuild text keeping non-common sentences plus any Gemma-restored ones.

    Preserves the original sentence order (restorations are not appended at
    the end). Falls back to the unfiltered text if nothing would remain,
    matching drop_common_sentences's own fallback.

    Args:
        text: Original (v2) county text, before frequency filtering.
        common_templates: Output of find_common_templates.
        restored: Sentences Gemma judged should be restored.

    Returns:
        Reconstructed text.
    """
    sentences = split_sentences(text)
    kept = [
        s for s in sentences if mask_sentence_template(s) not in common_templates or s in restored
    ]
    if not kept:
        return text
    return " ".join(kept)
```

Update `build_embedding_texts`'s signature, docstring, and body:

```python
def build_embedding_texts(
    df: pd.DataFrame,
    variant: str,
    gemma_client: GemmaClient | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[str]:
    """Produce the per-county text to embed for a cleaning variant.

    Args:
        df: DataFrame with county_name and raw_intro_text columns.
        variant: "raw" (no cleaning), "v2" (regex cleaning), "v3" (v2 +
            frequency filter), "v4" (v2 + outcome-gated frequency filter),
            or "v5" (v2 + frequency filter, with a local Gemma model
            reviewing and potentially restoring each dropped sentence).
        gemma_client: Client used for "v5" reviews; a default GemmaClient()
            is constructed if not given. Ignored for other variants.
        cache_path: Verdict cache path used for "v5". Ignored for other
            variants.

    Returns:
        One non-empty text per row, in row order.

    Raises:
        ValueError: If variant is not one of the supported names.
    """
    if variant not in ("raw", "v2", "v3", "v4", "v5"):
        raise ValueError(f"Unknown variant: {variant!r}")
    if variant == "raw":
        return df["raw_intro_text"].tolist()
    texts = [
        clean_for_embedding(raw, name)
        for name, raw in zip(df["county_name"], df["raw_intro_text"])
    ]
    if variant in ("v3", "v4", "v5"):
        templates = find_common_templates(texts, DEFAULT_MIN_COUNTY_FRACTION)
        logger.info("Frequency filter: %d common templates found", len(templates))
        filtered = [drop_common_sentences(t, templates) for t in texts]
        if variant == "v3":
            texts = filtered
        elif variant == "v4":
            texts = [
                candidate if len(candidate) >= MIN_CONTENT_LENGTH else original
                for original, candidate in zip(texts, filtered)
            ]
        else:  # v5
            client = gemma_client if gemma_client is not None else GemmaClient()
            texts = [
                _restore_ordered(
                    original,
                    templates,
                    set(
                        review_dropped_sentences(
                            candidate,
                            find_dropped_sentences(original, templates),
                            client,
                            cache_path,
                        )
                    ),
                )
                for original, candidate in zip(texts, filtered)
            ]
    return texts
```

Update the module docstring's `Variants:` list to add:

```
  v5: v2, then frequency-filter drops are reviewed by a local Gemma model
      (llm_boilerplate_review) and restored where Gemma judges they carry
      county-specific content the kept text lacks (see
      analysis-output/source-a-findings.md section 12's rural-county
      over-stripping finding, and docs/rounds/2026-07-08-llm-boilerplate-
      review/DESIGN.md).
```

Update the CLI argument:

```python
    parser.add_argument("--variant", choices=["raw", "v2", "v3", "v4", "v5"], required=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_reembed_source_a.py -v
```

Expected: all tests PASS (existing 3 + 3 new = 6).

- [ ] **Step 5: Run the full test suite (no regressions)**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add reembed_source_a.py tests/test_reembed_source_a.py
git commit -m "feat(source-a): add v5 variant with Gemma-reviewed boilerplate restoration"
```

---

### Task 6: Full-corpus v5 run and gate evaluation (operational, not TDD)

This task runs the real pipeline end-to-end. It has no unit-test cycle of its own — Tasks 1–5 already cover the code with tests using fake clients. This task requires Ollama running locally with the target model available.

**Prerequisite:**

```bash
ollama pull gemma2:9b
ollama list   # confirm gemma2:9b is present
```

- [ ] **Step 1: Re-embed the full corpus for v5**

```bash
uv run reembed_source_a.py --variant v5
```

Expected: `source_a_embeddings_v5.parquet` written with 3,144 rows. This will take considerably longer than v3/v4 (one Gemma call per county with ≥1 dropped sentence, likely most counties) — run with enough time budget and monitor `analysis-output/llm_review_cache.jsonl` growing as a progress signal.

- [ ] **Step 2: Evaluate against the real historical baseline**

```bash
uv run evaluate_source_a_variants.py source_a_embeddings.parquet source_a_embeddings_v5.parquet
```

Expected: printed comparison table plus `analysis-output/variant-eval.json` updated with a `source_a_embeddings_v5.parquet` entry.

- [ ] **Step 3: Check the five gate criteria from this plan's Decision Gate section**

Compare the printed `tracked_pair_mean`, `pairwise_similarity_std`, `mantel_r`/`mantel_p`, worst `top_far_similar_pairs` entry, and the Stutsman/Providence tracked-pair value specifically, against the thresholds above. Note the pass/fail for each.

- [ ] **Step 4: If v5 passes, adopt it as the new baseline**

```bash
cp source_a_embeddings_v5.parquet source_a_embeddings.parquet
```

If v5 fails any criterion, do not overwrite `source_a_embeddings.parquet` — v3 remains the baseline.

- [ ] **Step 5: Commit whichever result occurred**

```bash
git add source_a_embeddings.parquet analysis-output/variant-eval.json analysis-output/llm_review_cache.jsonl
git commit -m "feat(source-a): adopt v5 Gemma-reviewed embeddings after gated evaluation"
# OR, if rejected:
git add analysis-output/variant-eval.json analysis-output/llm_review_cache.jsonl
git commit -m "chore(source-a): record v5 Gemma-review evaluation (rejected)"
```

---

### Task 7: Record the outcome in `analysis-output/source-a-findings.md`

**Files:**
- Modify: `analysis-output/source-a-findings.md`

- [ ] **Step 1: Write a new dated section**

Following the existing §12/v4-rejection section's format exactly (Method, Results table, Gate check table, Decision, Next Actions, Claim candidates with Allowed/Forbidden wording), add a new section documenting v5's method (per this plan's Task 5), the measured metrics from Task 6 Step 2, the gate check against all five criteria from this plan's Decision Gate section, and the adopt/reject decision. If rejected, be explicit about which criterion failed and why, mirroring how the v4 section attributed its failure to the Stutsman/Providence pair specifically rather than describing it vaguely.

- [ ] **Step 2: Commit**

```bash
git add analysis-output/source-a-findings.md
git commit -m "docs(source-a): record v5 Gemma-review round outcome"
```
