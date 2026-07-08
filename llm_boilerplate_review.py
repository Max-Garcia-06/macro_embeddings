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

import hashlib
import json
import logging
from pathlib import Path

import requests

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
            verdicts = None
        if verdicts is not None:
            for idx, verdict in zip(uncached_indices, verdicts):
                cache[keys[idx]] = verdict
                append_cache_entry(cache_path, keys[idx], verdict)
        else:
            for idx in uncached_indices:
                cache[keys[idx]] = False

    return [s for s, k in zip(dropped_sentences, keys) if cache[k]]
