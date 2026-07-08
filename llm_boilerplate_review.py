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
