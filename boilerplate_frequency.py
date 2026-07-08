"""Corpus-frequency sentence filter for Source A intro texts.

Masks numbers and proper-noun runs in each sentence to a shape "template";
sentences whose template recurs across a configurable fraction of counties
are treated as boilerplate and dropped. Complements the regex patterns in
text_cleaning.py by catching templated sentence shapes not explicitly
enumerated (analysis-output/source-a-findings.md section 3.4, mechanism b).
"""

from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

_NUMBER_PATTERN = re.compile(r"\d[\d,./-]*")
# A run of one or more capitalized words collapses to a single <NAME> token,
# so "Lincoln Center" and "Newport" mask identically. Sentence-initial words
# get masked too; acceptable, since masking is applied uniformly to every
# sentence being compared.
_PROPER_NOUN_RUN_PATTERN = re.compile(r"\b[A-Z][a-zA-Z'’]*(?:\s+[A-Z][a-zA-Z'’]*)*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

# A sentence template present in >=5% of counties (~157 of 3,144) cannot be
# carrying county-specific narrative; it is templated boilerplate.
DEFAULT_MIN_COUNTY_FRACTION: float = 0.05


def split_sentences(text: str) -> list[str]:
    """Split whitespace-normalized intro text into sentences.

    Args:
        text: Single-line, whitespace-normalized text.

    Returns:
        Non-empty sentence strings, terminal punctuation retained.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]


def mask_sentence_template(sentence: str) -> str:
    """Reduce a sentence to its shape template.

    Numbers become <NUM>, proper-noun runs become <NAME>, whitespace is
    normalized, and the result is lowercased, so sentences differing only in
    county-specific values map to the same template.

    Args:
        sentence: One sentence.

    Returns:
        Lowercased template string.
    """
    masked = _PROPER_NOUN_RUN_PATTERN.sub("<NAME>", sentence)
    masked = _NUMBER_PATTERN.sub("<NUM>", masked)
    return _WHITESPACE_PATTERN.sub(" ", masked).strip().lower()


def find_common_templates(
    texts: list[str],
    min_fraction: float = DEFAULT_MIN_COUNTY_FRACTION,
) -> set[str]:
    """Find sentence templates shared by at least min_fraction of texts.

    Each text contributes each template at most once, so repetition within a
    single county cannot promote a template to "common".

    Args:
        texts: One cleaned intro text per county.
        min_fraction: Minimum fraction of counties a template must appear in.

    Returns:
        Set of common (boilerplate) templates.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update({mask_sentence_template(s) for s in split_sentences(text)})
    threshold = max(2, int(min_fraction * len(texts)))
    return {template for template, count in counts.items() if count >= threshold}


def drop_common_sentences(text: str, common_templates: set[str]) -> str:
    """Remove sentences whose template is in the common (boilerplate) set.

    Falls back to the input text unchanged if every sentence would be
    dropped, so no county ends up with an empty embedding input.

    Args:
        text: Cleaned intro text.
        common_templates: Output of find_common_templates.

    Returns:
        Text with boilerplate-template sentences removed.
    """
    kept = [s for s in split_sentences(text) if mask_sentence_template(s) not in common_templates]
    if not kept:
        return text
    return " ".join(kept)


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
