"""Pure text/HTML cleaning for Source A county intro texts.

Extracted from ingest_source_a.py so cleaning can be tested and iterated on
without importing the ingestion module (whose import triggers the county
crosswalk load). No network access, no model loading.
"""

from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
_CITATION_BRACKET_PATTERN = re.compile(r"\[\d+\]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_LEADING_PUNCTUATION_PATTERN = re.compile(r"^[\s,;:.]+")

# Generic connective phrasing shared near-verbatim by nearly every US county
# lead ("X is a county ... in the U.S. state of Y", "As of the 2020 census,
# the population was N", "The county seat is Z"). These clauses carry no
# distinguishing content themselves -- only the county-specific values inside
# them do -- so they are stripped while leaving those values (population
# figures, seat names, etc.) in place.
_OPENING_TOPIC_SENTENCE_PATTERN = re.compile(
    r"is (?:a|one of the \d+ counties?|a U\.S\. county|a parish)\b"
    r"[^.]*?(?:U\.S\. )?(?:state|Commonwealth) of",
    re.IGNORECASE,
)
_CENSUS_CLAUSE_PATTERN = re.compile(
    r"(?:As of(?: the)? \d{4}(?: United States)?|According to the \d{4}"
    r"|At the \d{4}(?: United States)?) [Cc]ensus\s*,?",
    re.IGNORECASE,
)
_COUNTY_SEAT_CLAUSE_PATTERN = re.compile(
    r"\b(?:Its|The) (?:county|parish) seat(?: and (?:largest|most populous) city)? is\b",
    re.IGNORECASE,
)


def isolate_lead_section(article_html: str) -> str:
    """Isolate the lead/introductory section of an article, dropping later sections.

    Wikimedia Enterprise returns a full HTML document whose <body> is split into
    top-level <section data-mw-section-id="N"> elements per MediaWiki's Parsoid
    section model; section "0" is always the lead, with later sections (each
    starting with an <h2>) covering the rest of the article.

    Args:
        article_html: Full article body HTML.

    Returns:
        HTML string containing only the lead section's contents.
    """
    soup = BeautifulSoup(article_html, "html.parser")
    lead_section = soup.find("section", attrs={"data-mw-section-id": "0"})
    return str(lead_section) if lead_section is not None else article_html


def strip_non_narrative_elements(lead_soup: BeautifulSoup) -> BeautifulSoup:
    """Remove infobox tables, citation markers, and non-content tags in place.

    Args:
        lead_soup: Parsed BeautifulSoup document of the lead section.

    Returns:
        The same BeautifulSoup object with non-narrative elements removed.
    """
    for tag in lead_soup.find_all(["table", "sup", "style", "script"]):
        tag.decompose()
    return lead_soup


def clean_intro_text(article_html: str) -> str:
    """Run the full cleaning pipeline on raw article HTML to produce narrative intro text.

    Args:
        article_html: Full article body HTML as returned by the API.

    Returns:
        Cleaned text, or "" if no narrative content remains.
    """
    lead_html = isolate_lead_section(article_html)
    lead_soup = BeautifulSoup(lead_html, "html.parser")
    lead_soup = strip_non_narrative_elements(lead_soup)

    text = lead_soup.get_text(separator=" ")
    text = _WIKI_LINK_PATTERN.sub(r"\2", text)
    text = _CITATION_BRACKET_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    if not text:
        return ""
    return text


def strip_self_reference(text: str, county_name: str) -> str:
    """Remove mentions of a county's own name and state name from its intro text.

    Wikipedia leads for US counties are heavily templated ("X County is a
    county ... in the U.S. state of Y ...") and are preceded by short-description
    and category breadcrumb text that repeats the state name (e.g. "County in
    Washington, United States County in Washington"). Left in, these shared
    proper nouns dominate embedding similarity between any two counties that
    happen to share a name token (e.g. "Washington County" and any county
    located in Washington state), independent of the counties' actual content.

    Args:
        text: Cleaned intro text, as returned by clean_intro_text.
        county_name: County display name, e.g. "Benton County, Washington".

    Returns:
        Text with the county's short name and state name removed, and any
        leading breadcrumb/hatnote text preceding the first mention of the
        county's own name dropped.
    """
    short_name, _, state_name = county_name.rpartition(", ")

    short_name_pattern = re.compile(re.escape(short_name), re.IGNORECASE)
    match = short_name_pattern.search(text)
    if match:
        text = text[match.start() :]
    text = short_name_pattern.sub("", text)

    if state_name:
        state_name_pattern = re.compile(r"\b" + re.escape(state_name) + r"\b", re.IGNORECASE)
        text = state_name_pattern.sub("", text)

    text = _LEADING_PUNCTUATION_PATTERN.sub("", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def strip_boilerplate_phrasing(text: str) -> str:
    """Remove generic templated connective clauses shared across county leads.

    Targets the near-universal "is a county ... in the U.S. state of",
    "As of the 2020 census, the population was", and "The county seat is"
    clauses, which contribute identical tokens to almost every county's intro
    regardless of content and would otherwise dominate embedding similarity
    for short articles. The county-specific values these clauses introduce
    (population figures, seat names) are left in place.

    Args:
        text: Intro text, typically already passed through strip_self_reference.

    Returns:
        Text with the templated connective clauses removed.
    """
    text = _OPENING_TOPIC_SENTENCE_PATTERN.sub("", text)
    text = _CENSUS_CLAUSE_PATTERN.sub("", text)
    text = _COUNTY_SEAT_CLAUSE_PATTERN.sub("", text)
    text = _LEADING_PUNCTUATION_PATTERN.sub("", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()
