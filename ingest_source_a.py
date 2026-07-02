"""Source A ingestion pipeline for E_macro: Wikipedia Introductory Corpora.

Queries the English Wikimedia Enterprise API for county Wikipedia articles,
isolates the lead/introductory section (excluding infobox, body sections, and
metadata), embeds the cleaned text with BAAI/bge-m3, L2-normalizes the
resulting vector, and stores the result set as a local Parquet file.

Requires WIKIMEDIA_USERNAME and WIKIMEDIA_PASSWORD environment variables set
to valid Wikimedia Enterprise credentials.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

WIKIMEDIA_AUTH_URL: str = "https://auth.enterprise.wikimedia.com/v1/login"
WIKIMEDIA_ARTICLES_URL_TEMPLATE: str = "https://api.enterprise.wikimedia.com/v2/articles/{name}"
WIKIMEDIA_PROJECT_FILTER: str = "enwiki"
REQUEST_TIMEOUT_SECONDS: int = 30

EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
EMBEDDING_DIM: int = 1024

OUTPUT_PARQUET_PATH: Path = Path(__file__).resolve().parent / "source_a_embeddings.parquet"

TEST_COUNTIES: list[str] = [
    "Allegheny County, Pennsylvania",
    "Loudoun County, Virginia",
    "Orange County, California",
    "Capitol Planning Region, Connecticut",
    "Western Connecticut Planning Region, Connecticut",
    "Fairfield County, Connecticut",
    "Orleans Parish, Louisiana",
    "North Slope Borough, Alaska",
    "Yukon-Koyukuk Census Area, Alaska",
    "Baltimore City, Maryland",
    "Baltimore County, Maryland",
    "St. Louis City, Missouri",
    "St. Louis County, Missouri",
    "Carson City, Nevada",
    "Richmond City, Virginia",
    "San Francisco County, California",
    "Philadelphia County, Pennsylvania",
    "Denver County, Colorado",
    "Broomfield County, Colorado",
    "Doña Ana County, New Mexico",
    "Coös County, New Hampshire",
    "O'Brien County, Iowa",
    "Prince George's County, Maryland",
    "DeKalb County, Indiana",
    "Miami-Dade County, Florida",
    "Oglala Lakota County, South Dakota",
    "Kusilvak Census Area, Alaska",
    "Chugach Census Area, Alaska",
    "Copper River Census Area, Alaska",
    "District of Columbia",
    "Kalawao County, Hawaii",
    "Loving County, Texas",
    "Los Angeles County, California",
]

# Independent cities and consolidated city-counties without a
# "[County] County, [State]" article structure, plus other names whose
# Wikipedia article title diverges from the county_name identifier.
# Not a general solution -- extend manually as new edge cases surface.
INDEPENDENT_CITY_ARTICLE_LOOKUP: dict[str, str] = {
    "St. Louis, Missouri": "St. Louis",
    "St. Louis City, Missouri": "St. Louis",
    "Baltimore, Maryland": "Baltimore",
    "Baltimore City, Maryland": "Baltimore",
    "Carson City, Nevada": "Carson City, Nevada",
    "Richmond City, Virginia": "Richmond, Virginia",
    "San Francisco County, California": "San Francisco",
    "Philadelphia County, Pennsylvania": "Philadelphia",
    "Denver County, Colorado": "Denver",
    "Broomfield County, Colorado": "Broomfield, Colorado",
    "District of Columbia": "Washington, D.C.",
    "Yukon-Koyukuk Census Area, Alaska": "Yukon–Koyukuk Census Area, Alaska",
}

# FIPS crosswalk for the counties in TEST_COUNTIES. Not a general solution --
# a full Census Bureau crosswalk is pending a later ingestion stage.
FIPS_CROSSWALK: dict[str, str] = {
    "Capitol Planning Region, Connecticut": "09110",
    "Western Connecticut Planning Region, Connecticut": "09190",
    "Fairfield County, Connecticut": "09001",
    "Orleans Parish, Louisiana": "22071",
    "North Slope Borough, Alaska": "02185",
    "Yukon-Koyukuk Census Area, Alaska": "02290",
    "Baltimore City, Maryland": "24510",
    "Baltimore County, Maryland": "24005",
    "St. Louis City, Missouri": "29510",
    "St. Louis County, Missouri": "29189",
    "Carson City, Nevada": "32510",
    "Richmond City, Virginia": "51760",
    "San Francisco County, California": "06075",
    "Philadelphia County, Pennsylvania": "42101",
    "Denver County, Colorado": "08031",
    "Broomfield County, Colorado": "08014",
    "Doña Ana County, New Mexico": "35013",
    "Coös County, New Hampshire": "33007",
    "O'Brien County, Iowa": "19141",
    "Prince George's County, Maryland": "24033",
    "DeKalb County, Indiana": "18033",
    "Miami-Dade County, Florida": "12086",
    "Oglala Lakota County, South Dakota": "46102",
    "Kusilvak Census Area, Alaska": "02158",
    "Chugach Census Area, Alaska": "02063",
    "Copper River Census Area, Alaska": "02066",
    "District of Columbia": "11001",
    "Kalawao County, Hawaii": "15005",
    "Loving County, Texas": "48301",
    "Los Angeles County, California": "06037",
}

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SourceAError(Exception):
    """Base exception for all Source A ingestion failures."""


class WikimediaAuthError(SourceAError):
    """Raised when authentication against the Wikimedia Enterprise API fails."""


class ArticleNotFoundError(SourceAError):
    """Raised when a county's Wikipedia article cannot be located."""


class EmptyIntroError(SourceAError):
    """Raised when the extracted/cleaned introduction text is empty."""


# --------------------------------------------------------------------------
# Wikimedia Enterprise API client
# --------------------------------------------------------------------------


class WikimediaEnterpriseClient:
    """Thin transport-layer client for the Wikimedia Enterprise API.

    Handles authentication and raw article retrieval only; HTML parsing and
    text cleaning are handled separately by the preprocessing functions.
    """

    def __init__(
        self,
        username: str,
        password: str,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the client with Wikimedia Enterprise credentials.

        Args:
            username: Wikimedia Enterprise account username.
            password: Wikimedia Enterprise account password.
            session: Optional pre-configured requests.Session, injectable for testing.
        """
        self._username = username
        self._password = password
        self._session = session or requests.Session()
        self._access_token: str | None = None

    @property
    def is_authenticated(self) -> bool:
        """Return True if a bearer token has been acquired."""
        return self._access_token is not None

    def authenticate(self) -> None:
        """Authenticate against the Wikimedia Enterprise API and store the bearer token.

        Raises:
            WikimediaAuthError: If the request fails or the response is missing
                an access token.
        """
        try:
            response = self._session.post(
                WIKIMEDIA_AUTH_URL,
                json={"username": self._username, "password": self._password},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise WikimediaAuthError(f"Authentication request failed: {exc}") from exc

        if response.status_code != 200:
            raise WikimediaAuthError(
                f"Authentication failed with status {response.status_code}: {response.text}"
            )

        token = response.json().get("access_token")
        if not token:
            raise WikimediaAuthError("Authentication response missing 'access_token'.")

        self._access_token = token
        logger.info("Authenticated with Wikimedia Enterprise API.")

    def get_article(self, article_name: str) -> dict[str, Any]:
        """Retrieve the raw article payload for a given article title.

        Args:
            article_name: Wikipedia article title, e.g. "Allegheny County, Pennsylvania".

        Returns:
            Parsed JSON article object as returned by the Wikimedia Enterprise API.

        Raises:
            WikimediaAuthError: If called before authenticate() or the token is rejected.
            ArticleNotFoundError: If the article does not exist (HTTP 404).
        """
        if not self.is_authenticated:
            raise WikimediaAuthError("Client is not authenticated; call authenticate() first.")

        url = WIKIMEDIA_ARTICLES_URL_TEMPLATE.format(name=article_name)
        try:
            response = self._session.post(
                url,
                json={
                    "filters": [
                        {"field": "is_part_of.identifier", "value": WIKIMEDIA_PROJECT_FILTER}
                    ]
                },
                headers={"Authorization": f"Bearer {self._access_token}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise SourceAError(f"Article request failed for '{article_name}': {exc}") from exc

        if response.status_code == 404:
            raise ArticleNotFoundError(f"Article not found: '{article_name}'")
        if response.status_code == 401:
            raise WikimediaAuthError(f"Bearer token rejected while fetching '{article_name}'")
        if response.status_code != 200:
            raise SourceAError(
                f"Unexpected status {response.status_code} fetching '{article_name}': "
                f"{response.text}"
            )

        articles = response.json()
        if not articles:
            raise ArticleNotFoundError(f"Article not found: '{article_name}'")
        return articles[0]


# --------------------------------------------------------------------------
# Text preprocessing
# --------------------------------------------------------------------------

_WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|]+\|)?([^\]]+)\]\]")
_CITATION_BRACKET_PATTERN = re.compile(r"\[\d+\]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_article_html(article_json: dict[str, Any]) -> str:
    """Extract the raw article body HTML from a Wikimedia Enterprise article payload.

    Args:
        article_json: Parsed JSON article object.

    Returns:
        Raw HTML string of the full article body.

    Raises:
        EmptyIntroError: If the article body HTML field is missing or empty.
    """
    html = article_json.get("article_body", {}).get("html", "")
    if not html:
        raise EmptyIntroError("Article payload has no 'article_body.html' content.")
    return html


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
        Cleaned, whitespace-normalized introductory text.

    Raises:
        EmptyIntroError: If the cleaned result is empty.
    """
    lead_html = isolate_lead_section(article_html)
    lead_soup = BeautifulSoup(lead_html, "html.parser")
    lead_soup = strip_non_narrative_elements(lead_soup)

    text = lead_soup.get_text(separator=" ")
    text = _WIKI_LINK_PATTERN.sub(r"\2", text)
    text = _CITATION_BRACKET_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    if not text:
        raise EmptyIntroError("Cleaned introduction text is empty.")
    return text


# --------------------------------------------------------------------------
# Embedding generation
# --------------------------------------------------------------------------


class BgeM3EmbeddingGenerator:
    """Wraps the BAAI/bge-m3 sentence-transformers model for dense embedding generation."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None) -> None:
        """Load the embedding model once at construction.

        Args:
            model_name: Hugging Face model identifier.
            device: Optional device override (e.g. "cpu", "cuda"); auto-detected if None.
        """
        logger.info("Loading embedding model '%s'...", model_name)
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a dense, non-normalized embedding vector.

        Relies on the tokenizer's native truncation at the model's max sequence
        length (bge-m3 supports up to 8,192 tokens); no manual chunking is applied.

        Args:
            text: Cleaned narrative text to embed.

        Returns:
            Raw (non-normalized) embedding vector of shape (EMBEDDING_DIM,).
        """
        vector = self._model.encode(text, normalize_embeddings=False)
        return np.asarray(vector, dtype=np.float32)

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        """Explicitly apply L2 normalization, mapping the vector onto the unit hypersphere.

        Args:
            vector: Raw embedding vector.

        Returns:
            L2-normalized embedding vector.

        Raises:
            ValueError: If the vector has zero norm.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot L2-normalize a zero-norm vector.")
        return vector / norm


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class CountyIngestionResult:
    """Holds the fully processed output for a single county."""

    county_name: str
    fips_code: str | None
    raw_intro_text: str
    embedding: list[float]


@dataclass
class IngestionSummary:
    """Tracks per-run success/failure outcomes across all counties."""

    succeeded: list[str]
    failed: dict[str, str]


def get_fips_code(county_name: str) -> str | None:
    """Look up the FIPS code for a county from FIPS_CROSSWALK.

    FIPS_CROSSWALK only covers the counties in TEST_COUNTIES; a full Census
    Bureau crosswalk is pending a later ingestion stage. Returns None for any
    county_name not present in the crosswalk.

    Args:
        county_name: County display name.

    Returns:
        FIPS code, or None if not in the crosswalk.
    """
    return FIPS_CROSSWALK.get(county_name)


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------


def process_county(
    county_name: str,
    client: WikimediaEnterpriseClient,
    embedder: BgeM3EmbeddingGenerator,
) -> CountyIngestionResult:
    """Run the full ingestion pipeline for a single county.

    Args:
        county_name: County display name, e.g. "Allegheny County, Pennsylvania".
        client: Authenticated Wikimedia Enterprise client.
        embedder: Loaded embedding generator.

    Returns:
        CountyIngestionResult with cleaned text and normalized embedding.

    Raises:
        ArticleNotFoundError: If the Wikipedia article cannot be found.
        EmptyIntroError: If the cleaned introduction text is empty.
        WikimediaAuthError: If authentication is rejected mid-run.
    """
    article_name = INDEPENDENT_CITY_ARTICLE_LOOKUP.get(county_name, county_name)
    article_json = client.get_article(article_name)
    article_html = extract_article_html(article_json)
    intro_text = clean_intro_text(article_html)

    raw_vector = embedder.encode(intro_text)
    normalized_vector = embedder.l2_normalize(raw_vector)

    return CountyIngestionResult(
        county_name=county_name,
        fips_code=get_fips_code(county_name),
        raw_intro_text=intro_text,
        embedding=normalized_vector.tolist(),
    )


def run_pipeline(
    county_names: list[str],
    client: WikimediaEnterpriseClient,
    embedder: BgeM3EmbeddingGenerator,
) -> tuple[list[CountyIngestionResult], IngestionSummary]:
    """Process all counties, isolating per-county failures from the batch.

    Args:
        county_names: List of county display names to process.
        client: Authenticated Wikimedia Enterprise client.
        embedder: Loaded embedding generator.

    Returns:
        Tuple of (successful results, run summary).

    Raises:
        WikimediaAuthError: Propagated immediately since it is unrecoverable for the batch.
    """
    results: list[CountyIngestionResult] = []
    summary = IngestionSummary(succeeded=[], failed={})

    for county_name in county_names:
        logger.info("Processing '%s'...", county_name)
        try:
            result = process_county(county_name, client, embedder)
        except (ArticleNotFoundError, EmptyIntroError) as exc:
            logger.warning("Skipping '%s': %s", county_name, exc)
            summary.failed[county_name] = str(exc)
            continue
        except WikimediaAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - defensive catch-all per county
            logger.error("Unexpected failure for '%s': %s", county_name, exc)
            summary.failed[county_name] = str(exc)
            continue

        results.append(result)
        summary.succeeded.append(county_name)

    return results, summary


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def build_dataframe(results: list[CountyIngestionResult]) -> pd.DataFrame:
    """Assemble ingestion results into a pandas DataFrame.

    Args:
        results: List of per-county ingestion results.

    Returns:
        DataFrame with columns: county_name, fips_code, raw_intro_text, embedding.
    """
    return pd.DataFrame(
        {
            "county_name": [r.county_name for r in results],
            "fips_code": [r.fips_code for r in results],
            "raw_intro_text": [r.raw_intro_text for r in results],
            "embedding": [r.embedding for r in results],
        }
    )


def export_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the ingestion DataFrame to a local Parquet file.

    Args:
        df: DataFrame to export.
        output_path: Destination Parquet file path.
    """
    df.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    """Run the Source A ingestion pipeline over the test county list."""
    configure_logging()

    username = os.environ.get("WIKIMEDIA_USERNAME")
    password = os.environ.get("WIKIMEDIA_PASSWORD")
    if not username or not password:
        logger.error("WIKIMEDIA_USERNAME / WIKIMEDIA_PASSWORD not set in environment.")
        sys.exit(1)

    client = WikimediaEnterpriseClient(username, password)
    try:
        client.authenticate()
    except WikimediaAuthError as exc:
        logger.error("Authentication failed; aborting: %s", exc)
        sys.exit(1)

    embedder = BgeM3EmbeddingGenerator(device="cpu")

    try:
        results, summary = run_pipeline(TEST_COUNTIES, client, embedder)
    except WikimediaAuthError as exc:
        logger.error("Authentication rejected mid-run; aborting: %s", exc)
        sys.exit(1)

    df = build_dataframe(results)
    export_to_parquet(df, OUTPUT_PARQUET_PATH)

    logger.info("Succeeded: %d, Failed: %d", len(summary.succeeded), len(summary.failed))
    for county, reason in summary.failed.items():
        logger.warning("  %s -> %s", county, reason)


if __name__ == "__main__":
    main()
