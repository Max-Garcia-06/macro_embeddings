"""Source A ingestion pipeline for E_macro: Wikipedia Introductory Corpora.

Queries the English Wikimedia Enterprise API for county Wikipedia articles,
isolates the lead/introductory section (excluding infobox, body sections, and
metadata), embeds the cleaned text with BAAI/bge-m3, L2-normalizes the
resulting vector, and stores the result set as a local Parquet file.

Requires WIKIMEDIA_USERNAME and WIKIMEDIA_PASSWORD environment variables set
to valid Wikimedia Enterprise credentials.
"""

from __future__ import annotations

import io
import logging
import os
import random
import re
import sys
import zipfile
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

logger = logging.getLogger(__name__)

GAZETTEER_URL: str = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2024_Gazetteer/2024_Gaz_counties_national.zip"
)
COUNTY_CROSSWALK_CACHE_PATH: Path = (
    Path(__file__).resolve().parent / "county_crosswalk.parquet"
)

# USPS state/territory abbreviation -> full name, as used in the Census
# Gazetteer counties file.
_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "PR": "Puerto Rico",
}


def _build_county_display_name(name: str, usps: str) -> str:
    """Construct a county display name matching Wikipedia article-title conventions.

    Args:
        name: Census Gazetteer `NAME` field, e.g. "Richmond city".
        usps: Two-letter USPS state/territory abbreviation.

    Returns:
        Display name, e.g. "Richmond City, Virginia".
    """
    if usps == "DC":
        return "District of Columbia"
    if usps == "PR":
        return f"{name.removesuffix(' Municipio')}, Puerto Rico"
    if name.endswith(" city"):
        name = f"{name[: -len(' city')]} City"
    return f"{name}, {_STATE_NAMES[usps]}"


def fetch_county_crosswalk(cache_path: Path) -> pd.DataFrame:
    """Load the full county name/FIPS crosswalk, downloading and caching on first use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with columns `county_name`, `fips_code` covering all US
        counties, county-equivalents, and Puerto Rico municipios.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    logger.info("Downloading Census Gazetteer counties file...")
    response = requests.get(GAZETTEER_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        (member_name,) = archive.namelist()
        with archive.open(member_name) as f:
            gaz = pd.read_csv(f, sep="\t", encoding="utf-8", dtype={"GEOID": str})

    gaz.columns = gaz.columns.str.strip()
    gaz = gaz[gaz["USPS"] != "PR"].reset_index(drop=True)
    crosswalk = pd.DataFrame(
        {
            "county_name": [
                _build_county_display_name(n, u)
                for n, u in zip(gaz["NAME"], gaz["USPS"])
            ],
            "fips_code": gaz["GEOID"].str.strip(),
        }
    )

    crosswalk.to_parquet(cache_path, engine="pyarrow", index=False)
    logger.info("Cached county crosswalk to %s", cache_path)
    return crosswalk


_COUNTY_CROSSWALK_DF = fetch_county_crosswalk(COUNTY_CROSSWALK_CACHE_PATH)
ALL_COUNTIES: list[str] = _COUNTY_CROSSWALK_DF["county_name"].tolist()

# Exploratory sample size for a partial ingestion run: enough counties spread
# across every state for the PCA visualization to show real regional
# structure, without paying the cost of all ~3,222 counties. Fixed seed for
# reproducibility between runs.
SAMPLE_SIZE: int = 300
SAMPLE_SEED: int = 42
SAMPLE_COUNTIES: list[str] = random.Random(SAMPLE_SEED).sample(ALL_COUNTIES, SAMPLE_SIZE)

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
    "San Francisco County, California": "San Francisco",
    "Philadelphia County, Pennsylvania": "Philadelphia",
    "Denver County, Colorado": "Denver",
    "Broomfield County, Colorado": "Broomfield, Colorado",
    "District of Columbia": "Washington, D.C.",
    "Yukon-Koyukuk Census Area, Alaska": "Yukon–Koyukuk Census Area, Alaska",
    "Coos County, New Hampshire": "Coös County, New Hampshire",
    # Virginia independent cities: Wikipedia article titles drop the " City"
    # suffix present in the Census Gazetteer NAME field (e.g. "Richmond City,
    # Virginia" -> "Richmond, Virginia"). All 37 of Virginia's independent
    # cities need this mapping; without it every one fails with a 404.
    "Alexandria City, Virginia": "Alexandria, Virginia",
    "Bristol City, Virginia": "Bristol, Virginia",
    "Buena Vista City, Virginia": "Buena Vista, Virginia",
    "Charlottesville City, Virginia": "Charlottesville, Virginia",
    "Chesapeake City, Virginia": "Chesapeake, Virginia",
    "Colonial Heights City, Virginia": "Colonial Heights, Virginia",
    "Covington City, Virginia": "Covington, Virginia",
    "Danville City, Virginia": "Danville, Virginia",
    "Emporia City, Virginia": "Emporia, Virginia",
    "Fairfax City, Virginia": "Fairfax, Virginia",
    "Falls Church City, Virginia": "Falls Church, Virginia",
    "Fredericksburg City, Virginia": "Fredericksburg, Virginia",
    "Galax City, Virginia": "Galax, Virginia",
    "Hampton City, Virginia": "Hampton, Virginia",
    "Harrisonburg City, Virginia": "Harrisonburg, Virginia",
    "Hopewell City, Virginia": "Hopewell, Virginia",
    "Lexington City, Virginia": "Lexington, Virginia",
    "Lynchburg City, Virginia": "Lynchburg, Virginia",
    "Manassas City, Virginia": "Manassas, Virginia",
    "Manassas Park City, Virginia": "Manassas Park, Virginia",
    "Martinsville City, Virginia": "Martinsville, Virginia",
    "Newport News City, Virginia": "Newport News, Virginia",
    "Norfolk City, Virginia": "Norfolk, Virginia",
    "Norton City, Virginia": "Norton, Virginia",
    "Petersburg City, Virginia": "Petersburg, Virginia",
    "Poquoson City, Virginia": "Poquoson, Virginia",
    "Portsmouth City, Virginia": "Portsmouth, Virginia",
    "Radford City, Virginia": "Radford, Virginia",
    "Richmond City, Virginia": "Richmond, Virginia",
    "Roanoke City, Virginia": "Roanoke, Virginia",
    "Salem City, Virginia": "Salem, Virginia",
    "Staunton City, Virginia": "Staunton, Virginia",
    "Suffolk City, Virginia": "Suffolk, Virginia",
    "Virginia Beach City, Virginia": "Virginia Beach, Virginia",
    "Waynesboro City, Virginia": "Waynesboro, Virginia",
    "Williamsburg City, Virginia": "Williamsburg, Virginia",
    "Winchester City, Virginia": "Winchester, Virginia",
}

# Full FIPS crosswalk derived from the Census Gazetteer counties file (see
# fetch_county_crosswalk), covering every county in ALL_COUNTIES.
FIPS_CROSSWALK: dict[str, str] = dict(
    zip(_COUNTY_CROSSWALK_DF["county_name"], _COUNTY_CROSSWALK_DF["fips_code"])
)


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

    Returns None for any county_name not present in the crosswalk.

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
    embedding_text = strip_self_reference(intro_text, county_name)
    embedding_text = strip_boilerplate_phrasing(embedding_text)

    raw_vector = embedder.encode(embedding_text)
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
    """Run the Source A ingestion pipeline over a sample of US counties."""
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
        results, summary = run_pipeline(ALL_COUNTIES, client, embedder)
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
