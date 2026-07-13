"""Source C ingestion pipeline for E_macro: FRED Time-Series Slope Derivatives.

Queries the Federal Reserve Economic Data (FRED) API for county-level annual
unemployment rate and real GDP series, computes the rolling 3-year first
derivative of each, and stores the result set as a local Parquet file.

Both series are pulled at annual frequency: FRED's monthly county
unemployment series use ad-hoc state/county-abbreviation codes that are not
derivable from a FIPS code, while the annual series (`LAUCN{FIPS}0000000003A`)
and the GDP series (`REALGDPALL{FIPS}`) are both cleanly FIPS-derivable.
Annual data has no seasonal component, so no deseasonalization step is
needed before differencing.

Requires FRED_API_KEY set in the environment.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FRED_OBSERVATIONS_URL: str = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT_SECONDS: int = 30

# Conservative token-bucket rate limit; FRED's actual limit is 120 req/min.
RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
_MIN_SECONDS_BETWEEN_REQUESTS: float = 60.0 / RATE_LIMIT_REQUESTS_PER_MINUTE

# Retries for transient failures (network errors, 5xx) only -- a 400 "series
# does not exist" is a permanent condition and is never retried.
TRANSIENT_ERROR_MAX_ATTEMPTS: int = 3
TRANSIENT_ERROR_BACKOFF_SECONDS: float = 2.0

VELOCITY_WINDOW_YEARS: int = 3

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_c_fred.parquet"
COUNTY_CROSSWALK_CACHE_PATH: Path = DATA_DIR / "county_crosswalk.parquet"

logger = logging.getLogger(__name__)


def unemployment_series_id(fips_code: str) -> str:
    """Return the FRED series ID for a county's annual unemployment rate."""
    return f"LAUCN{fips_code}0000000003A"


def gdp_series_id(fips_code: str) -> str:
    """Return the FRED series ID for a county's annual real GDP."""
    return f"REALGDPALL{fips_code}"


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SourceCError(Exception):
    """Base exception for all Source C ingestion failures."""


class SeriesNotFoundError(SourceCError):
    """Raised when a FRED series does not exist for a given ID."""


class InsufficientHistoryError(SourceCError):
    """Raised when a series has fewer than VELOCITY_WINDOW_YEARS + 1 annual observations."""


# --------------------------------------------------------------------------
# FRED API client
# --------------------------------------------------------------------------


class FredClient:
    """Thin transport-layer client for the FRED REST API with request-rate limiting."""

    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        """Initialize the client with a FRED developer API key.

        Args:
            api_key: FRED developer API key.
            session: Optional pre-configured requests.Session, injectable for testing.
        """
        self._api_key = api_key
        self._session = session or requests.Session()
        self._last_request_time: float | None = None

    def _throttle(self) -> None:
        """Sleep as needed to stay under the configured requests-per-minute rate."""
        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            remaining = _MIN_SECONDS_BETWEEN_REQUESTS - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _get_with_retry(self, series_id: str) -> requests.Response:
        """Issue the observations request, retrying transient failures only.

        A network error or 5xx response is retried up to
        TRANSIENT_ERROR_MAX_ATTEMPTS times with a fixed backoff; any other
        response (200, or a 4xx like the "series does not exist" 400) is
        returned immediately without retrying, since those are permanent
        conditions the caller must handle, not blips to wait out.

        Args:
            series_id: FRED series ID.

        Returns:
            The final HTTP response.

        Raises:
            SourceCError: If every attempt fails with a network error or 5xx.
        """
        last_exc: Exception | None = None
        for attempt in range(1, TRANSIENT_ERROR_MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                response = self._session.get(
                    FRED_OBSERVATIONS_URL,
                    params={
                        "series_id": series_id,
                        "api_key": self._api_key,
                        "file_type": "json",
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                last_exc = exc
            else:
                if response.status_code < 500:
                    return response
                last_exc = SourceCError(
                    f"Status {response.status_code} fetching '{series_id}': {response.text}"
                )

            if attempt < TRANSIENT_ERROR_MAX_ATTEMPTS:
                logger.warning(
                    "Transient failure fetching '%s' (attempt %d/%d): %s -- retrying",
                    series_id,
                    attempt,
                    TRANSIENT_ERROR_MAX_ATTEMPTS,
                    last_exc,
                )
                time.sleep(TRANSIENT_ERROR_BACKOFF_SECONDS)

        raise SourceCError(f"Request failed for '{series_id}' after {TRANSIENT_ERROR_MAX_ATTEMPTS} attempts: {last_exc}") from last_exc

    def get_annual_observations(self, series_id: str) -> pd.Series:
        """Fetch all annual observations for a FRED series, indexed by year.

        Args:
            series_id: FRED series ID, e.g. "LAUCN010010000000003A".

        Returns:
            Series of float values indexed by integer year, sorted ascending.

        Raises:
            SeriesNotFoundError: If the series does not exist (FRED 400 "does not exist").
            SourceCError: On a request failure that persists across retries.
        """
        response = self._get_with_retry(series_id)

        if response.status_code == 400 and "does not exist" in response.text:
            raise SeriesNotFoundError(f"Series not found: '{series_id}'")
        if response.status_code != 200:
            raise SourceCError(
                f"Unexpected status {response.status_code} fetching '{series_id}': "
                f"{response.text}"
            )

        payload = response.json()
        observations = payload.get("observations", [])
        years: list[int] = []
        values: list[float] = []
        for obs in observations:
            if obs["value"] == ".":
                continue
            years.append(int(obs["date"][:4]))
            values.append(float(obs["value"]))

        return pd.Series(values, index=pd.Index(years, name="year")).sort_index()


# --------------------------------------------------------------------------
# Velocity computation
# --------------------------------------------------------------------------


def compute_velocity(annual_series: pd.Series) -> tuple[float, float, int]:
    """Compute the rolling 3-year first derivative over the most recent window.

    Args:
        annual_series: Float values indexed by integer year, sorted ascending.

    Returns:
        Tuple of (velocity, latest_value, latest_year), where velocity is the
        average annual change over the trailing VELOCITY_WINDOW_YEARS years.

    Raises:
        InsufficientHistoryError: If fewer than VELOCITY_WINDOW_YEARS + 1
            observations are available.
    """
    if len(annual_series) < VELOCITY_WINDOW_YEARS + 1:
        raise InsufficientHistoryError(
            f"Need at least {VELOCITY_WINDOW_YEARS + 1} annual observations, "
            f"got {len(annual_series)}."
        )

    latest_year = annual_series.index[-1]
    latest_value = annual_series.iloc[-1]
    prior_value = annual_series.iloc[-1 - VELOCITY_WINDOW_YEARS]
    velocity = (latest_value - prior_value) / VELOCITY_WINDOW_YEARS

    return velocity, latest_value, int(latest_year)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass
class CountyVelocityResult:
    """Holds the fully processed output for a single county."""

    county_name: str
    fips_code: str
    unemployment_velocity: float | None
    unemployment_rate_latest: float | None
    unemployment_latest_year: int | None
    gdp_velocity: float | None
    gdp_latest: float | None
    gdp_latest_year: int | None


@dataclass
class IngestionSummary:
    """Tracks per-county/per-series success and failure outcomes across a run."""

    succeeded: list[str]
    partial: dict[str, str]
    failed: dict[str, str]


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------


def _fetch_velocity(
    client: FredClient, series_id: str
) -> tuple[float | None, float | None, int | None, str | None]:
    """Fetch a series and compute its velocity, isolating per-series failures.

    Returns:
        Tuple of (velocity, latest_value, latest_year, error_message). All
        four are None/None/None/None on success with error_message set to
        None; on failure the first three are None and error_message is set.
    """
    try:
        series = client.get_annual_observations(series_id)
        velocity, latest_value, latest_year = compute_velocity(series)
        return velocity, latest_value, latest_year, None
    except SourceCError as exc:
        return None, None, None, str(exc)


def process_county(county_name: str, fips_code: str, client: FredClient) -> CountyVelocityResult:
    """Run the full ingestion pipeline for a single county.

    Unemployment and GDP series are fetched and processed independently: a
    missing GDP series (a real, expected condition for some small independent
    cities) still yields a partial result with unemployment data populated,
    rather than dropping the county entirely.

    Args:
        county_name: County display name, e.g. "Allegheny County, Pennsylvania".
        fips_code: 5-digit FIPS code.
        client: FRED API client.

    Returns:
        CountyVelocityResult with whatever series succeeded; unavailable
        series are left as None.
    """
    unemployment_velocity, unemployment_latest, unemployment_year, unemployment_error = (
        _fetch_velocity(client, unemployment_series_id(fips_code))
    )
    gdp_velocity, gdp_latest, gdp_year, gdp_error = _fetch_velocity(
        client, gdp_series_id(fips_code)
    )

    if unemployment_error:
        logger.warning("'%s' unemployment series unavailable: %s", county_name, unemployment_error)
    if gdp_error:
        logger.warning("'%s' GDP series unavailable: %s", county_name, gdp_error)

    return CountyVelocityResult(
        county_name=county_name,
        fips_code=fips_code,
        unemployment_velocity=unemployment_velocity,
        unemployment_rate_latest=unemployment_latest,
        unemployment_latest_year=unemployment_year,
        gdp_velocity=gdp_velocity,
        gdp_latest=gdp_latest,
        gdp_latest_year=gdp_year,
    )


def run_pipeline(
    counties: list[tuple[str, str]], client: FredClient
) -> tuple[list[CountyVelocityResult], IngestionSummary]:
    """Process all counties, isolating per-county/per-series failures from the batch.

    Args:
        counties: List of (county_name, fips_code) pairs to process.
        client: FRED API client.

    Returns:
        Tuple of (all results, run summary). A county is "succeeded" if both
        series resolved, "partial" if exactly one did, "failed" if neither did.
    """
    results: list[CountyVelocityResult] = []
    summary = IngestionSummary(succeeded=[], partial={}, failed={})

    for county_name, fips_code in counties:
        logger.info("Processing '%s' (%s)...", county_name, fips_code)
        result = process_county(county_name, fips_code, client)
        results.append(result)

        has_unemployment = result.unemployment_velocity is not None
        has_gdp = result.gdp_velocity is not None
        if has_unemployment and has_gdp:
            summary.succeeded.append(county_name)
        elif has_unemployment or has_gdp:
            missing = "gdp" if has_unemployment else "unemployment"
            summary.partial[county_name] = f"missing {missing} series"
        else:
            summary.failed[county_name] = "no series available"

    return results, summary


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def build_dataframe(results: list[CountyVelocityResult]) -> pd.DataFrame:
    """Assemble ingestion results into a pandas DataFrame.

    Args:
        results: List of per-county ingestion results.

    Returns:
        DataFrame with columns: county_name, fips_code, unemployment_velocity,
        unemployment_rate_latest, unemployment_latest_year, gdp_velocity,
        gdp_latest, gdp_latest_year.
    """
    return pd.DataFrame(
        {
            "county_name": [r.county_name for r in results],
            "fips_code": [r.fips_code for r in results],
            "unemployment_velocity": [r.unemployment_velocity for r in results],
            "unemployment_rate_latest": [r.unemployment_rate_latest for r in results],
            "unemployment_latest_year": [r.unemployment_latest_year for r in results],
            "gdp_velocity": [r.gdp_velocity for r in results],
            "gdp_latest": [r.gdp_latest for r in results],
            "gdp_latest_year": [r.gdp_latest_year for r in results],
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
    """Run the Source C ingestion pipeline over all US counties."""
    configure_logging()

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not set in environment.")
        sys.exit(1)

    crosswalk = pd.read_parquet(COUNTY_CROSSWALK_CACHE_PATH)
    counties = list(zip(crosswalk["county_name"], crosswalk["fips_code"]))

    client = FredClient(api_key)
    results, summary = run_pipeline(counties, client)

    df = build_dataframe(results)
    export_to_parquet(df, OUTPUT_PARQUET_PATH)

    logger.info(
        "Full: %d, Partial: %d, Failed: %d",
        len(summary.succeeded),
        len(summary.partial),
        len(summary.failed),
    )
    for county, reason in summary.partial.items():
        logger.warning("  partial: %s -> %s", county, reason)
    for county, reason in summary.failed.items():
        logger.warning("  failed: %s -> %s", county, reason)


if __name__ == "__main__":
    main()
