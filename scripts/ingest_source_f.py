"""Source F ingestion pipeline for E_macro: USDA ERS County Typology Codes.

Downloads the 2025-edition USDA Economic Research Service County Typology
Codes ("Structural Resilience Baseline"), pivots the long-format source file
into one row per county, one-hot encodes the mutually-exclusive Industry
Dependence category, and joins the result onto the project's county/FIPS
crosswalk. A single static file download -- no API key or rate limiting is
required, and the typology is a decennial/annual-refresh baseline anchor
rather than a time series.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TYPOLOGY_CSV_URL: str = (
    "https://www.ers.usda.gov/media/6174/ers-county-typology-codes-2025-edition.csv"
)
REQUEST_TIMEOUT_SECONDS: int = 30

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_f_usda_typology.parquet"
COUNTY_CROSSWALK_CACHE_PATH: Path = DATA_DIR / "county_crosswalk.parquet"

logger = logging.getLogger(__name__)

# Industry_Dependence_2025 code -> dominant-industry name. Mutually
# exclusive; 0 means no single industry dominates the county's economy.
_INDUSTRY_DEPENDENCE_CODES: dict[int, str] = {
    0: "none",
    1: "farming",
    2: "mining",
    3: "manufacturing",
    4: "government",
    5: "recreation",
}

# Binary economic/demographic attributes in the ERS long-format file, mapped
# to their output column name.
_BINARY_ATTRIBUTE_COLUMNS: dict[str, str] = {
    "High_Farming_2025": "high_farming",
    "High_Mining_2025": "high_mining",
    "High_Manufacturing_2025": "high_manufacturing",
    "High_Government_2025": "high_government",
    "High_Recreation_2025": "high_recreation",
    "Nonspecialized_2025": "nonspecialized",
    "Low_PostSecondary_Ed_2025": "low_postsecondary_ed",
    "Low_Employment_2025": "low_employment",
    "Population_Loss_2025": "population_loss",
    "Housing_Stress_2025": "housing_stress",
    "Retirement_Destination_2025": "retirement_destination",
    "Persistent_Poverty_1721": "persistent_poverty",
}

# ERS sentinel codes for "not classified" (economic typology only applies to
# a subset of counties) and "insufficient data"; both map to a null value
# rather than a boolean.
_MISSING_VALUE_CODES: frozenset[int] = frozenset({-1, 99})


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SourceFError(Exception):
    """Base exception for all Source F ingestion failures."""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_typology_csv(url: str) -> pd.DataFrame:
    """Download and parse the ERS County Typology Codes long-format CSV.

    The source file has one row per (county, attribute) pair, e.g. columns
    `FIPStxt, State, County_Name, Metro2023, Attribute, Value`.

    Args:
        url: Direct URL to the ERS County Typology Codes CSV export.

    Returns:
        Raw long-format DataFrame as published.

    Raises:
        SourceFError: If the download fails.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceFError(f"Failed to download County Typology Codes CSV: {exc}") from exc

    return pd.read_csv(io.StringIO(response.text), dtype={"FIPStxt": str})


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def _to_nullable_bool(series: pd.Series) -> pd.Series:
    """Map a 0/1/sentinel integer series to a nullable boolean series."""
    cleaned = series.mask(series.isin(_MISSING_VALUE_CODES))
    return cleaned.map({0: False, 1: True}).astype("boolean")


def _one_hot_industry_dependence(series: pd.Series) -> pd.DataFrame:
    """One-hot encode the mutually-exclusive Industry_Dependence_2025 code.

    Args:
        series: Raw Industry_Dependence_2025 values, indexed by FIPS code.

    Returns:
        DataFrame with one nullable-boolean column per dependence category,
        named `industry_dependence_{name}`; all columns are null for
        counties with a sentinel (not-classified) code.
    """
    cleaned = series.mask(series.isin(_MISSING_VALUE_CODES))
    return pd.DataFrame(
        {
            f"industry_dependence_{name}": (cleaned == code).astype("boolean").mask(cleaned.isna())
            for code, name in _INDUSTRY_DEPENDENCE_CODES.items()
        }
    )


def transform_typology(raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format ERS file into one row per FIPS code with typed columns.

    Args:
        raw: Long-format DataFrame as returned by download_typology_csv.

    Returns:
        Wide DataFrame indexed by `fips_code` with `metro_2023`, one column
        per binary attribute, and one-hot `industry_dependence_*` columns.
    """
    wide = raw.pivot(index="FIPStxt", columns="Attribute", values="Value")

    result = pd.DataFrame(index=wide.index)
    result["metro_2023"] = raw.drop_duplicates("FIPStxt").set_index("FIPStxt")["Metro2023"].astype(
        "boolean"
    )
    for source_col, output_col in _BINARY_ATTRIBUTE_COLUMNS.items():
        result[output_col] = _to_nullable_bool(wide[source_col])
    result = result.join(_one_hot_industry_dependence(wide["Industry_Dependence_2025"]))

    result.index.name = "fips_code"
    return result.reset_index()


def join_county_names(typology: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join typology rows onto the project's county/FIPS crosswalk.

    An inner join: the ERS file also carries 8 legacy Connecticut county FIPS
    codes (superseded by the 9 Connecticut planning regions the crosswalk
    uses instead), which are dropped here rather than mapped, since the
    planning-region rows already provide full Connecticut coverage.

    Args:
        typology: Wide typology DataFrame from transform_typology, with a
            `fips_code` column.
        crosswalk: DataFrame with `county_name`, `fips_code` columns.

    Returns:
        DataFrame with `county_name` first, followed by all typology columns.
    """
    merged = crosswalk.merge(typology, on="fips_code", how="inner")

    unmatched = set(crosswalk["fips_code"]) - set(merged["fips_code"])
    if unmatched:
        logger.warning("%d crosswalk counties had no typology match: %s", len(unmatched), sorted(unmatched))

    return merged


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


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
    """Run the Source F ingestion pipeline over all US counties."""
    configure_logging()

    if not COUNTY_CROSSWALK_CACHE_PATH.exists():
        logger.error(
            "County crosswalk not found at %s; run ingest_source_a.py or "
            "ingest_source_c.py first to populate it.",
            COUNTY_CROSSWALK_CACHE_PATH,
        )
        sys.exit(1)
    crosswalk = pd.read_parquet(COUNTY_CROSSWALK_CACHE_PATH)

    try:
        raw = download_typology_csv(TYPOLOGY_CSV_URL)
    except SourceFError as exc:
        logger.error("Aborting: %s", exc)
        sys.exit(1)

    typology = transform_typology(raw)
    df = join_county_names(typology, crosswalk)

    export_to_parquet(df, OUTPUT_PARQUET_PATH)
    logger.info("Covered %d of %d crosswalk counties.", len(df), len(crosswalk))


if __name__ == "__main__":
    main()
