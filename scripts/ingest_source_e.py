"""Source E ingestion pipeline for E_macro: IRS Statistics of Income Panel
Data (Capital Composition).

Downloads the IRS SOI Division's Tax Year 2022 pre-aggregated county file
(`22incyallnoagi.csv` -- `AGI_STUB` fixed at 0, i.e. IRS's own county totals,
not the 8-AGI-bracket breakdown) and computes the ratio of investment income
(net capital gains + qualified dividends) to W-2 wage income per county, per
`docs/plans/ingestion_recon.md (Source E)`'s Phase 0 findings.

No credentials required. `www.irs.gov` has no bot protection on either the
landing pages or the `/pub/irs-soi/*.csv` file host -- plain `requests`
works throughout, unlike Source B/D's landing-page/data-host split.

Target columns are referenced via `SOI_COLUMN_MAP` (conceptual name -> SOI
variable code) rather than positional indexing, per the pre-scoping spec's
proposed mitigation for upstream schema mutation -- a future year renaming
or dropping one of these fields fails loudly (KeyError) instead of silently
misreading a shifted column.

Known limitation, not fixable from this file alone (docs/plans/ingestion_recon.md (Source E) Risk
3): unlike BLS QCEW's `disclosure_code`, the IRS county file carries no
suppression flag. A handful of ultra-low-population counties show exact
zero amounts that could be a true zero or a suppressed small-cell value --
both are written identically. Shipped as-is rather than guessed at.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from pillar_vintage import stamp_vintage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TAX_YEAR: int = 2022
SOURCE_CSV_URL: str = "https://www.irs.gov/pub/irs-soi/22incyallnoagi.csv"
REQUEST_TIMEOUT_SECONDS: int = 60
CSV_ENCODING: str = "latin-1"

STATE_TOTAL_COUNTYFIPS: str = "000"

# Conceptual name -> raw SOI variable code, per docs/plans/ingestion_recon.md (Source E)'s Risk 2
# mitigation (insulates the pipeline from future line-item renumbering).
SOI_COLUMN_MAP: dict[str, str] = {
    "num_returns": "N1",
    "agi_thousands": "A00100",
    "wages_salaries_thousands": "A00200",
    "qualified_dividends_thousands": "A00650",
    "net_cap_gain_thousands": "A01000",
}

# Below this return count, `capital_to_wage_ratio` is dominated by a handful of
# one-time transactions (typically farm/ranch land sales) rather than a
# sustained investment-income base -- see source-e-findings.md finding 3, where
# seven of the top-25 highest-ratio counties fall under this threshold against a
# national median of ~11,700 returns.
LOW_RETURN_THRESHOLD: int = 2_200

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_e_irs_soi.parquet"
COUNTY_CROSSWALK_CACHE_PATH: Path = DATA_DIR / "county_crosswalk.parquet"

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


class SourceEError(Exception):
    """Base exception for all Source E ingestion failures."""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_county_csv(url: str) -> pd.DataFrame:
    """Download the IRS SOI pre-aggregated county CSV into a DataFrame.

    Args:
        url: Direct URL to `22incyallnoagi.csv`.

    Returns:
        Raw DataFrame, all columns string-typed except where IRS ships
        numeric literals (read with pandas' default inference).

    Raises:
        SourceEError: If the download or the required columns are missing.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceEError(f"Failed to download IRS SOI county file: {exc}") from exc

    from io import StringIO

    df = pd.read_csv(
        StringIO(response.content.decode(CSV_ENCODING)),
        dtype={"STATEFIPS": str, "COUNTYFIPS": str},
    )

    missing = [code for code in SOI_COLUMN_MAP.values() if code not in df.columns]
    if missing:
        raise SourceEError(
            f"IRS SOI file is missing expected columns {missing} -- "
            "upstream schema likely changed, see docs/plans/ingestion_recon.md (Source E) Risk 2."
        )
    return df


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop state-total rows and derive the capital-to-wage ratio.

    Args:
        raw: DataFrame from download_county_csv.

    Returns:
        DataFrame with `fips_code`, conceptual income columns,
        `capital_to_wage_ratio`, and `low_return_flag`.
    """
    counties = raw.loc[raw["COUNTYFIPS"] != STATE_TOTAL_COUNTYFIPS].copy()
    counties["fips_code"] = counties["STATEFIPS"] + counties["COUNTYFIPS"]

    for concept, code in SOI_COLUMN_MAP.items():
        counties[concept] = pd.to_numeric(counties[code], errors="coerce")

    counties["capital_to_wage_ratio"] = (
        counties["net_cap_gain_thousands"] + counties["qualified_dividends_thousands"]
    ) / counties["wages_salaries_thousands"]
    counties["low_return_flag"] = counties["num_returns"] < LOW_RETURN_THRESHOLD

    return counties[
        ["fips_code", *SOI_COLUMN_MAP.keys(), "capital_to_wage_ratio", "low_return_flag"]
    ]


def join_county_names(transformed: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join transformed IRS SOI rows onto the project's county/FIPS crosswalk.

    Args:
        transformed: DataFrame from transform, with a `fips_code` column.
        crosswalk: DataFrame with `county_name`, `fips_code` columns.

    Returns:
        DataFrame with `county_name` first, followed by all IRS SOI columns.
    """
    merged = crosswalk.merge(transformed, on="fips_code", how="inner")

    unmatched = set(crosswalk["fips_code"]) - set(merged["fips_code"])
    if unmatched:
        logger.warning(
            "%d crosswalk counties had no IRS SOI county-level match: %s",
            len(unmatched), sorted(unmatched),
        )

    return merged


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def export_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the ingestion DataFrame to a local Parquet file.

    The pillar's `as_of_date` is stamped on before writing, so the vintage
    travels with the data rather than only with the docs (`pillar_vintage`).

    Args:
        df: DataFrame to export.
        output_path: Destination Parquet file path.
    """
    stamp_vintage(df, "E").to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    """Run the Source E ingestion pipeline over all US counties."""
    configure_logging()

    if not COUNTY_CROSSWALK_CACHE_PATH.exists():
        logger.error(
            "County crosswalk not found at %s; run ingest_source_a.py or "
            "ingest_source_c.py first to populate it.",
            COUNTY_CROSSWALK_CACHE_PATH,
        )
        sys.exit(1)
    crosswalk = pd.read_parquet(COUNTY_CROSSWALK_CACHE_PATH)

    logger.info("Downloading IRS SOI Tax Year %d county file from %s...", TAX_YEAR, SOURCE_CSV_URL)
    try:
        raw = download_county_csv(SOURCE_CSV_URL)
    except SourceEError as exc:
        logger.error("Aborting: %s", exc)
        sys.exit(1)

    logger.info("Downloaded %d raw rows.", len(raw))

    transformed = transform(raw)
    df = join_county_names(transformed, crosswalk)

    export_to_parquet(df, OUTPUT_PARQUET_PATH)
    logger.info("Covered %d of %d crosswalk counties.", len(df), len(crosswalk))


if __name__ == "__main__":
    main()
