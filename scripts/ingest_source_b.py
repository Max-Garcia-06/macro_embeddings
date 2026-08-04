"""Source B ingestion pipeline for E_macro: BLS Quarterly Census of Employment
and Wages (Industrial Core).

Downloads BLS QCEW's bulk `{year}_qtrly_singlefile.zip` (all quarters, areas,
ownerships, industries, and size classes in one file) and stream-filters it to
2025 Q4, Private ownership (own_code=5), county-level rows (agglvl_code=74)
across the 20 primary 2-digit NAICS sectors. Individual per-industry-slice
downloads were tried first but 3 of the 20 combined codes (31-33, 44-45,
48-49) 404 as slice URLs -- see docs/plans/ingestion_recon.md (Source B)'s Phase 1b for the detour --
so the bulk file is filtered locally instead.

Pivots the long-format (one row per county x sector) result into one row per
county: `lq_emp_{naics2}` (that sector's Location Quotient, using the
quarter's third-month value) plus a matching `disclosure_{naics2}` flag (True
where BLS suppressed the cell for small-employer privacy, per
`disclosure_code == "N"`). Suppressed cells are left null rather than
estimated -- docs/plans/ingestion_recon.md (Source B)'s Phase 1b empirically tested a state-level
fallback and a proportional-allocation proxy for the spec's proposed IPF fix,
and neither meaningfully beat a plain null (r~=0.33-0.34 either way, barely
above a global-mean baseline), so an honest null with a disclosure flag was
chosen over a number that would read as more precise than it is.

No credentials required. `www.bls.gov` (the docs pages) is bot-gated, but the
actual file host, `data.bls.gov`, has no bot protection and no TLS quirk --
plain `requests` works, unlike Source D's `faf.ornl.gov` host.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

YEAR: int = 2025
SINGLEFILE_URL_TEMPLATE: str = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"
REQUEST_TIMEOUT_SECONDS: int = 60
CHUNK_ROWS: int = 500_000

# Reference quarter and slice, per docs/plans/ingestion_recon.md (Source B) Phase 1/1b decisions.
TARGET_QTR: str = "4"
PRIVATE_OWN_CODE: str = "5"
COUNTY_AGGLVL_CODE: str = "74"

# The 20 primary 2-digit NAICS sectors QCEW ships pre-computed Location
# Quotients for at the private/county slice (confirmed in Phase 0/1b).
NAICS2_CODES: tuple[str, ...] = (
    "11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51", "52",
    "53", "54", "55", "56", "61", "62", "71", "72", "81", "99",
)

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_b_qcew.parquet"
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


class SourceBError(Exception):
    """Base exception for all Source B ingestion failures."""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def download_singlefile_zip(url: str, dest_path: Path) -> None:
    """Stream-download the QCEW bulk quarterly singlefile zip to disk.

    Streamed rather than loaded into memory: the compressed file is ~287MB.

    Args:
        url: Direct URL to the `{year}_qtrly_singlefile.zip` bulk file.
        dest_path: Local path to write the downloaded zip to.

    Raises:
        SourceBError: If the download fails.
    """
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
    except requests.RequestException as exc:
        raise SourceBError(f"Failed to download QCEW singlefile zip: {exc}") from exc


def stream_filter_county_sectors(zip_path: Path) -> pd.DataFrame:
    """Stream-parse the bulk singlefile CSV, keeping only the target slice.

    Reads the ~2.2GB uncompressed CSV in chunks to avoid holding the full
    file in memory, filtering each chunk to the reference quarter, Private
    ownership, and county-level rows before concatenating.

    Args:
        zip_path: Path to the downloaded `{year}_qtrly_singlefile.zip`.

    Returns:
        Long-format DataFrame (one row per county x NAICS-2 sector) with all
        original QCEW columns, string-typed.

    Raises:
        SourceBError: If no rows match the target filter.
    """
    chunks: list[pd.DataFrame] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            except StopIteration:
                raise SourceBError("No CSV member found in QCEW singlefile zip.")
            with archive.open(csv_name) as f:
                for i, chunk in enumerate(pd.read_csv(f, dtype=str, chunksize=CHUNK_ROWS)):
                    mask = (
                        (chunk["qtr"] == TARGET_QTR)
                        & (chunk["own_code"] == PRIVATE_OWN_CODE)
                        & (chunk["agglvl_code"] == COUNTY_AGGLVL_CODE)
                    )
                    filtered = chunk.loc[mask]
                    if not filtered.empty:
                        chunks.append(filtered)
                    if i % 10 == 0:
                        logger.info("Processed chunk %d...", i)
    except zipfile.BadZipFile as exc:
        raise SourceBError(f"Downloaded QCEW zip is corrupt: {exc}") from exc

    if not chunks:
        raise SourceBError(
            f"No rows matched qtr={TARGET_QTR}, own_code={PRIVATE_OWN_CODE}, "
            f"agglvl_code={COUNTY_AGGLVL_CODE}."
        )
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def transform_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format county x sector rows into one row per county.

    Args:
        long_df: Raw filtered rows from stream_filter_county_sectors, with
            `area_fips`, `industry_code`, `disclosure_code`, and
            `lq_month3_emplvl` columns.

    Returns:
        Wide DataFrame indexed by `fips_code`, with `lq_emp_{naics2}`
        (nullable float) and `disclosure_{naics2}` (nullable boolean, True =
        BLS-suppressed) columns for each of the 20 NAICS-2 sectors.
    """
    long_df = long_df.copy()
    long_df["lq_month3_emplvl"] = pd.to_numeric(long_df["lq_month3_emplvl"], errors="coerce")
    long_df["suppressed"] = (long_df["disclosure_code"] == "N").astype("boolean")

    lq_wide = long_df.pivot(index="area_fips", columns="industry_code", values="lq_month3_emplvl")
    lq_wide = lq_wide.reindex(columns=NAICS2_CODES)
    lq_wide.columns = [f"lq_emp_{code}" for code in NAICS2_CODES]

    disclosure_wide = long_df.pivot(index="area_fips", columns="industry_code", values="suppressed")
    disclosure_wide = disclosure_wide.reindex(columns=NAICS2_CODES)
    disclosure_wide.columns = [f"disclosure_{code}" for code in NAICS2_CODES]

    # BLS reports a suppressed cell's lq_* value as 0 rather than blank, which
    # is indistinguishable from a genuine zero LQ without the disclosure flag
    # (confirmed in docs/plans/ingestion_recon.md (Source B)'s Phase 0) -- null it out explicitly here
    # rather than trust the raw value at face value.
    for code in NAICS2_CODES:
        lq_wide.loc[disclosure_wide[f"disclosure_{code}"].fillna(False), f"lq_emp_{code}"] = pd.NA

    result = lq_wide.join(disclosure_wide)
    result.index.name = "fips_code"
    return result.reset_index()


def join_county_names(wide: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join wide QCEW rows onto the project's county/FIPS crosswalk.

    Args:
        wide: Wide DataFrame from transform_wide, with a `fips_code` column.
        crosswalk: DataFrame with `county_name`, `fips_code` columns.

    Returns:
        DataFrame with `county_name` first, followed by all QCEW columns.
    """
    merged = crosswalk.merge(wide, on="fips_code", how="inner")

    unmatched = set(crosswalk["fips_code"]) - set(merged["fips_code"])
    if unmatched:
        logger.warning("%d crosswalk counties had no QCEW county-level match: %s", len(unmatched), sorted(unmatched))

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
    """Run the Source B ingestion pipeline over all US counties."""
    configure_logging()

    if not COUNTY_CROSSWALK_CACHE_PATH.exists():
        logger.error(
            "County crosswalk not found at %s; run ingest_source_a.py or "
            "ingest_source_c.py first to populate it.",
            COUNTY_CROSSWALK_CACHE_PATH,
        )
        sys.exit(1)
    crosswalk = pd.read_parquet(COUNTY_CROSSWALK_CACHE_PATH)

    url = SINGLEFILE_URL_TEMPLATE.format(year=YEAR)
    with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
        logger.info("Downloading QCEW %d bulk singlefile from %s...", YEAR, url)
        try:
            download_singlefile_zip(url, Path(tmp.name))
            logger.info(
                "Stream-filtering to qtr=%s, own_code=%s (Private), agglvl_code=%s (county)...",
                TARGET_QTR, PRIVATE_OWN_CODE, COUNTY_AGGLVL_CODE,
            )
            long_df = stream_filter_county_sectors(Path(tmp.name))
        except SourceBError as exc:
            logger.error("Aborting: %s", exc)
            sys.exit(1)

    logger.info("Filtered to %d county x sector rows.", len(long_df))

    wide = transform_wide(long_df)
    df = join_county_names(wide, crosswalk)

    export_to_parquet(df, OUTPUT_PARQUET_PATH)
    logger.info("Covered %d of %d crosswalk counties.", len(df), len(crosswalk))


if __name__ == "__main__":
    main()
