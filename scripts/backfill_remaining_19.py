"""Backfill the final 19 counties/county-equivalents missing from source_a_embeddings.parquet.

All 19 failed ingestion because their Census Gazetteer NAME field diverges
from their Wikipedia article title (8 Alaska boroughs/census areas, 5 NYC
boroughs, and 6 one-off spelling/diacritic mismatches); see
`analysis-output/source-a/source-a-findings.md` section 7. `INDEPENDENT_CITY_ARTICLE_LOOKUP`
in `ingest_source_a.py` now maps all 19 to their correct Wikipedia article
titles. This script re-runs ingestion for just those 19 counties and merges
the results into the existing parquet, leaving all other rows untouched.

Historical: this was a one-shot repair run, already applied. It is retained
as a record of the fix, not as part of the pipeline -- the lookup table it
depended on now lives in `ingest_source_a.py`, and
`data/source_a_embeddings.parquet` is no longer regenerated (see README).
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from ingest_source_a import (
    BgeM3EmbeddingGenerator,
    OUTPUT_PARQUET_PATH,
    WikimediaEnterpriseClient,
    WikimediaAuthError,
    build_dataframe,
    configure_logging,
    export_to_parquet,
    run_pipeline,
)

logger = logging.getLogger(__name__)

MISSING_COUNTIES: list[str] = [
    "Anchorage Municipality, Alaska",
    "Hoonah-Angoon Census Area, Alaska",
    "Juneau City and Borough, Alaska",
    "Prince of Wales-Hyder Census Area, Alaska",
    "Sitka City and Borough, Alaska",
    "Skagway Municipality, Alaska",
    "Wrangell City and Borough, Alaska",
    "Yakutat City and Borough, Alaska",
    "Hawaii County, Hawaii",
    "De Witt County, Illinois",
    "Larue County, Kentucky",
    "De Soto Parish, Louisiana",
    "Nantucket County, Massachusetts",
    "Bronx County, New York",
    "Kings County, New York",
    "New York County, New York",
    "Queens County, New York",
    "Richmond County, New York",
    "Le Flore County, Oklahoma",
]


def main() -> None:
    """Ingest the 19 remaining counties and merge into the parquet."""
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
        results, summary = run_pipeline(MISSING_COUNTIES, client, embedder)
    except WikimediaAuthError as exc:
        logger.error("Authentication rejected mid-run; aborting: %s", exc)
        sys.exit(1)

    logger.info("Backfill succeeded: %d, failed: %d", len(summary.succeeded), len(summary.failed))
    for county, reason in summary.failed.items():
        logger.warning("  %s -> %s", county, reason)

    new_rows = build_dataframe(results)
    existing = pd.read_parquet(OUTPUT_PARQUET_PATH)
    merged = (
        pd.concat([existing, new_rows], ignore_index=True)
        .drop_duplicates(subset="county_name", keep="last")
        .reset_index(drop=True)
    )
    export_to_parquet(merged, OUTPUT_PARQUET_PATH)
    logger.info(
        "Parquet grew from %d to %d rows (+%d)",
        len(existing),
        len(merged),
        len(merged) - len(existing),
    )


if __name__ == "__main__":
    main()
