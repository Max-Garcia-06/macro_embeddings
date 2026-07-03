"""Backfill the 37 Virginia independent cities missing from source_a_embeddings.parquet.

These all failed ingestion with 404s (wrong article title) or a 429 rate-limit
(Richmond City, Roanoke City) in the original run; see
`analysis-output/source-a-findings.md` section 7.
`INDEPENDENT_CITY_ARTICLE_LOOKUP` in `ingest_source_a.py` now maps all 37 to
their correct Wikipedia article titles. This script re-runs ingestion for
just those 37 counties and merges the results into the existing parquet,
leaving all other rows untouched.
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

MISSING_VIRGINIA_CITIES: list[str] = [
    "Alexandria City, Virginia",
    "Bristol City, Virginia",
    "Buena Vista City, Virginia",
    "Charlottesville City, Virginia",
    "Chesapeake City, Virginia",
    "Colonial Heights City, Virginia",
    "Covington City, Virginia",
    "Danville City, Virginia",
    "Emporia City, Virginia",
    "Fairfax City, Virginia",
    "Falls Church City, Virginia",
    "Fredericksburg City, Virginia",
    "Galax City, Virginia",
    "Hampton City, Virginia",
    "Harrisonburg City, Virginia",
    "Hopewell City, Virginia",
    "Lexington City, Virginia",
    "Lynchburg City, Virginia",
    "Manassas City, Virginia",
    "Manassas Park City, Virginia",
    "Martinsville City, Virginia",
    "Newport News City, Virginia",
    "Norfolk City, Virginia",
    "Norton City, Virginia",
    "Petersburg City, Virginia",
    "Poquoson City, Virginia",
    "Portsmouth City, Virginia",
    "Radford City, Virginia",
    "Richmond City, Virginia",
    "Roanoke City, Virginia",
    "Salem City, Virginia",
    "Staunton City, Virginia",
    "Suffolk City, Virginia",
    "Virginia Beach City, Virginia",
    "Waynesboro City, Virginia",
    "Williamsburg City, Virginia",
    "Winchester City, Virginia",
]


def main() -> None:
    """Ingest the missing Virginia independent cities and merge into the parquet."""
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
        results, summary = run_pipeline(MISSING_VIRGINIA_CITIES, client, embedder)
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
