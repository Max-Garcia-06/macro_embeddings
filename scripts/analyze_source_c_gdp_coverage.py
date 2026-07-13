"""Quantify and characterize the missing-GDP-series gap in Source C data.

`ingest_source_c.py` treats a missing FRED GDP series as an expected,
per-county failure mode rather than an error (see `Winchester City, VA`,
which has a valid unemployment series but no `REALGDPALL51840` series). This
script turns that expectation into an actual measured rate and checks
whether the gap concentrates in a particular pattern (e.g. independent
cities, specific states) -- the numeric-data analog of how Source B's spec
anticipates BLS data suppression.

Output: `source_c_gdp_coverage.csv` (missing-GDP counties) and a short
console/log summary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from visualize_source_c import SOURCE_C_PARQUET_PATH, load_source_c

OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_c_gdp_coverage.csv"

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def extract_state(county_name: str) -> str:
    """Extract the state name from a "County Name, State Name" display string.

    Args:
        county_name: e.g. "Winchester City, Virginia".

    Returns:
        The state name, e.g. "Virginia".
    """
    return county_name.rpartition(", ")[2]


def summarize_gdp_coverage(df: pd.DataFrame) -> dict:
    """Compute overall and state-level missing-GDP-series statistics.

    Args:
        df: Source C DataFrame with `county_name` and `gdp_velocity`.

    Returns:
        Dict with `total_counties`, `missing_gdp_count`, `missing_gdp_rate`,
        `missing_gdp_is_independent_city_rate`, and `missing_by_state`
        (state -> missing count, states with zero missing omitted).
    """
    missing = df[df["gdp_velocity"].isna()].copy()
    missing["state"] = missing["county_name"].map(extract_state)
    missing["is_independent_city"] = missing["county_name"].str.contains(
        " City,", regex=False
    )

    missing_by_state = missing["state"].value_counts().to_dict()

    return {
        "total_counties": int(len(df)),
        "missing_gdp_count": int(len(missing)),
        "missing_gdp_rate": float(len(missing) / len(df)) if len(df) else 0.0,
        "missing_gdp_is_independent_city_rate": (
            float(missing["is_independent_city"].mean()) if len(missing) else 0.0
        ),
        "missing_by_state": missing_by_state,
    }


def main() -> None:
    """Run the GDP coverage-gap characterization."""
    configure_logging()

    df = load_source_c(SOURCE_C_PARQUET_PATH)
    missing = df[df["gdp_velocity"].isna()].copy()
    missing["state"] = missing["county_name"].map(extract_state)
    missing["is_independent_city"] = missing["county_name"].str.contains(
        " City,", regex=False
    )
    missing[["county_name", "fips_code", "state", "is_independent_city"]].to_csv(
        OUTPUT_CSV_PATH, index=False
    )
    logger.info("Wrote %d missing-GDP counties to %s", len(missing), OUTPUT_CSV_PATH)

    summary = summarize_gdp_coverage(df)
    logger.info(
        "GDP series missing for %d/%d counties (%.1f%%); %.1f%% of those are independent cities.",
        summary["missing_gdp_count"],
        summary["total_counties"],
        summary["missing_gdp_rate"] * 100,
        summary["missing_gdp_is_independent_city_rate"] * 100,
    )
    for state, count in sorted(summary["missing_by_state"].items(), key=lambda kv: -kv[1]):
        logger.info("  %s: %d missing", state, count)


if __name__ == "__main__":
    main()
