"""Per-pillar data vintage, stamped onto every shipped parquet.

The six pillars describe periods spanning three years -- FAF freight is 2022,
USDA typology 2023, QCEW and FRED unemployment 2025 -- and until now they were
joined into one county snapshot with nothing in the data saying so. That is
acceptable for a static feature layer and a leakage vector the moment a
downstream model backtests against a pre-2025 outcome, because Sources B and C
would then carry information from after the outcome was observed. A consuming
team cannot detect that from the parquets alone, which is what this module
fixes.

`as_of_date` is the **end of the period the data describes**, not the date it was
downloaded. That is the date leakage checks need: a 2022 freight total is safe
against a 2023 outcome regardless of when it was fetched. Sources A and F
describe no fixed period, so they carry the tightest available upper bound on
what they can know -- A's scrape date and F's publication edition. Both are
flagged in `reference_period` rather than silently mixed in.

Every ingest script calls `stamp_vintage` before writing. `python -m
pillar_vintage` back-fills the column onto parquets already on disk without
refetching, and writes `outputs/pillar_vintages.csv` for the handoff.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "pillar_vintages.csv"

VINTAGE_COLUMN: str = "as_of_date"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PillarVintage:
    """Reference period of one pillar's shipped data.

    Attributes:
        pillar: Pillar letter, "A".."F".
        parquet_name: Filename under `data/`.
        as_of_date: ISO date ending the period the data describes.
        reference_period: Human-readable period, for the handoff table.
        cadence: How often the upstream source republishes.
        note: Anything a consumer needs beyond the date itself.
    """

    pillar: str
    parquet_name: str
    as_of_date: str
    reference_period: str
    cadence: str
    note: str


# Source A carries the scrape date rather than a reference period: a Wikipedia
# article describes no fixed window, and the corpus was last refetched
# 2026-08-03 (source-a-findings.md 14). Treat it as "current as of" rather than
# as a period end -- it is the one pillar whose content can change without any
# upstream release.
PILLAR_VINTAGES: tuple[PillarVintage, ...] = (
    PillarVintage(
        pillar="A",
        parquet_name="source_a_text_features.parquet",
        as_of_date="2026-08-03",
        reference_period="scrape date (no reference period)",
        cadence="continuous",
        note="Wikipedia lead and section text; refetched 2026-08-03.",
    ),
    PillarVintage(
        pillar="B",
        parquet_name="source_b_qcew.parquet",
        as_of_date="2025-12-31",
        reference_period="2025 Q4",
        cadence="quarterly",
        note="BLS QCEW private-ownership county employment; ~35% of LQ cells suppressed.",
    ),
    PillarVintage(
        pillar="C",
        parquet_name="source_c_fred.parquet",
        as_of_date="2025-12-31",
        reference_period="unemployment through 2025, real GDP through 2024",
        cadence="monthly (unemployment) / annual (GDP)",
        note="Velocities are 3-year slopes; per-row years stay in "
        "unemployment_latest_year and gdp_latest_year.",
    ),
    PillarVintage(
        pillar="D",
        parquet_name="source_d_faf.parquet",
        as_of_date="2022-12-31",
        reference_period="2022",
        cadence="every ~5 years",
        note="BTS FAF5 county freight tonnage; oldest pillar in the matrix.",
    ),
    PillarVintage(
        pillar="E",
        parquet_name="source_e_irs_soi.parquet",
        as_of_date="2022-12-31",
        reference_period="tax year 2022 (cross-year columns span TY2018-TY2022)",
        cadence="annual",
        note="IRS SOI county file; no suppression flag published upstream. The "
        "cross-year columns are built only from years at or before this date, so "
        "the stamp stays a valid upper bound. Per-year rows are in "
        "source_e_irs_soi_panel.parquet, which carries tax_year instead.",
    ),
    PillarVintage(
        pillar="F",
        parquet_name="source_f_usda_typology.parquet",
        as_of_date="2025-12-31",
        reference_period="2025 edition (2023 OMB metro delineation)",
        cadence="every ~10 years",
        note="USDA ERS county typology codes, 2025 edition. Date is the "
        "publication edition, not a period end: the codes are built from "
        "several upstream series with different windows, all predating "
        "publication, so this is an upper bound on what they can know.",
    ),
)

VINTAGE_BY_PILLAR: dict[str, PillarVintage] = {v.pillar: v for v in PILLAR_VINTAGES}


def stamp_vintage(df: pd.DataFrame, pillar: str) -> pd.DataFrame:
    """Return `df` with the pillar's `as_of_date` attached as a column.

    Args:
        df: Pillar frame about to be written to Parquet.
        pillar: Pillar letter, "A".."F".

    Returns:
        A copy carrying `as_of_date`. The input is not mutated.

    Raises:
        KeyError: If `pillar` is not one of "A".."F".
    """
    vintage = VINTAGE_BY_PILLAR[pillar]
    return df.assign(**{VINTAGE_COLUMN: vintage.as_of_date})


def main() -> None:
    """Back-fill `as_of_date` onto parquets already on disk and write the table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    for vintage in PILLAR_VINTAGES:
        path = DATA_DIR / vintage.parquet_name
        if not path.exists():
            logger.warning("Skipping Source %s: %s not found", vintage.pillar, path)
            continue
        df = pd.read_parquet(path)
        if VINTAGE_COLUMN in df.columns and df[VINTAGE_COLUMN].eq(vintage.as_of_date).all():
            logger.info("Source %s already stamped %s", vintage.pillar, vintage.as_of_date)
            continue
        stamp_vintage(df, vintage.pillar).to_parquet(path, engine="pyarrow", index=False)
        logger.info("Stamped Source %s with %s", vintage.pillar, vintage.as_of_date)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([vars(v) for v in PILLAR_VINTAGES]).to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %s", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
