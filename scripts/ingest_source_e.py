"""Source E ingestion pipeline for E_macro: IRS Statistics of Income Panel
Data (Capital Composition).

Downloads the IRS SOI Division's pre-aggregated county files
(`{yy}incyallnoagi.csv` -- `AGI_STUB` fixed at 0, i.e. IRS's own county
totals, not the 8-AGI-bracket breakdown) for `TAX_YEARS` and derives each
county's capital-vs-wage income position, per
`docs/plans/ingestion_recon.md (Source E)`'s Phase 0 findings.

No credentials required. `www.irs.gov` has no bot protection on either the
landing pages or the `/pub/irs-soi/*.csv` file host -- plain `requests`
works throughout, unlike Source B/D's landing-page/data-host split.

Target columns are referenced via `SOI_COLUMN_MAP` (conceptual name -> SOI
variable code) rather than positional indexing, per the pre-scoping spec's
proposed mitigation for upstream schema mutation -- a future year renaming
or dropping one of these fields fails loudly (KeyError) instead of silently
misreading a shifted column.

Three properties of this source drive the shipped schema, all measured in
`source-e-findings.md` §9-§13:

1. `capital_to_wage_ratio` is a product of three separable quantities -- how
   many filers report investment income, how much each reports, and how much
   wage income sits underneath (R^2 = 0.975 on its log, near-unit
   elasticities). The components ship alongside it rather than collapsed in.
2. Its *level* is set by the market year: the unweighted county mean runs
   0.095 (TY2020), 0.156 (TY2021), 0.108 (TY2022). A panel across `TAX_YEARS`
   therefore ships too, normalized against each year's national aggregate.
3. The file carries no suppression flag, unlike BLS QCEW's `disclosure_code`
   (`docs/plans/ingestion_recon.md (Source E)` Risk 3). Its footprint is small
   -- 3 counties -- but the counts columns (`N00200`/`N00650`/`N01000`) now
   ship so a consumer can see a thin sample instead of inferring it.
"""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from pillar_vintage import stamp_vintage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# TY2015 is the earliest county file with the same variable codes; TY2023 was
# not published as of this writing (HTTP 404).
TAX_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)
LATEST_TAX_YEAR: int = TAX_YEARS[-1]
SOURCE_CSV_URL_TEMPLATE: str = "https://www.irs.gov/pub/irs-soi/{yy}incyallnoagi.csv"
REQUEST_TIMEOUT_SECONDS: int = 60
CSV_ENCODING: str = "latin-1"

STATE_TOTAL_COUNTYFIPS: str = "000"
STATE_FIPS_WIDTH: int = 2
COUNTY_FIPS_WIDTH: int = 3

# Conceptual name -> raw SOI variable code, per docs/plans/ingestion_recon.md (Source E)'s Risk 2
# mitigation (insulates the pipeline from future line-item renumbering). The
# `n_returns_*` codes are counts of returns reporting the matching amount.
SOI_COLUMN_MAP: dict[str, str] = {
    "num_returns": "N1",
    "agi_thousands": "A00100",
    "wages_salaries_thousands": "A00200",
    "n_returns_wages": "N00200",
    "qualified_dividends_thousands": "A00650",
    "n_returns_qualified_dividends": "N00650",
    "net_cap_gain_thousands": "A01000",
    "n_returns_net_cap_gain": "N01000",
}

# Below this return count a county holds a negligible share of the national
# income base -- the 325 flagged counties are 10.3% of rows but 0.14% of
# national investment income. It is a *materiality* flag, not a noise flag:
# these counties are no less stable year over year than large ones
# (source-e-findings.md §11).
LOW_RETURN_THRESHOLD: int = 2_200

# Below this many filers actually reporting a net capital gain, the county's
# numerator rests on a sample too thin to read as a composition signal, and a
# suppressed cell is indistinguishable from a true zero.
THIN_CLAIMER_THRESHOLD: int = 100

# Above this percentile of gain-per-claiming-return, the ratio is driven by a
# few unusually large realizations (typically farm/ranch land sales) rather
# than a broad investment base -- the mechanism `low_return_flag` was
# originally introduced to catch but does not actually isolate.
CONCENTRATED_GAIN_PERCENTILE: float = 0.95

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_e_irs_soi.parquet"
PANEL_PARQUET_PATH: Path = DATA_DIR / "source_e_irs_soi_panel.parquet"
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


def download_county_csv(tax_year: int) -> pd.DataFrame:
    """Download one tax year's IRS SOI pre-aggregated county CSV.

    Args:
        tax_year: Four-digit tax year, e.g. 2022.

    Returns:
        Raw DataFrame, read with pandas' default type inference except for
        the two FIPS columns, which stay string-typed.

    Raises:
        SourceEError: If the download fails or a required column is missing.
    """
    url = SOURCE_CSV_URL_TEMPLATE.format(yy=str(tax_year)[-2:])
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceEError(f"Failed to download IRS SOI county file for TY{tax_year}: {exc}") from exc

    df = pd.read_csv(
        StringIO(response.content.decode(CSV_ENCODING)),
        dtype={"STATEFIPS": str, "COUNTYFIPS": str},
        low_memory=False,
    )

    missing = [code for code in SOI_COLUMN_MAP.values() if code not in df.columns]
    if missing:
        raise SourceEError(
            f"IRS SOI TY{tax_year} file is missing expected columns {missing} -- "
            "upstream schema likely changed, see docs/plans/ingestion_recon.md (Source E) Risk 2."
        )
    return df


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def transform(raw: pd.DataFrame, tax_year: int) -> pd.DataFrame:
    """Drop state-total rows and derive one tax year's county measures.

    The ratio's three separable drivers ship as their own columns: the share
    of filers reporting each income type, the average size of a reported
    capital gain, and wage income per return.

    Args:
        raw: DataFrame from download_county_csv.
        tax_year: Four-digit tax year the rows describe.

    Returns:
        One row per county, carrying `fips_code`, `tax_year`, the mapped SOI
        columns, and the derived component columns.
    """
    # TY2019 ships both FIPS columns unpadded ("1"/"1" where every other year
    # writes "01"/"001"), so padding has to happen before the state-total rows
    # are dropped -- otherwise those rows survive the filter and the
    # concatenated key lands on the wrong county.
    padded = raw.assign(
        STATEFIPS=raw["STATEFIPS"].str.strip().str.zfill(STATE_FIPS_WIDTH),
        COUNTYFIPS=raw["COUNTYFIPS"].str.strip().str.zfill(COUNTY_FIPS_WIDTH),
    )
    counties = padded.loc[padded["COUNTYFIPS"] != STATE_TOTAL_COUNTYFIPS].copy()
    counties["fips_code"] = counties["STATEFIPS"] + counties["COUNTYFIPS"]
    counties["tax_year"] = tax_year

    duplicates = counties["fips_code"].duplicated().sum()
    if duplicates:
        raise SourceEError(
            f"IRS SOI TY{tax_year} produced {duplicates} duplicate county FIPS codes "
            "after padding -- upstream key format likely changed again."
        )

    for concept, code in SOI_COLUMN_MAP.items():
        counties[concept] = pd.to_numeric(counties[code], errors="coerce")

    investment_income = (
        counties["net_cap_gain_thousands"] + counties["qualified_dividends_thousands"]
    )
    counties["capital_to_wage_ratio"] = investment_income / counties["wages_salaries_thousands"]
    counties["capgain_participation_rate"] = (
        counties["n_returns_net_cap_gain"] / counties["num_returns"]
    )
    counties["dividend_participation_rate"] = (
        counties["n_returns_qualified_dividends"] / counties["num_returns"]
    )
    # Null, not zero, where nobody reported a gain: there is no average of an
    # empty set, and a zero here would read as "gains were small".
    counties["gain_per_claimer_thousands"] = counties["net_cap_gain_thousands"] / counties[
        "n_returns_net_cap_gain"
    ].replace(0, pd.NA)
    counties["wage_per_return_thousands"] = (
        counties["wages_salaries_thousands"] / counties["num_returns"]
    )

    # National aggregate for this year: total investment income over total
    # wages, not the mean of the county ratios. Dividing by it removes the
    # market-cycle level shift that otherwise moves every county at once.
    national_ratio = investment_income.sum() / counties["wages_salaries_thousands"].sum()
    counties["national_capital_to_wage_ratio"] = national_ratio
    counties["capital_to_wage_ratio_normalized"] = counties["capital_to_wage_ratio"] / national_ratio

    derived = [
        "capital_to_wage_ratio",
        "capital_to_wage_ratio_normalized",
        "national_capital_to_wage_ratio",
        "capgain_participation_rate",
        "dividend_participation_rate",
        "gain_per_claimer_thousands",
        "wage_per_return_thousands",
    ]
    return counties[["fips_code", "tax_year", *SOI_COLUMN_MAP.keys(), *derived]]


def build_panel() -> pd.DataFrame:
    """Download and transform every tax year in `TAX_YEARS`.

    Returns:
        Long-format DataFrame, one row per county per tax year.

    Raises:
        SourceEError: Propagated from download_county_csv.
    """
    frames: list[pd.DataFrame] = []
    for tax_year in TAX_YEARS:
        logger.info("Downloading IRS SOI TY%d county file...", tax_year)
        year_frame = transform(download_county_csv(tax_year), tax_year)
        logger.info(
            "TY%d: %d counties, national capital-to-wage ratio %.4f",
            tax_year,
            len(year_frame),
            year_frame["national_capital_to_wage_ratio"].iloc[0],
        )
        frames.append(year_frame)
    return pd.concat(frames, ignore_index=True)


def summarize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the multi-year panel to one row per county.

    Averaging the *normalized* ratio rather than the raw one is the point: a
    raw multi-year mean would still carry whichever market years happened to
    fall in the window.

    Args:
        panel: Long-format frame from build_panel.

    Returns:
        DataFrame keyed on `fips_code` with the cross-year summary columns.
    """
    grouped = panel.groupby("fips_code")["capital_to_wage_ratio_normalized"]
    return pd.DataFrame(
        {
            "capital_to_wage_ratio_normalized_mean": grouped.mean(),
            "capital_to_wage_ratio_normalized_std": grouped.std(),
            "n_tax_years_observed": grouped.size(),
        }
    ).reset_index()


def add_reliability_flags(latest: pd.DataFrame) -> pd.DataFrame:
    """Attach the three flags describing how far a county's ratio can be trusted.

    Args:
        latest: Latest-tax-year county frame from transform.

    Returns:
        A copy carrying `low_return_flag`, `thin_claimer_flag`, and
        `concentrated_gain_flag`. The input is not mutated.
    """
    concentrated_gain_cutoff = latest["gain_per_claimer_thousands"].quantile(
        CONCENTRATED_GAIN_PERCENTILE
    )
    logger.info(
        "Concentrated-gain cutoff (p%d of gain per claiming return): %.1f thousand USD",
        int(CONCENTRATED_GAIN_PERCENTILE * 100),
        concentrated_gain_cutoff,
    )
    return latest.assign(
        low_return_flag=latest["num_returns"] < LOW_RETURN_THRESHOLD,
        thin_claimer_flag=latest["n_returns_net_cap_gain"] < THIN_CLAIMER_THRESHOLD,
        concentrated_gain_flag=(
            latest["gain_per_claimer_thousands"] > concentrated_gain_cutoff
        ).fillna(False),
    )


def join_county_names(transformed: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join transformed IRS SOI rows onto the project's county/FIPS crosswalk.

    Args:
        transformed: DataFrame with a `fips_code` column.
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

    try:
        panel = build_panel()
    except SourceEError as exc:
        logger.error("Aborting: %s", exc)
        sys.exit(1)

    latest = panel.loc[panel["tax_year"] == LATEST_TAX_YEAR].drop(columns=["tax_year"])
    latest = add_reliability_flags(latest).merge(summarize_panel(panel), on="fips_code", how="left")

    df = join_county_names(latest, crosswalk)
    export_to_parquet(df, OUTPUT_PARQUET_PATH)

    # The panel is not vintage-stamped: every row already states the period it
    # describes in `tax_year`, and a single `as_of_date` across five tax years
    # would be less informative than the column already there.
    joined_panel = join_county_names(panel, crosswalk)
    joined_panel.to_parquet(PANEL_PARQUET_PATH, engine="pyarrow", index=False)
    logger.info("Wrote %d panel rows to %s", len(joined_panel), PANEL_PARQUET_PATH)

    logger.info(
        "Covered %d of %d crosswalk counties across TY%d-TY%d.",
        len(df), len(crosswalk), TAX_YEARS[0], LATEST_TAX_YEAR,
    )


if __name__ == "__main__":
    main()
