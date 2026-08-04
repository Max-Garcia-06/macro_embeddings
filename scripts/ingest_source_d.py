"""Source D ingestion pipeline for E_macro: BTS FAF5 Experimental County Commodity Flows.

Downloads the BTS Freight Analysis Framework Version 5 (FAF5) Experimental
County-Level Estimates -- per-state zip files hosted at faf.ornl.gov, each
containing origin-destination tables at county-to-county, county-to-FAF-zone,
and FAF-zone-to-county granularity (a fourth, FAF-zone-to-FAF-zone table is
also present but never touches a US county directly, so it isn't needed here).
For each state's own ("home") counties, computes domestic-only (trade_type=1)
tonnage totals by commodity supergroup (sctgG5) plus a Herfindahl-Hirschman
trade-partner concentration index, pooling county-level and FAF-zone-level
partner rows into a single distribution per county. See docs/plans/ingestion_recon.md (Source D) for
the empirical comparison behind this feature set: scalar totals plus pooled
HHI concentration outperformed top-K partner columns and a distance-weighted
"reach" metric, both dropped after testing on Rhode Island and New Jersey
samples (a real hub county's long-distance reach is structurally excluded
from the county-to-county table alone, which made the distance metric point
backward in both tests).

A county only gets a result from its own home-state zip, not from appearing
as an "adjacent state" county in a neighbor's zip -- only the home zip
guarantees that county's complete adjacent-neighbor set.

No credentials required. The BTS landing page (bts.gov) is behind an Akamai
bot-detection gate that blocks plain HTTP clients, but the actual zip files
are hosted on faf.ornl.gov (Oak Ridge National Laboratory), which has no such
gate -- however, that plain IIS file server consistently resets the
connection during the TLS handshake against Python's ssl/urllib3 stack
specifically (reproduced with default settings and with TLS 1.2 forced),
while the system `curl` binary connects cleanly every time with zero special
headers or session state. Downloads shell out to `curl` for this reason; it
is an ordinary HTTP-client compatibility quirk against a public,
unauthenticated file server, not a security control being routed around.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pillar_vintage import stamp_vintage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FAF_COUNTY_BASE_URL: str = "https://faf.ornl.gov/faf5/Data/County/"

# Some per-state zips run to several hundred MB (Illinois alone is ~674MB);
# --max-time bounds curl's *entire* invocation including retries, so it must
# be generous enough to cover a large file at modest throughput plus a few
# retries on the transient connection resets this host exhibits.
CURL_MAX_TIME_SECONDS: int = 1800
CURL_RETRIES: int = 5
CURL_RETRY_DELAY_SECONDS: int = 2

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PARQUET_PATH: Path = DATA_DIR / "source_d_faf.parquet"
COUNTY_CROSSWALK_CACHE_PATH: Path = DATA_DIR / "county_crosswalk.parquet"

logger = logging.getLogger(__name__)

# Domestic-only flows (trade_type == 1): import/export legs (2/3) involve a
# foreign region rather than a second US county and are out of scope for a
# county-to-county trade gravity feature.
DOMESTIC_TRADE_TYPE: int = 1

# 5 SCTG commodity supergroups used by the experimental county product,
# aggregated from FAF5's base 2-digit SCTG codes.
SCTG_GROUPS: tuple[str, ...] = ("sctg0109", "sctg1014", "sctg1519", "sctg2033", "sctg3499")

# 2-digit state FIPS -> exact zip filename fragment (URL-encoded) on
# faf.ornl.gov/faf5/Data/County/. Scraped once from the bts.gov/faf/county
# landing page (loaded via a real browser session to clear its Akamai bot
# gate); the zip files themselves are hosted on an ungated server. 50 states
# plus the District of Columbia.
STATE_ZIP_FILES: dict[str, str] = {
    "01": "01%20-%20Alabama.zip",
    "02": "02%20-%20Alaska.zip",
    "04": "04%20-%20Arizona.zip",
    "05": "05%20-%20Arkansas.zip",
    "06": "06%20-%20California.zip",
    "08": "08%20-%20Colorado.zip",
    "09": "09%20-%20Connecticut.zip",
    "10": "10%20-%20Delaware.zip",
    "11": "11%20-%20Washington%20DC.zip",
    "12": "12%20-%20Florida.zip",
    "13": "13%20-%20Georgia.zip",
    "15": "15%20-%20Hawaii.zip",
    "16": "16%20-%20Idaho.zip",
    "17": "17%20-%20Illinois.zip",
    "18": "18%20-%20Indiana.zip",
    "19": "19%20-%20Iowa.zip",
    "20": "20%20-%20Kansas.zip",
    "21": "21%20-%20Kentucky.zip",
    "22": "22%20-%20Louisiana.zip",
    "23": "23%20-%20Maine.zip",
    "24": "24%20-%20Maryland.zip",
    "25": "25%20-%20Massachusetts.zip",
    "26": "26%20-%20Michigan.zip",
    "27": "27%20-%20Minnesota.zip",
    "28": "28%20-%20Mississippi.zip",
    "29": "29%20-%20Missouri.zip",
    "30": "30%20-%20Montana.zip",
    "31": "31%20-%20Nebraska.zip",
    "32": "32%20-%20Nevada.zip",
    "33": "33%20-%20New%20Hampshire.zip",
    "34": "34%20-%20New%20Jersey.zip",
    "35": "35%20-%20New%20Mexico.zip",
    "36": "36%20-%20New%20York.zip",
    "37": "37%20-%20North%20Carolina.zip",
    "38": "38%20-%20North%20Dakota.zip",
    "39": "39%20-%20Ohio.zip",
    "40": "40%20-%20Oklahoma.zip",
    "41": "41%20-%20Oregon.zip",
    "42": "42%20-%20Pennsylvania.zip",
    "44": "44%20-%20Rhode%20Island.zip",
    "45": "45%20-%20South%20Carolina.zip",
    "46": "46%20-%20South%20Dakota.zip",
    "47": "47%20-%20Tennessee.zip",
    "48": "48%20-%20Texas.zip",
    "49": "49%20-%20Utah.zip",
    "50": "50%20-%20Vermont.zip",
    "51": "51%20-%20Virginia.zip",
    "53": "53%20-%20Washington.zip",
    "54": "54%20-%20West%20Virginia.zip",
    "55": "55%20-%20Wisconsin.zip",
    "56": "56%20-%20Wyoming.zip",
}


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class SourceDError(Exception):
    """Base exception for all Source D ingestion failures."""


class StateDownloadError(SourceDError):
    """Raised when a state's FAF5 zip file cannot be downloaded or parsed."""


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


class FafCountyClient:
    """Thin transport-layer client for the FAF5 experimental county-level zips.

    Downloads via the system `curl` binary rather than `requests` -- see the
    module docstring for why. `curl`'s own `--retry` handles the transient
    connection resets this host exhibits.
    """

    def download_state_tables(self, zip_filename: str) -> dict[str, pd.DataFrame]:
        """Download and parse a state's FAF5 origin-destination tables.

        Args:
            zip_filename: URL-encoded zip filename fragment from STATE_ZIP_FILES.

        Returns:
            Dict with keys "county_to_county", "county_to_faf", "faf_to_county",
            each the raw (all trade types) DataFrame read from the zip's CSV.

        Raises:
            StateDownloadError: If the download or zip parsing fails.
        """
        url = FAF_COUNTY_BASE_URL + zip_filename
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            try:
                subprocess.run(
                    [
                        "curl",
                        "-sf",
                        "--max-time",
                        str(CURL_MAX_TIME_SECONDS),
                        "--retry",
                        str(CURL_RETRIES),
                        "--retry-delay",
                        str(CURL_RETRY_DELAY_SECONDS),
                        "-o",
                        tmp.name,
                        url,
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else str(exc)
                raise StateDownloadError(f"Failed to download '{zip_filename}': {stderr}") from exc

            try:
                with zipfile.ZipFile(tmp.name) as archive:
                    return {
                        "county_to_county": _read_csv_member(archive, "1.County-to-County.csv"),
                        "county_to_faf": _read_csv_member(archive, "2.County-to-FAF.csv"),
                        "faf_to_county": _read_csv_member(archive, "2.FAF-to-County.csv"),
                    }
            except (zipfile.BadZipFile, KeyError) as exc:
                raise StateDownloadError(f"Failed to parse '{zip_filename}': {exc}") from exc


def _read_csv_member(archive: zipfile.ZipFile, member_name: str) -> pd.DataFrame:
    """Read one CSV member of a FAF5 zip archive as a string-typed DataFrame.

    Geography/commodity columns are read as strings to preserve leading zeros
    in FIPS and FAF zone codes; only `tons_2022` and `trade_type` are numeric.
    """
    with archive.open(member_name) as f:
        df = pd.read_csv(f, dtype=str)
    df["tons_2022"] = pd.to_numeric(df["tons_2022"])
    df["trade_type"] = df["trade_type"].astype(int)
    return df


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------


def _domestic_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to domestic-only flows (trade_type == 1)."""
    return df[df["trade_type"] == DOMESTIC_TRADE_TYPE]


def _sctg_breakdown(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Sum tons by sctgG5 group, returning a full-width dict (0.0 for absent groups)."""
    totals = df.groupby("sctgG5")["tons_2022"].sum()
    return {f"{prefix}_{group}": float(totals.get(group, 0.0)) for group in SCTG_GROUPS}


def _hhi(partner_tons: pd.Series) -> float | None:
    """Compute the Herfindahl-Hirschman concentration index over a partner-tons distribution.

    Args:
        partner_tons: Tons summed per distinct partner (county FIPS, or a FAF
            zone code prefixed "zone:" to keep the two ID spaces distinct).

    Returns:
        HHI in (0, 1] (1 = all flow to a single partner), or None if there is
        no flow at all.
    """
    total = partner_tons.sum()
    if total <= 0:
        return None
    shares = partner_tons / total
    return float((shares**2).sum())


@dataclass
class CountyTradeFlowResult:
    """Holds Source D's computed feature set for a single county."""

    fips_code: str
    total_outbound_tons: float
    total_inbound_tons: float
    out_partner_hhi: float | None
    in_partner_hhi: float | None
    out_sctg0109: float
    out_sctg1014: float
    out_sctg1519: float
    out_sctg2033: float
    out_sctg3499: float
    in_sctg0109: float
    in_sctg1014: float
    in_sctg1519: float
    in_sctg2033: float
    in_sctg3499: float


def compute_county_flows(
    fips_code: str,
    domestic_county_to_county: pd.DataFrame,
    domestic_county_to_faf: pd.DataFrame,
    domestic_faf_to_county: pd.DataFrame,
) -> CountyTradeFlowResult:
    """Compute Source D's feature set for a single county.

    Pools county-level partners (from county_to_county) and FAF-zone-level
    partners (from county_to_faf / faf_to_county) into one partner-tons
    distribution per direction before computing concentration, since a
    county's true trade-partner spread includes both nearby counties and the
    zone-collapsed long tail -- restricting the HHI calc to county-level
    partners alone (as tested in docs/plans/ingestion_recon.md (Source D)'s Phase 1b spike) understates
    concentration for any county with substantial long-distance flow.

    Args:
        fips_code: County FIPS (or CT planning-region) code, home to the state
            these tables were downloaded for.
        domestic_county_to_county: This state's county-to-county table,
            already filtered to trade_type == 1.
        domestic_county_to_faf: This state's county-to-FAF-zone table,
            already filtered to trade_type == 1.
        domestic_faf_to_county: This state's FAF-zone-to-county table,
            already filtered to trade_type == 1.

    Returns:
        CountyTradeFlowResult with tonnage totals, commodity breakdowns, and
        pooled-partner HHI concentration for both directions.
    """
    out_county = domestic_county_to_county[domestic_county_to_county["dms_orig_cnty"] == fips_code]
    out_zone = domestic_county_to_faf[domestic_county_to_faf["dms_orig_cnty"] == fips_code]
    in_county = domestic_county_to_county[domestic_county_to_county["dms_dest_cnty"] == fips_code]
    in_zone = domestic_faf_to_county[domestic_faf_to_county["dms_dest_cnty"] == fips_code]

    total_out = float(out_county["tons_2022"].sum() + out_zone["tons_2022"].sum())
    total_in = float(in_county["tons_2022"].sum() + in_zone["tons_2022"].sum())

    out_sctg = _sctg_breakdown(pd.concat([out_county, out_zone]), "out")
    in_sctg = _sctg_breakdown(pd.concat([in_county, in_zone]), "in")

    out_zone_partners = out_zone.groupby("dms_dest")["tons_2022"].sum()
    out_zone_partners.index = out_zone_partners.index.map(lambda z: f"zone:{z}")
    out_partners = pd.concat([out_county.groupby("dms_dest_cnty")["tons_2022"].sum(), out_zone_partners])

    in_zone_partners = in_zone.groupby("dms_orig")["tons_2022"].sum()
    in_zone_partners.index = in_zone_partners.index.map(lambda z: f"zone:{z}")
    in_partners = pd.concat([in_county.groupby("dms_orig_cnty")["tons_2022"].sum(), in_zone_partners])

    return CountyTradeFlowResult(
        fips_code=fips_code,
        total_outbound_tons=total_out,
        total_inbound_tons=total_in,
        out_partner_hhi=_hhi(out_partners),
        in_partner_hhi=_hhi(in_partners),
        **out_sctg,
        **in_sctg,
    )


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------


@dataclass
class IngestionSummary:
    """Tracks per-state success/failure outcomes across the full run."""

    succeeded: list[str]
    failed: dict[str, str]


def process_state(
    zip_filename: str,
    home_fips_codes: list[str],
    client: FafCountyClient,
) -> list[CountyTradeFlowResult]:
    """Download one state's FAF5 zip and compute flow features for its home counties.

    Args:
        zip_filename: URL-encoded zip filename fragment for this state.
        home_fips_codes: Crosswalk FIPS codes whose 2-digit prefix matches
            this state -- only these get a result, since a county's complete
            adjacent-neighbor view only exists in its own home-state zip.
        client: FAF county data client.

    Returns:
        One CountyTradeFlowResult per home county.

    Raises:
        StateDownloadError: Propagated if the zip cannot be downloaded/parsed.
    """
    tables = client.download_state_tables(zip_filename)
    domestic_c2c = _domestic_only(tables["county_to_county"])
    domestic_c2f = _domestic_only(tables["county_to_faf"])
    domestic_f2c = _domestic_only(tables["faf_to_county"])

    return [
        compute_county_flows(fips_code, domestic_c2c, domestic_c2f, domestic_f2c)
        for fips_code in home_fips_codes
    ]


def run_pipeline(
    crosswalk: pd.DataFrame,
    client: FafCountyClient,
) -> tuple[list[CountyTradeFlowResult], IngestionSummary]:
    """Process every state's FAF5 zip, isolating per-state download failures.

    Args:
        crosswalk: DataFrame with `county_name`, `fips_code` columns.
        client: FAF county data client.

    Returns:
        Tuple of (all county results across all states, run summary).
    """
    results: list[CountyTradeFlowResult] = []
    summary = IngestionSummary(succeeded=[], failed={})

    for state_fips, zip_filename in STATE_ZIP_FILES.items():
        home_fips_codes = [fips for fips in crosswalk["fips_code"] if fips.startswith(state_fips)]
        if not home_fips_codes:
            continue

        logger.info("Processing state %s (%d home counties)...", state_fips, len(home_fips_codes))
        try:
            state_results = process_state(zip_filename, home_fips_codes, client)
        except StateDownloadError as exc:
            logger.warning("Skipping state %s: %s", state_fips, exc)
            summary.failed[state_fips] = str(exc)
            continue

        results.extend(state_results)
        summary.succeeded.append(state_fips)

    return results, summary


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def build_dataframe(results: list[CountyTradeFlowResult]) -> pd.DataFrame:
    """Assemble per-county flow results into a DataFrame.

    Args:
        results: List of per-county flow results.

    Returns:
        DataFrame with one row per county, keyed by `fips_code`.
    """
    return pd.DataFrame([vars(r) for r in results])


def join_county_names(flows: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join flow results onto the project's county/FIPS crosswalk.

    Args:
        flows: DataFrame from build_dataframe, with a `fips_code` column.
        crosswalk: DataFrame with `county_name`, `fips_code` columns.

    Returns:
        DataFrame with `county_name` first, followed by all flow columns.
    """
    merged = crosswalk.merge(flows, on="fips_code", how="inner")

    unmatched = set(crosswalk["fips_code"]) - set(merged["fips_code"])
    if unmatched:
        logger.warning("%d crosswalk counties had no FAF5 flow data: %s", len(unmatched), sorted(unmatched))

    return merged


def export_to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write the ingestion DataFrame to a local Parquet file.

    The pillar's `as_of_date` is stamped on before writing, so the vintage
    travels with the data rather than only with the docs (`pillar_vintage`).

    Args:
        df: DataFrame to export.
        output_path: Destination Parquet file path.
    """
    stamp_vintage(df, "D").to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def main() -> None:
    """Run the Source D ingestion pipeline over all US counties."""
    configure_logging()

    if not COUNTY_CROSSWALK_CACHE_PATH.exists():
        logger.error(
            "County crosswalk not found at %s; run ingest_source_a.py or "
            "ingest_source_c.py first to populate it.",
            COUNTY_CROSSWALK_CACHE_PATH,
        )
        sys.exit(1)
    crosswalk = pd.read_parquet(COUNTY_CROSSWALK_CACHE_PATH)

    client = FafCountyClient()
    results, summary = run_pipeline(crosswalk, client)

    if not results:
        logger.error("No counties were successfully processed; aborting without writing output.")
        sys.exit(1)

    df = build_dataframe(results)
    df = join_county_names(df, crosswalk)

    export_to_parquet(df, OUTPUT_PARQUET_PATH)
    logger.info("Succeeded: %d states, Failed: %d states", len(summary.succeeded), len(summary.failed))
    for state, reason in summary.failed.items():
        logger.warning("  %s -> %s", state, reason)


if __name__ == "__main__":
    main()
