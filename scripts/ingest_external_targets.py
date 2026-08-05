"""County-level outcomes from outside all six E_macro pillars.

Every validation in this repo to date is pillar-versus-pillar: the 50-pair
crossvalidation sweep, the 29-target matrix sweep, and the Source A marginal
harness all predict one pillar's feature from the others. That design measures
coherence and redundancy between six federal sources. It cannot measure
usefulness, because usefulness is defined relative to a target -- and every
target in those sweeps is another pillar's feature, which penalizes a source
precisely for agreeing with the pillars it will ship alongside
(`docs/downstream_target.md` Part 2).

No downstream label is obtainable for this project: it has no access to company
data and was scoped to public sources for exactly that reason
(`docs/PROJECT_GOAL.md`, "Operating constraints"). That closes
`docs/plans/source_a_next_steps.md` question 2 with "no", which makes an
external public target mandatory rather than optional.

This module supplies three, all from the ACS 2023 5-year summary file:

| column | ACS table | why this one |
|---|---|---|
| `broadband_rate` | B28002 | Closest public analogue to a FreeWheel-adjacent outcome -- household broadband adoption sits in the same domain as a Comcast downstream model. |
| `median_household_income` | B19013 | Economic level. Different texture, and the outcome most likely to be predicted by the size baseline alone, which makes it a useful hard case. |
| `median_age` | B01002 | Demographic texture, near-orthogonal to county size, and a real driver of ad audience composition. The size-only baseline should be weakest here. |

**Three rather than one, deliberately.** A single target repeats a mistake this
repo already caught: the Source A headline of +0.0010 rests on a basket that is
71% one BLS table, and `source-a-findings.md` §17.3 forbids publishing it without
that composition attached. One external target would be a basket of one.

None of the three is derived from any pillar, so none is circular. All three
correlate with county size and urbanicity, which is why the scoring harness in
`analyze_external_target.py` reports lift over a size baseline rather than raw
R2.

**No API key.** The Census data API began requiring one; these are the
table-based summary files published at `www2.census.gov`, which do not. Same
host and same keyless pattern as `county_population.py`.

Output: `data/external_targets.parquet`, cached after the first download.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EXTERNAL_TARGETS_PATH: Path = REPO_ROOT / "data" / "external_targets.parquet"

ACS_VINTAGE: str = "ACS 2023 5-year"
ACS_AS_OF_DATE: str = "2023-12-31"
ACS_BASE_URL: str = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
    "table-based-SF/data/5YRData/acsdt5y2023-{table}.dat"
)
REQUEST_TIMEOUT_SECONDS: int = 180

# Summary-file rows are one per geography across every summary level. County
# rows carry this GEO_ID prefix followed by the 5-digit FIPS code.
COUNTY_GEO_PREFIX: str = "0500000US"


@dataclass(frozen=True)
class ExternalTarget:
    """One county-level outcome drawn from an ACS summary-file table.

    Attributes:
        column: Output column name in `external_targets.parquet`.
        table: ACS table identifier, lowercase, as it appears in the file name.
        numerator: Estimate column forming the numerator, e.g. `B28002_E004`.
        denominator: Estimate column forming the denominator, or None when the
            table already publishes the quantity directly (a median).
        label: Human-readable description used in reports.
    """

    column: str
    table: str
    numerator: str
    denominator: str | None
    label: str


# B28002 line 004 is "Broadband of any type", line 001 the household universe.
# Verified against Autauga County, AL: E001 = E002 + E012 + E013 and
# E002 = E003 + E004, so the line numbering is the published hierarchy.
EXTERNAL_TARGETS: tuple[ExternalTarget, ...] = (
    ExternalTarget(
        column="broadband_rate",
        table="b28002",
        numerator="B28002_E004",
        denominator="B28002_E001",
        label="household broadband adoption rate",
    ),
    ExternalTarget(
        column="median_household_income",
        table="b19013",
        numerator="B19013_E001",
        denominator=None,
        label="median household income",
    ),
    ExternalTarget(
        column="median_age",
        table="b01002",
        numerator="B01002_E001",
        denominator=None,
        label="median age",
    ),
)

logger = logging.getLogger(__name__)


def _fetch_acs_table(target: ExternalTarget) -> pd.DataFrame:
    """Download one ACS summary-file table and reduce it to county rows.

    Args:
        target: The target whose table should be fetched.

    Returns:
        DataFrame with `fips_code` and the target's single output column.

    Raises:
        requests.HTTPError: If the Census download fails.
        KeyError: If an expected estimate column is absent, which means the
            table's line numbering changed and the mapping needs revisiting.
    """
    url = ACS_BASE_URL.format(table=target.table)
    logger.info("Downloading %s for %s...", target.table.upper(), target.column)
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    columns = ["GEO_ID", target.numerator]
    if target.denominator is not None:
        columns.append(target.denominator)

    raw = pd.read_csv(
        io.BytesIO(response.content),
        sep="|",
        usecols=columns,
        dtype={"GEO_ID": str},
        na_values=["", ".", "-", "null"],
    )

    counties = raw[raw["GEO_ID"].str.startswith(COUNTY_GEO_PREFIX)].copy()
    counties["fips_code"] = counties["GEO_ID"].str.removeprefix(COUNTY_GEO_PREFIX)

    numerator = pd.to_numeric(counties[target.numerator], errors="coerce")
    if target.denominator is None:
        # Census publishes medians with negative sentinels (-666666666 and
        # relatives) where an estimate could not be computed. Anything negative
        # is a sentinel for both of the medians used here.
        values = numerator.where(numerator >= 0)
    else:
        denominator = pd.to_numeric(counties[target.denominator], errors="coerce")
        values = numerator / denominator.where(denominator > 0)

    logger.info(
        "  %s: %d counties, %d null",
        target.column,
        len(counties),
        int(values.isna().sum()),
    )
    return pd.DataFrame({"fips_code": counties["fips_code"], target.column: values})


def fetch_external_targets(cache_path: Path = EXTERNAL_TARGETS_PATH) -> pd.DataFrame:
    """Load the external target table, downloading and caching on first use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with `fips_code`, one column per entry in EXTERNAL_TARGETS,
        and `as_of_date` matching the convention in `pillar_vintage.py`.

    Raises:
        requests.HTTPError: If any Census download fails.
    """
    if cache_path.exists():
        logger.info("Loading cached external targets from %s", cache_path)
        return pd.read_parquet(cache_path)

    frames = [_fetch_acs_table(target) for target in EXTERNAL_TARGETS]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="fips_code", how="outer")

    merged["as_of_date"] = ACS_AS_OF_DATE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, engine="pyarrow", index=False)
    logger.info("Cached %d counties to %s", len(merged), cache_path)
    return merged


def main() -> None:
    """Refresh the external-target cache and report coverage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    targets = fetch_external_targets()
    logger.info("%s: %d counties", ACS_VINTAGE, len(targets))
    for target in EXTERNAL_TARGETS:
        series = targets[target.column].dropna()
        logger.info(
            "  %-26s n=%4d  mean=%12.3f  sd=%12.3f",
            target.column,
            len(series),
            series.mean(),
            series.std(),
        )


if __name__ == "__main__":
    main()
