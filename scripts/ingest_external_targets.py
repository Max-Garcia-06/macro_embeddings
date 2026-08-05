"""County-level outcomes from outside all six E_macro pillars, with their noise.

Every validation in this repo before these targets existed is
pillar-versus-pillar: the 50-pair crossvalidation sweep, the 29-target matrix
sweep, and the Source A marginal harness all predict one pillar's feature from
the others. That design measures coherence and redundancy between six federal
sources. It cannot measure usefulness, because usefulness is defined relative to
a target -- and every target in those sweeps is another pillar's feature, which
penalizes a source precisely for agreeing with the pillars it will ship
alongside (`docs/downstream_target.md` Part 2).

No downstream label is obtainable for this project: it has no access to company
data and was scoped to public sources for exactly that reason
(`docs/PROJECT_GOAL.md`, "Operating constraints"). That closes
`docs/plans/source_a_next_steps.md` question 2 with "no", which makes an
external public target mandatory rather than optional.

This module supplies five, all from the ACS 2023 5-year summary file:

| column | ACS table | texture |
|---|---|---|
| `broadband_rate` | B28002 | Closest public analogue to a FreeWheel-adjacent outcome. |
| `median_household_income` | B19013 | Economic level; the hardest case for beating a size baseline. |
| `median_age` | B01002 | Demographic; near-orthogonal to county size. |
| `median_home_value` | B25077 | Asset and wealth, distinct from income flow. |
| `mean_commute_minutes` | B08013 / B08012 | Settlement geometry and labour-market shape. |

**Five rather than one, deliberately.** A single target repeats a mistake this
repo already caught: the Source A headline of +0.0010 rests on a basket that is
71% one BLS table, and `source-a-findings.md` §17.3 forbids publishing it without
that composition attached. One external target is a basket of one.

None is derived from any pillar, so none is circular. All correlate with county
size and urbanicity, which is why `analyze_external_target.py` reports lift over
a size baseline rather than raw R2.

## Margins of error ship alongside the estimates

ACS publishes a 90% margin of error for every estimate, and county estimates in
small counties are noisy: a share of their variance is sampling error that no
model can predict. Without that quantified, a model scoring badly on small
counties is indistinguishable from a target that is mostly noise there -- which
is exactly the ambiguity that left `external-target-findings.md` §5 unable to
settle whether E_macro helps more or less on thin units.

Each target therefore ships a `{column}_se` companion, the standard error on its
own scale, so the harness can compute a per-stratum noise floor. Derived
quantities use the Census ACS General Handbook formulas:

- **Proportion** (numerator is a subset of the denominator, e.g. broadband
  households out of all households):
  `SE(p) = sqrt(SE(N)^2 - p^2 * SE(D)^2) / D`, falling back to the ratio form
  when the radicand goes negative, as the handbook directs.
- **Ratio** (numerator is a different quantity, e.g. aggregate travel minutes
  over workers): `SE(R) = sqrt(SE(N)^2 + R^2 * SE(D)^2) / D`.
- **Published medians** carry their margin of error directly.

**No API key.** The Census data API began requiring one; these are the
table-based summary files published at `www2.census.gov`, which do not. Same
host and keyless pattern as `county_population.py`.

Output: `data/external_targets.parquet`, cached after the first download.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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

# ACS publishes margins of error at 90% confidence: MOE = 1.645 * SE.
MOE_Z_SCORE: float = 1.645


@dataclass(frozen=True)
class ExternalTarget:
    """One county-level outcome drawn from the ACS summary file.

    Attributes:
        column: Output column name in `external_targets.parquet`.
        table: ACS table holding the numerator, lowercase as in the file name.
        numerator: Estimate line for the numerator, e.g. `B28002_E004`.
        denominator: Estimate line for the denominator, or None for a published
            median that needs no division.
        denominator_table: Table holding the denominator when it differs from
            `table`; None means the same table.
        kind: "median" for a directly published value, "proportion" when the
            numerator is a subset of the denominator, "ratio" otherwise. Selects
            the standard-error formula.
        label: Human-readable description used in reports.
    """

    column: str
    table: str
    numerator: str
    denominator: str | None
    denominator_table: str | None
    kind: str
    label: str


# B28002 line 004 is "Broadband of any type", line 001 the household universe.
# Verified against Autauga County, AL: E001 = E002 + E012 + E013 and
# E002 = E003 + E004, so the line numbering is the published hierarchy.
#
# B08013 is aggregate travel time to work in minutes; B08012 line 001 is the
# matching universe of workers. Autauga: 689,705 / 25,415 = 27.1 minutes.
EXTERNAL_TARGETS: tuple[ExternalTarget, ...] = (
    ExternalTarget(
        column="broadband_rate",
        table="b28002",
        numerator="B28002_E004",
        denominator="B28002_E001",
        denominator_table=None,
        kind="proportion",
        label="household broadband adoption rate",
    ),
    ExternalTarget(
        column="median_household_income",
        table="b19013",
        numerator="B19013_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median household income",
    ),
    ExternalTarget(
        column="median_age",
        table="b01002",
        numerator="B01002_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median age",
    ),
    ExternalTarget(
        column="median_home_value",
        table="b25077",
        numerator="B25077_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median owner-occupied home value",
    ),
    ExternalTarget(
        column="mean_commute_minutes",
        table="b08013",
        numerator="B08013_E001",
        denominator="B08012_E001",
        denominator_table="b08012",
        kind="ratio",
        label="mean commute time to work",
    ),
)

logger = logging.getLogger(__name__)


def _download_table(table: str, columns: list[str]) -> pd.DataFrame:
    """Fetch one ACS summary-file table and reduce it to county rows.

    Args:
        table: Lowercase table identifier as it appears in the file name.
        columns: Estimate and margin columns to retain, alongside `GEO_ID`.

    Returns:
        DataFrame indexed by `fips_code` carrying the requested columns.

    Raises:
        requests.HTTPError: If the Census download fails.
        ValueError: If an expected column is absent, which means the table's
            line numbering changed and the mapping needs revisiting.
    """
    response = requests.get(ACS_BASE_URL.format(table=table), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    raw = pd.read_csv(
        io.BytesIO(response.content),
        sep="|",
        usecols=["GEO_ID", *columns],
        dtype={"GEO_ID": str},
        na_values=["", ".", "-", "null"],
    )
    counties = raw[raw["GEO_ID"].str.startswith(COUNTY_GEO_PREFIX)].copy()
    counties["fips_code"] = counties["GEO_ID"].str.removeprefix(COUNTY_GEO_PREFIX)
    return counties.set_index("fips_code")[columns].apply(pd.to_numeric, errors="coerce")


def _moe_column(estimate_column: str) -> str:
    """Return the margin-of-error column paired with an estimate column.

    Summary-file columns are named `{TABLE}_E{line}` and `{TABLE}_M{line}`.

    Args:
        estimate_column: Estimate column name, e.g. `B28002_E004`.

    Returns:
        The matching margin-of-error column name.
    """
    table, line = estimate_column.split("_E")
    return f"{table}_M{line}"


def _derive(target: ExternalTarget) -> pd.DataFrame:
    """Compute one target's value and standard error for every county.

    Args:
        target: The target to derive.

    Returns:
        DataFrame with `fips_code`, the target column, and its `_se` companion.

    Raises:
        requests.HTTPError: If any Census download fails.
        ValueError: If `target.kind` is not one of the three supported forms.
    """
    logger.info("Downloading %s for %s...", target.table.upper(), target.column)
    numerator_columns = [target.numerator, _moe_column(target.numerator)]
    frame = _download_table(target.table, numerator_columns)

    numerator = frame[target.numerator]
    numerator_se = frame[_moe_column(target.numerator)] / MOE_Z_SCORE

    if target.kind == "median":
        # Census flags uncomputable medians with negative sentinels
        # (-666666666 and relatives); anything negative is a sentinel here.
        values = numerator.where(numerator >= 0)
        standard_error = numerator_se.where(numerator >= 0)
    elif target.kind in {"proportion", "ratio"}:
        denominator_table = target.denominator_table or target.table
        if denominator_table != target.table:
            logger.info("  and %s for its denominator...", denominator_table.upper())
            denominator_columns = [target.denominator, _moe_column(target.denominator)]
            denominator_frame = _download_table(denominator_table, denominator_columns)
        else:
            denominator_frame = _download_table(
                target.table, [target.denominator, _moe_column(target.denominator)]
            )

        denominator = denominator_frame[target.denominator].reindex(frame.index)
        denominator_se = (
            denominator_frame[_moe_column(target.denominator)].reindex(frame.index) / MOE_Z_SCORE
        )
        safe_denominator = denominator.where(denominator > 0)
        values = numerator / safe_denominator

        # ACS General Handbook: the proportion form subtracts, and falls back to
        # the ratio form when the radicand goes negative.
        squared = numerator_se**2 - (values**2) * (denominator_se**2)
        if target.kind == "ratio":
            squared = numerator_se**2 + (values**2) * (denominator_se**2)
        else:
            fallback = numerator_se**2 + (values**2) * (denominator_se**2)
            squared = squared.where(squared >= 0, fallback)
        standard_error = np.sqrt(squared) / safe_denominator
    else:
        raise ValueError(f"Unsupported target kind: {target.kind!r}")

    logger.info(
        "  %s: %d counties, %d null, median SE %.4g",
        target.column,
        len(values),
        int(values.isna().sum()),
        float(standard_error.median(skipna=True)),
    )
    return pd.DataFrame(
        {
            "fips_code": frame.index,
            target.column: values.to_numpy(),
            f"{target.column}_se": standard_error.to_numpy(),
        }
    )


def fetch_external_targets(cache_path: Path = EXTERNAL_TARGETS_PATH) -> pd.DataFrame:
    """Load the external target table, downloading and caching on first use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with `fips_code`, one value and one `_se` column per entry in
        EXTERNAL_TARGETS, and `as_of_date` matching `pillar_vintage.py`.

    Raises:
        requests.HTTPError: If any Census download fails.
    """
    if cache_path.exists():
        logger.info("Loading cached external targets from %s", cache_path)
        return pd.read_parquet(cache_path)

    merged = _derive(EXTERNAL_TARGETS[0])
    for target in EXTERNAL_TARGETS[1:]:
        merged = merged.merge(_derive(target), on="fips_code", how="outer")

    merged["as_of_date"] = ACS_AS_OF_DATE
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, engine="pyarrow", index=False)
    logger.info("Cached %d counties to %s", len(merged), cache_path)
    return merged


def main() -> None:
    """Refresh the external-target cache and report coverage and noise."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    targets = fetch_external_targets()
    logger.info("%s: %d counties", ACS_VINTAGE, len(targets))
    for target in EXTERNAL_TARGETS:
        values = targets[target.column].dropna()
        errors = targets[f"{target.column}_se"].dropna()
        noise_share = float((errors**2).mean() / values.var()) if len(values) > 1 else float("nan")
        logger.info(
            "  %-24s n=%4d  mean=%11.3f  sd=%10.3f  sampling noise = %.1f%% of variance",
            target.column,
            len(values),
            values.mean(),
            values.std(),
            100 * noise_share,
        )


if __name__ == "__main__":
    main()
