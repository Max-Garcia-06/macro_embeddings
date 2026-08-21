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

This module supplies 42, all from the ACS 2023 5-year summary file, spanning
eight constructs -- income, housing cost, housing stock, education,
labour-force structure, household composition, technology access, and
commuting -- and capped at no more than 6 targets per ACS table family. The
original five are the seed:

| column | ACS table | texture |
|---|---|---|
| `broadband_rate` | B28002 | Closest public analogue to a FreeWheel-adjacent outcome. |
| `median_household_income` | B19013 | Economic level; the hardest case for beating a size baseline. |
| `median_age` | B01002 | Demographic; near-orthogonal to county size. |
| `median_home_value` | B25077 | Asset and wealth, distinct from income flow. |
| `mean_commute_minutes` | B08013 / B08012 | Settlement geometry and labour-market shape. |

**Many rather than one, deliberately.** A single target repeats a mistake this
repo already caught: the Source A headline of +0.0010 rests on a basket that is
71% one BLS table, and `source-a-findings.md` §17.3 forbids publishing it without
that composition attached. One external target is a basket of one -- and the
28-target matrix sweep's defect, 71% from a single table, is exactly what the
6-per-table cap forbids happening again here.

Each target is screened against all six pillars before admission and recorded
in `TARGET_CIRCULARITY`: `"clean"` means no pillar measures the construct;
`"ablated"` means a pillar column comes close enough to restate it, and
`analyze_external_target.py`'s `TARGET_RESTATEMENTS` drops that column from the
design when the target is scored. Nothing here is derived from a pillar
outright -- unemployment rate, sector employment shares, and freight volume
were candidate constructs and are rejected rather than admitted, because
Source C, Source B, and Source D measure them directly. All targets correlate
with county size and urbanicity, which is why `analyze_external_target.py`
reports lift over a size baseline rather than raw R2.

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
from functools import lru_cache
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


# Every target's standing against the six pillars, decided before it was
# admitted. "clean" means no pillar measures the construct; "ablated" means a
# pillar column is close enough to restate it, and `analyze_external_target.py`
# must carry a matching `TARGET_RESTATEMENTS` entry that removes it from the
# design. Constructs a pillar measures outright are not listed because they were
# rejected rather than admitted:
#
#   - county unemployment rate     -> Source C measures it directly
#   - any sector employment share  -> Source B location quotients
#   - freight or logistics volume  -> Source D
#
TARGET_CIRCULARITY: dict[str, str] = {
    "broadband_rate": "clean",
    "median_household_income": "ablated",   # Source E wage_per_return_thousands
    "median_age": "ablated",                # Source F retirement_destination
    "median_home_value": "clean",
    "mean_commute_minutes": "clean",
    "per_capita_income": "ablated",         # Source E wage_per_return_thousands
    "median_family_income": "ablated",      # Source E wage_per_return_thousands
    "median_gross_rent": "clean",
    "median_contract_rent": "clean",
    "median_monthly_housing_cost": "ablated",  # Source F housing_stress
    "median_year_built": "clean",
    "mean_household_size": "clean",
    "owner_occupied_share": "clean",
    "poverty_rate": "ablated",              # Source F persistent_poverty
    "family_household_share": "clean",
    "single_unit_share": "clean",
    "bachelors_share": "ablated",           # Source F low_postsecondary_ed
    "graduate_share": "ablated",            # Source F low_postsecondary_ed
    "labor_force_participation": "ablated", # Source F low_employment
    # Step 5 additions, probed against b25004/b25040/b08301/b05002/b07001/
    # b19052/b19055/b25081/b28003/b09002. None of these constructs -- vacancy,
    # heating fuel, commute mode, nativity, residential mobility, household
    # earnings/SS receipt, mortgage status, computer ownership, child family
    # structure -- is measured by any of the six pillars, with one exception:
    # household Social Security receipt is close enough to Source F's
    # retirement_destination migration flag to ablate on the same reasoning
    # already applied to median_age.
    "housing_vacancy_rate": "clean",
    "electric_heating_share": "clean",
    "gas_heating_share": "clean",
    "bottled_gas_heating_share": "clean",
    "fuel_oil_heating_share": "clean",
    "no_fuel_used_share": "clean",
    "drove_alone_share": "clean",
    "carpooled_share": "clean",
    "public_transit_share": "clean",
    "walked_share": "clean",
    "work_from_home_share": "clean",
    "foreign_born_share": "clean",
    "naturalized_share_of_foreign_born": "clean",
    "same_house_share": "clean",
    "moved_within_county_share": "clean",
    "moved_different_state_share": "clean",
    "children_married_couple_share": "clean",
    "children_female_householder_share": "clean",
    "children_male_householder_share": "clean",
    "household_earnings_share": "clean",
    "household_ss_income_share": "ablated",  # Source F retirement_destination
    "mortgaged_share": "clean",
    "computer_ownership_share": "clean",
}

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
    # Verified against Autauga County, AL (see task-3-brief.md Step 4). Every
    # numerator/denominator pair here reconciles: B11001_E001 and B25003_E001
    # both read 22,523 for Autauga -- households and occupied housing units
    # are the same universe.
    ExternalTarget(
        column="per_capita_income",
        table="b19301",
        numerator="B19301_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="per capita income",
    ),
    ExternalTarget(
        column="median_family_income",
        table="b19113",
        numerator="B19113_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median family income",
    ),
    ExternalTarget(
        column="median_gross_rent",
        table="b25064",
        numerator="B25064_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median gross rent",
    ),
    ExternalTarget(
        column="median_contract_rent",
        table="b25058",
        numerator="B25058_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median contract rent",
    ),
    ExternalTarget(
        column="median_monthly_housing_cost",
        table="b25105",
        numerator="B25105_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median monthly owner housing cost",
    ),
    ExternalTarget(
        column="median_year_built",
        table="b25035",
        numerator="B25035_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="median year structure built",
    ),
    ExternalTarget(
        column="mean_household_size",
        table="b25010",
        numerator="B25010_E001",
        denominator=None,
        denominator_table=None,
        kind="median",
        label="mean household size",
    ),
    ExternalTarget(
        column="owner_occupied_share",
        table="b25003",
        numerator="B25003_E002",
        denominator="B25003_E001",
        denominator_table=None,
        kind="proportion",
        label="owner-occupied housing share",
    ),
    ExternalTarget(
        column="poverty_rate",
        table="b17001",
        numerator="B17001_E002",
        denominator="B17001_E001",
        denominator_table=None,
        kind="proportion",
        label="share below the poverty line",
    ),
    ExternalTarget(
        column="family_household_share",
        table="b11001",
        numerator="B11001_E002",
        denominator="B11001_E001",
        denominator_table=None,
        kind="proportion",
        label="family-household share",
    ),
    ExternalTarget(
        column="single_unit_share",
        table="b25024",
        numerator="B25024_E002",
        denominator="B25024_E001",
        denominator_table=None,
        kind="proportion",
        label="single-unit detached housing share",
    ),
    ExternalTarget(
        column="bachelors_share",
        table="b15003",
        numerator="B15003_E022",
        denominator="B15003_E001",
        denominator_table=None,
        kind="proportion",
        label="bachelor's degree share, age 25+",
    ),
    ExternalTarget(
        column="graduate_share",
        table="b15003",
        numerator="B15003_E023",
        denominator="B15003_E001",
        denominator_table=None,
        kind="proportion",
        label="graduate degree share, age 25+",
    ),
    ExternalTarget(
        column="labor_force_participation",
        table="b23025",
        numerator="B23025_E002",
        denominator="B23025_E001",
        denominator_table=None,
        kind="proportion",
        label="labour force participation rate",
    ),
    # Step 5: probed candidates, admitted only after county coverage exceeded
    # 2,500 and the Autauga value checked out arithmetically (task-3-report.md
    # carries the full probe output). B25004's universe is vacant housing
    # units only, so its rate is built against B25024's total-housing-units
    # denominator -- the same table already used for single_unit_share, and
    # the two reconcile exactly: 22,523 occupied + 2,208 vacant = 24,731.
    ExternalTarget(
        column="housing_vacancy_rate",
        table="b25004",
        numerator="B25004_E001",
        denominator="B25024_E001",
        denominator_table="b25024",
        kind="proportion",
        label="housing vacancy rate",
    ),
    ExternalTarget(
        column="electric_heating_share",
        table="b25040",
        numerator="B25040_E004",
        denominator="B25040_E001",
        denominator_table=None,
        kind="proportion",
        label="occupied units heated by electricity",
    ),
    ExternalTarget(
        column="gas_heating_share",
        table="b25040",
        numerator="B25040_E002",
        denominator="B25040_E001",
        denominator_table=None,
        kind="proportion",
        label="occupied units heated by utility gas",
    ),
    ExternalTarget(
        column="bottled_gas_heating_share",
        table="b25040",
        numerator="B25040_E003",
        denominator="B25040_E001",
        denominator_table=None,
        kind="proportion",
        label="occupied units heated by bottled, tank, or LP gas",
    ),
    ExternalTarget(
        column="fuel_oil_heating_share",
        table="b25040",
        numerator="B25040_E005",
        denominator="B25040_E001",
        denominator_table=None,
        kind="proportion",
        label="occupied units heated by fuel oil, kerosene, etc.",
    ),
    ExternalTarget(
        column="no_fuel_used_share",
        table="b25040",
        numerator="B25040_E010",
        denominator="B25040_E001",
        denominator_table=None,
        kind="proportion",
        label="occupied units using no heating fuel",
    ),
    ExternalTarget(
        column="drove_alone_share",
        table="b08301",
        numerator="B08301_E003",
        denominator="B08301_E001",
        denominator_table=None,
        kind="proportion",
        label="workers who drove alone to work",
    ),
    ExternalTarget(
        column="carpooled_share",
        table="b08301",
        numerator="B08301_E004",
        denominator="B08301_E001",
        denominator_table=None,
        kind="proportion",
        label="workers who carpooled to work",
    ),
    ExternalTarget(
        column="public_transit_share",
        table="b08301",
        numerator="B08301_E010",
        denominator="B08301_E001",
        denominator_table=None,
        kind="proportion",
        label="workers who used public transportation",
    ),
    ExternalTarget(
        column="walked_share",
        table="b08301",
        numerator="B08301_E019",
        denominator="B08301_E001",
        denominator_table=None,
        kind="proportion",
        label="workers who walked to work",
    ),
    ExternalTarget(
        column="work_from_home_share",
        table="b08301",
        numerator="B08301_E021",
        denominator="B08301_E001",
        denominator_table=None,
        kind="proportion",
        label="workers who worked from home",
    ),
    ExternalTarget(
        column="foreign_born_share",
        table="b05002",
        numerator="B05002_E013",
        denominator="B05002_E001",
        denominator_table=None,
        kind="proportion",
        label="foreign-born population share",
    ),
    ExternalTarget(
        column="naturalized_share_of_foreign_born",
        table="b05002",
        numerator="B05002_E014",
        denominator="B05002_E013",
        denominator_table=None,
        kind="proportion",
        label="naturalized citizens as a share of the foreign-born",
    ),
    ExternalTarget(
        column="same_house_share",
        table="b07001",
        numerator="B07001_E017",
        denominator="B07001_E001",
        denominator_table=None,
        kind="proportion",
        label="population living in the same house one year ago",
    ),
    ExternalTarget(
        column="moved_within_county_share",
        table="b07001",
        numerator="B07001_E033",
        denominator="B07001_E001",
        denominator_table=None,
        kind="proportion",
        label="population that moved within the same county",
    ),
    ExternalTarget(
        column="moved_different_state_share",
        table="b07001",
        numerator="B07001_E065",
        denominator="B07001_E001",
        denominator_table=None,
        kind="proportion",
        label="population that moved from a different state",
    ),
    ExternalTarget(
        column="children_married_couple_share",
        table="b09002",
        numerator="B09002_E002",
        denominator="B09002_E001",
        denominator_table=None,
        kind="proportion",
        label="own children under 18 in married-couple families",
    ),
    ExternalTarget(
        column="children_female_householder_share",
        table="b09002",
        numerator="B09002_E015",
        denominator="B09002_E001",
        denominator_table=None,
        kind="proportion",
        label="own children under 18 with a female householder, no spouse present",
    ),
    ExternalTarget(
        column="children_male_householder_share",
        table="b09002",
        numerator="B09002_E009",
        denominator="B09002_E001",
        denominator_table=None,
        kind="proportion",
        label="own children under 18 with a male householder, no spouse present",
    ),
    ExternalTarget(
        column="household_earnings_share",
        table="b19052",
        numerator="B19052_E002",
        denominator="B19052_E001",
        denominator_table=None,
        kind="proportion",
        label="households with earnings",
    ),
    ExternalTarget(
        column="household_ss_income_share",
        table="b19055",
        numerator="B19055_E002",
        denominator="B19055_E001",
        denominator_table=None,
        kind="proportion",
        label="households with Social Security income",
    ),
    ExternalTarget(
        column="mortgaged_share",
        table="b25081",
        numerator="B25081_E002",
        denominator="B25081_E001",
        denominator_table=None,
        kind="proportion",
        label="owner-occupied units with a mortgage",
    ),
    ExternalTarget(
        column="computer_ownership_share",
        table="b28003",
        numerator="B28003_E002",
        denominator="B28003_E001",
        denominator_table=None,
        kind="proportion",
        label="households with a computer",
    ),
)

logger = logging.getLogger(__name__)


def _fetch_table_uncached(table: str) -> pd.DataFrame:
    """Fetch one ACS summary-file table and reduce it to county rows.

    Retains every column, so one download serves every target drawn from this
    table. Callers select the lines they need.

    Args:
        table: Lowercase table identifier as it appears in the file name.

    Returns:
        DataFrame indexed by `fips_code` carrying all numeric table columns.

    Raises:
        requests.HTTPError: If the Census download fails.
    """
    response = requests.get(ACS_BASE_URL.format(table=table), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    raw = pd.read_csv(
        io.BytesIO(response.content),
        sep="|",
        dtype={"GEO_ID": str},
        na_values=["", ".", "-", "null"],
        low_memory=False,
    )
    counties = raw[raw["GEO_ID"].str.startswith(COUNTY_GEO_PREFIX)].copy()
    counties["fips_code"] = counties["GEO_ID"].str.removeprefix(COUNTY_GEO_PREFIX)
    counties = counties.set_index("fips_code").drop(columns=["GEO_ID"])
    return counties.apply(pd.to_numeric, errors="coerce")


@lru_cache(maxsize=None)
def _download_table(table: str) -> pd.DataFrame:
    """Memoized `_fetch_table_uncached`, so one table downloads once per run.

    Args:
        table: Lowercase table identifier.

    Returns:
        The cached county-row frame for that table.
    """
    return _fetch_table_uncached(table)


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
    frame = _download_table(target.table)
    missing = [c for c in numerator_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{target.column}: {target.table.upper()} is missing {missing}; "
            "the table's line numbering changed and the mapping needs revisiting"
        )

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
            denominator_frame = _download_table(denominator_table)
            denominator_missing = [c for c in denominator_columns if c not in denominator_frame.columns]
            if denominator_missing:
                raise ValueError(
                    f"{target.column}: {denominator_table.upper()} is missing {denominator_missing}; "
                    "the table's line numbering changed and the mapping needs revisiting"
                )
        else:
            denominator_columns = [target.denominator, _moe_column(target.denominator)]
            denominator_frame = _download_table(target.table)
            denominator_missing = [c for c in denominator_columns if c not in denominator_frame.columns]
            if denominator_missing:
                raise ValueError(
                    f"{target.column}: {target.table.upper()} is missing {denominator_missing}; "
                    "the table's line numbering changed and the mapping needs revisiting"
                )

        denominator = denominator_frame[target.denominator].reindex(frame.index)
        denominator_se = (
            denominator_frame[_moe_column(target.denominator)].reindex(frame.index) / MOE_Z_SCORE
        )

        # Census flags suppressed cells with negative sentinels on either side
        # of the ratio, not just in the numerator: a suppressed denominator
        # cell carries the same -666666666-family markers. Mask both before
        # dividing, or a suppressed numerator survives as a large negative
        # rate instead of the null it should be.
        safe_numerator = numerator.where(numerator >= 0)
        safe_numerator_se = numerator_se.where(numerator >= 0)
        safe_denominator = denominator.where(denominator > 0)
        safe_denominator_se = denominator_se.where(denominator >= 0)
        values = safe_numerator / safe_denominator

        # ACS General Handbook: the proportion form subtracts, and falls back to
        # the ratio form when the radicand goes negative.
        squared = safe_numerator_se**2 - (values**2) * (safe_denominator_se**2)
        if target.kind == "ratio":
            squared = safe_numerator_se**2 + (values**2) * (safe_denominator_se**2)
        else:
            fallback = safe_numerator_se**2 + (values**2) * (safe_denominator_se**2)
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
