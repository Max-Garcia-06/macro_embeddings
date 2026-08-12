"""Publish the frozen feature schemas for Sources B, C, D and F.

`docs/PROJECT_GOAL.md` sets this repo's bar at production feature-store quality:
a stable schema keyed on `fips_code` with documented coverage and null
semantics. Sources A and E had that; B, C, D and F shipped validated signal with
no schema document at all (`docs/pillar_status.md`, "missing paperwork").

Written from the parquets rather than by hand, on the same argument
`export_source_a_schema.py` makes: coverage figures quoted in prose drift away
from the data they describe, and a schema document that has drifted is worse
than none, because a consumer trusts it.

Two things this module does that the Source A exporter did not:

- **Column status is read from `pillar_matrix`, not restated here.** A column in
  `blocks[pillar]` ships; one in `SIZE_COLUMNS` is held out as a scale measure;
  one in a pillar's diagnostic tuple is written but never served. That makes it
  impossible for a schema doc to claim a column ships when the matrix withholds
  it -- the failure mode a hand-written doc invites.
- **Derived columns are documented alongside the parquet's own.** Source D's ten
  commodity shares and Source F's `distress_count` exist only after
  `pillar_matrix.derived_pillar_frames()` runs, and a consumer reading the
  parquet needs to know they are the shipping features.

Outputs: `docs/source_{b,c,d,f}_feature_schema.md` and the matching
`outputs/source_{b,c,d,f}_feature_schema.csv`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analyze_source_b_industry_mix import NAICS2_LABELS
from ingest_source_b import NAICS2_CODES
from ingest_source_d import SCTG_GROUPS
from pillar_matrix import (
    NON_FEATURE_COLUMNS,
    SIZE_COLUMNS,
    build_matrix,
    derived_pillar_frames,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
DOCS_DIR: Path = REPO_ROOT / "docs"

SIZE_DEPENDENCE_PATH: Path = OUTPUTS_DIR / "feature_size_dependence.csv"

# Tier boundaries for |r| with county size, matching the tiering published in
# `docs/downstream_target.md` Part 2 and used by `export_source_a_schema.py`.
SIZE_TIER_ONE_THRESHOLD: float = 0.30
SIZE_TIER_TWO_THRESHOLD: float = 0.15

# FAF5's five SCTG supergroups, as the county product ships them.
SCTG_LABELS: dict[str, str] = {
    "sctg0109": "SCTG 01-09, agriculture and food products",
    "sctg1014": "SCTG 10-14, mining and nonmetallic minerals",
    "sctg1519": "SCTG 15-19, coal and petroleum products",
    "sctg2033": "SCTG 20-33, chemicals, plastics, wood, paper and base metals",
    "sctg3499": "SCTG 34-99, machinery, electronics, vehicles and mixed freight",
}

logger = logging.getLogger(__name__)


def _size_tier(abs_r: float) -> str:
    """Bucket a feature's correlation with county size into a reporting tier.

    Args:
        abs_r: Absolute Pearson r against `log10(Census population)`.

    Returns:
        Tier label, matching `export_source_a_schema._size_tier`.
    """
    if abs_r >= SIZE_TIER_ONE_THRESHOLD:
        return "1 (size in disguise)"
    if abs_r >= SIZE_TIER_TWO_THRESHOLD:
        return "2 (partly size)"
    return "3 (size-free)"


def load_size_dependence() -> pd.Series:
    """Read each feature's correlation with county size.

    Returns:
        Series indexed by feature name holding `r_with_log_size`, empty if the
        scan has not been run.
    """
    if not SIZE_DEPENDENCE_PATH.exists():
        logger.warning("%s absent; every size tier will read 'unscanned'", SIZE_DEPENDENCE_PATH)
        return pd.Series(dtype="float64")
    return pd.read_csv(SIZE_DEPENDENCE_PATH).set_index("feature")["r_with_log_size"]


@dataclass(frozen=True)
class PillarSpec:
    """One pillar's schema document.

    Attributes:
        pillar: Pillar letter, "B" through "F".
        parquet: Filename under `data/`.
        doc_name: Output filename under `docs/`.
        descriptions: One line per column, keyed by column name.
        status_overrides: Status text for columns the generic rule mislabels.
        body: Markdown template. Filled with `columns_table`, `size_table`,
            `row_count`, `ships_count` and `as_of_date`.
    """

    pillar: str
    parquet: str
    doc_name: str
    descriptions: dict[str, str]
    status_overrides: dict[str, str]
    body: str


def column_status(
    column: str, pillar: str, blocks: dict[str, list[str]], overrides: dict[str, str]
) -> str:
    """Classify one column by how `pillar_matrix` treats it.

    Args:
        column: Column name.
        pillar: Owning pillar letter.
        blocks: Pillar-to-columns mapping from `build_matrix`.
        overrides: Per-column status text that wins over the generic rule.

    Returns:
        One of "ships", "size control", "identifier", or an override string.
    """
    if column in overrides:
        return overrides[column]
    if column in blocks[pillar]:
        return "ships"
    if column in SIZE_COLUMNS:
        return "size control"
    if column in NON_FEATURE_COLUMNS:
        return "identifier"
    return "diagnostic"


def build_schema(
    spec: PillarSpec,
    frame: pd.DataFrame,
    blocks: dict[str, list[str]],
    size_dependence: pd.Series,
) -> pd.DataFrame:
    """Assemble one row per column from the frame it describes.

    Args:
        spec: The pillar being documented.
        frame: Its parquet joined with the columns `pillar_matrix` derives.
        blocks: Pillar-to-columns mapping from `build_matrix`.
        size_dependence: Output of `load_size_dependence`.

    Returns:
        DataFrame with columns `column`, `dtype`, `status`, `description`,
        `nulls`, `fires_on`, `fire_rate`, `size_tier` and `r_with_size`.

    Raises:
        ValueError: If the frame carries a column the spec does not describe.
    """
    undocumented = [col for col in frame.columns if col not in spec.descriptions]
    if undocumented:
        raise ValueError(
            f"Source {spec.pillar} has undocumented column(s): {undocumented}. "
            "Every column a consumer receives needs a line in COLUMN_DESCRIPTIONS."
        )

    rows: list[dict[str, object]] = []
    for column in frame.columns:
        values = frame[column]
        is_boolean = pd.api.types.is_bool_dtype(values)
        r = float(size_dependence.get(column, float("nan")))
        firing = int(values.fillna(False).sum()) if is_boolean else None
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "status": column_status(column, spec.pillar, blocks, spec.status_overrides),
                "description": spec.descriptions[column],
                "nulls": int(values.isna().sum()),
                "fires_on": firing,
                "fire_rate": round(firing / len(frame), 4) if is_boolean else None,
                "size_tier": _size_tier(abs(r)) if r == r else "unscanned",
                "r_with_size": round(r, 3) if r == r else None,
            }
        )
    return pd.DataFrame(rows)


def columns_table(schema: pd.DataFrame) -> str:
    """Render the per-column contract table.

    Args:
        schema: Output of `build_schema`.

    Returns:
        Markdown table text.
    """
    header = (
        "| column | dtype | status | nulls | size tier | description |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in schema.itertuples():
        description = row.description
        # `fires_on` is None for continuous columns, which pandas stores as NaN
        # once the column holds any integer -- test for presence, not for None.
        if pd.notna(row.fires_on):
            description = (
                f"{description} Fires on {int(row.fires_on):,} counties ({row.fire_rate:.1%})."
            )
        lines.append(
            f"| `{row.column}` | {row.dtype} | {row.status} | {row.nulls:,} | "
            f"{row.size_tier} | {description} |"
        )
    return header + "\n".join(lines)


def size_table(schema: pd.DataFrame) -> str:
    """Render the how-much-of-this-is-county-size table for shipping columns.

    Args:
        schema: Output of `build_schema`.

    Returns:
        Markdown table text, ordered by `|r|` descending.
    """
    scored = schema[(schema["status"] == "ships") & schema["r_with_size"].notna()].copy()
    scored["abs_r"] = scored["r_with_size"].abs()
    scored = scored.sort_values("abs_r", ascending=False)
    header = "| column | r vs `log_population` | tier |\n|---|---|---|\n"
    lines = [
        f"| `{row.column}` | {row.r_with_size:+.3f} | {row.size_tier} |"
        for row in scored.itertuples()
    ]
    return header + "\n".join(lines)


def _lq_descriptions() -> dict[str, str]:
    """Build Source B's 60 per-sector column descriptions from the NAICS labels.

    Returns:
        Mapping of column name to description.
    """
    descriptions: dict[str, str] = {}
    for code in NAICS2_CODES:
        label = NAICS2_LABELS[code]
        descriptions[f"lq_emp_{code}"] = (
            f"Location quotient for {label} (NAICS {code}) -- the county's share of "
            "private employment in this sector over the national share. 1.0 is the "
            "national average. Null means unknown, never zero."
        )
        descriptions[f"emp_{code}"] = (
            f"Private employment level in {label} (NAICS {code}). A level, not a "
            "composition, so it is held in the size control rather than shipped."
        )
        descriptions[f"disclosure_{code}"] = (
            f"True where BLS suppressed the {label} cell for employer privacy. Null "
            "where the sector has no county row at all -- a third state, distinct "
            "from suppressed and from disclosed."
        )
    return descriptions


SOURCE_B = PillarSpec(
    pillar="B",
    parquet="source_b_qcew.parquet",
    doc_name="source_b_feature_schema.md",
    descriptions={
        "county_name": 'e.g. `"Harris County, Texas"`, from `county_crosswalk.parquet`.',
        "fips_code": "5-digit state+county FIPS. Join key.",
        **_lq_descriptions(),
        "emp_total_private": (
            "Total private employment, the denominator every location quotient is "
            "built on. A pure scale measure, held in the size control."
        ),
        "as_of_date": "`2025-12-31`, the end of the 2025 Q4 reference quarter.",
    },
    status_overrides={},
    body="""# Source B — Frozen Feature Schema

> Generated by `scripts/export_pillar_schema.py` from
> `data/source_b_qcew.parquet`. Do not edit the tables by hand: the coverage
> figures are read from the parquet, so regenerating is how they stay true.
> Machine-readable copy in `outputs/source_b_feature_schema.csv`.

Keyed on `fips_code`, {row_count:,} rows. **{ships_count} columns ship**: the
20-dimensional location-quotient vector, plus the 20 `disclosure_*` flags that
give it its null semantics. The flags are features rather than metadata —
whether BLS could disclose a sector is itself informative about county size and
sector structure, which is why the matrix's imputer carries a missingness
indicator rather than filling the gap away.

The 20 employment levels underneath the quotients, and the private-employment
total, are held in `pillar_matrix.SIZE_COLUMNS`: a county's employment in a
sector is a level, and the location quotient built from it is the composition
this pillar exists to carry.

Shipping the vector rather than a scalar is what rescued this pillar. The
strongest cross-pillar link surviving the size control anywhere in the sweep is
Source B's Real Estate & Rental & Leasing LQ against Source E's capital-to-wage
ratio, r = 0.394 raw / 0.382 size-controlled — roughly five times anything else
that survives (`docs/PROJECT_GOAL.md`).

## Null semantics

**Null means unknown, never zero.** This is the pillar where that distinction
costs the most, because BLS reports a suppressed cell's location quotient as a
literal `0` rather than leaving it blank. An early draft of `ingest_source_b.py`
trusted that value and silently reintroduced false zeros; `lq_emp_{{code}}` is now
explicitly nulled wherever the matching `disclosure_{{code}}` flag is `True`
(`source-b-findings.md` §1, finding 2).

A county × sector cell is in exactly one of three states, and the pair of
columns is what tells them apart:

| `disclosure_*` | `lq_emp_*` | meaning | cells |
|---|---|---|---|
| `True` | null | BLS suppressed it for employer privacy | {suppressed_cells:,} ({suppressed_share:.1%}) |
| null | null | the sector has no county row at all | {absent_cells:,} ({absent_share:.1%}) |
| `False` | a number | disclosed | {disclosed_cells:,} ({disclosed_share:.1%}) |

Both null states are genuinely unknown to a downstream model, which is why
**{total_null_share:.1%} of the LQ matrix arrives null** — more than the 30.0%
suppression rate `source-b-findings.md` §3.3 quotes, which is suppression among
the cells BLS reports at all
({suppressed_cells:,} / ({suppressed_cells:,} + {disclosed_cells:,})). Quote the
30.0% for what BLS withholds and the {total_null_share:.1%} for what a consumer
has to handle; they answer different questions. Imputation is the consumer's
decision and is deliberately not baked in: Phase 1b tested a state-level LQ
fallback (MAE = 0.786, r = 0.334 against held-out disclosed cells) and a
proportional-allocation proxy for the spec's proposed IPF matrix completion
(MAE = 0.786, r = 0.340), and neither meaningfully beat a global-mean baseline
(MAE = 0.947). A number that reads as more precise than it is would be worse
than the null.

**10 counties have no disclosed or present sector at all** — every one of their
20 cells is null. They are Mineral County, CO; Banner and Hayes Counties, NE;
Harding County, NM; Slope County, ND; Wheeler County, OR; Ziebach County, SD;
Daggett and Piute Counties, UT; and Menominee County, WI, all sparsely
populated, consistent with suppressed cells having a median of 5 establishments
against 40 for disclosed ones.

**1 county has no Source B row at all**: Kalawao County, HI (`15005`, population
~90), which has no QCEW private-sector county row of any kind — absent rather
than suppressed. Source E has the same gap independently. Downstream models need
an explicit policy for it.

## Scope

**Private ownership only** (`own_code` 5), so government employment is outside
every location quotient here. That is a deliberate scoping decision rather than
an oversight, and it matters for reading Source F: USDA's `high_government`
typology flag has no counterpart in this pillar.

Suppression is not random — it targets small-establishment counties — so any
per-sector statistic computed over the disclosed subset rests on a size-biased
sample. `analyze_source_b_source_c_correlation.py` applies an n ≥ 100 floor,
which reduces the bias without removing it.

## Columns

{columns_table}

## How much of each shipping column is county size

Pearson r against `log10(Census population)` from
`outputs/feature_size_dependence.csv`: tier 1 at ≥0.30, tier 2 at ≥0.15, tier 3
below. **The downstream target is rate-shaped**, so county size is a control and
tier 3 is what transfers cleanly (`docs/downstream_target.md` Part 1).

Read the sparse sectors' correlations with the suppression bias above in mind:
`lq_emp_11` (Agriculture) is scored on the 1,457 counties where agriculture is
large enough to disclose, and those counties are systematically rural.

{size_table}

Tier 1 is not a cut list. A size-loaded column is scored on marginal lift over a
`target ~ log_population + density` baseline and kept if it beats it.

## Related

- `analysis-output/source-b/source-b-findings.md` — ingestion, the suppression
  handling, and the dominant-sector cross-checks.
- `docs/plans/ingestion_recon.md` § Source B, Phase 1b — the full imputation
  comparison behind the null-passthrough decision.
- `outputs/pillar_vintages.csv` — `as_of_date` for all six pillars.
""",
)


SOURCE_C = PillarSpec(
    pillar="C",
    parquet="source_c_fred.parquet",
    doc_name="source_c_feature_schema.md",
    descriptions={
        "county_name": 'e.g. `"Cook County, Illinois"`, from `county_crosswalk.parquet`.',
        "fips_code": "5-digit state+county FIPS. Join key.",
        "unemployment_velocity": (
            "3-year slope of the county's annual unemployment rate, in percentage "
            "points per year. Positive means unemployment is rising."
        ),
        "unemployment_rate_latest": "Most recent annual unemployment rate, in percent.",
        "unemployment_latest_year": (
            "Year the unemployment reading comes from. Bookkeeping, not a feature."
        ),
        "gdp_velocity": (
            "3-year slope of real GDP, in chained dollars per year. Confounded with "
            "economy size — prefer `gdp_velocity_pct`. Still inside the matrix's "
            "Source C block; see the note above."
        ),
        "gdp_latest": (
            "Most recent real GDP level. A pure scale measure, held in the size "
            "control and re-exposed as `log_gdp_latest`."
        ),
        "gdp_latest_year": "Year the GDP reading comes from. Bookkeeping, not a feature.",
        "gdp_velocity_pct": (
            "`gdp_velocity / gdp_latest` — GDP growth as a rate rather than a dollar "
            "amount. **The column a downstream model should prefer.**"
        ),
        "as_of_date": (
            "`2025-12-31`. Unemployment runs through 2025, real GDP through 2024; the "
            "per-row years are in `unemployment_latest_year` and `gdp_latest_year`."
        ),
    },
    status_overrides={},
    body="""# Source C — Frozen Feature Schema

> Generated by `scripts/export_pillar_schema.py` from
> `data/source_c_fred.parquet`. Do not edit the tables by hand: the coverage
> figures are read from the parquet, so regenerating is how they stay true.
> Machine-readable copy in `outputs/source_c_feature_schema.csv`.

Keyed on `fips_code`, {row_count:,} rows. **{ships_count} columns ship.** Each
velocity is a rolling 3-year first derivative of an annual FRED series, so this
pillar measures momentum rather than level — a distinct thing from every other
pillar here, and the reason Source F's structural flags do not track it
(r = 0.007 and −0.019 against the two velocities, `source-f-findings.md` §3.4).

## Prefer the normalized velocity

`gdp_velocity` is denominated in chained dollars per year, so a large county
posts a large slope for being large. `gdp_velocity_pct` divides it by the
county's own GDP level and is the column to use. Against
`log10(Census population)`, the dollar column runs r = +0.420 — tier 1, size in
disguise — while the normalized one sits at +0.101, inside the size-free tier.

**`gdp_velocity` is nevertheless still inside `pillar_matrix`'s Source C
block**, not held out in `SIZE_COLUMNS` alongside `gdp_latest`. Stated here
because a schema document that quietly showed only the preferred column would
misdescribe what a consumer of the matrix actually receives. `docs/PROJECT_GOAL.md`
records the metric fix as done, and it is done in the sense that every reported
result uses `gdp_velocity_pct`; the dollar column's continued membership in the
block is an open item rather than a decision.

## Null semantics

**Nulls are coverage gaps, not zeros.** A county with no FRED series for a
metric gets null for every column derived from it; there is no sentinel.

| coverage | counties |
|---|---|
| both series | 3,080 (98.0%) |
| unemployment only | 63 (2.0%) |
| neither | 1 |

The one county with neither is Kalawao County, HI (`15005`, population ~90) —
the same county Sources B and E are missing.

The 64 counties without a GDP series split into two unrelated upstream causes
(`source-c-findings.md` §3.2), neither of which this repo backfills:

- **Virginia independent cities, 51 of 64.** Virginia's independent-city
  structure falls below whatever population threshold BEA uses to publish
  county-level GDP for some — not all — of these cities.
- **Connecticut, 9 of 64.** All nine Planning Regions. Connecticut dissolved its
  counties as administrative units in 2022 and BEA/FRED has not backfilled GDP
  under the new geography. Source F hits the identical gap independently, which
  is how the repo knows it is an upstream data lag rather than an ingestion bug.

Every Connecticut geography is missing GDP; none is missing unemployment.

## Columns

{columns_table}

## How much of each shipping column is county size

Pearson r against `log10(Census population)` from
`outputs/feature_size_dependence.csv`: tier 1 at ≥0.30, tier 2 at ≥0.15, tier 3
below. **The downstream target is rate-shaped**, so county size is a control and
tier 3 is what transfers cleanly (`docs/downstream_target.md` Part 1).

{size_table}

The two velocities and the unemployment level are among the cleanest columns in
the whole matrix by this measure. The exception is the one the pillar already
recommends against using.

## Related

- `analysis-output/source-c/source-c-findings.md` — ingestion, the GDP coverage
  decomposition, and the size-normalization argument.
- `outputs/source_c_gdp_coverage.csv` — the 64 counties, by cause.
- `outputs/pillar_vintages.csv` — `as_of_date` for all six pillars.
""",
)


def _source_d_descriptions() -> dict[str, str]:
    """Build Source D's per-commodity column descriptions.

    Returns:
        Mapping of column name to description.
    """
    descriptions: dict[str, str] = {}
    for group in SCTG_GROUPS:
        label = SCTG_LABELS[group]
        for direction, word in (("out", "Outbound"), ("in", "Inbound")):
            descriptions[f"{direction}_{group}"] = (
                f"{word} tons in {label}. A level wearing a commodity label — held in "
                "the size control since 2026-08-05."
            )
            descriptions[f"share_{direction}_{group}"] = (
                f"{word} tons in {label} as a share of the county's own {word.lower()} "
                "total. The composition the raw tonnage was standing in for."
            )
    return descriptions


SOURCE_D = PillarSpec(
    pillar="D",
    parquet="source_d_faf.parquet",
    doc_name="source_d_feature_schema.md",
    descriptions={
        "county_name": 'e.g. `"Harris County, Texas"`, from `county_crosswalk.parquet`.',
        "fips_code": "5-digit state+county FIPS. Join key.",
        "total_outbound_tons": (
            "Total domestic outbound tonnage, 2022. Superseded in the matrix by "
            "`log_outbound_tons`; spans six orders of magnitude on the raw scale."
        ),
        "total_inbound_tons": (
            "Total domestic inbound tonnage, 2022. Superseded in the matrix by "
            "`log_inbound_tons`."
        ),
        "out_partner_hhi": (
            "Herfindahl-Hirschman concentration of outbound tonnage across partners, "
            "pooling county-level and FAF-zone-level partner rows into one "
            "distribution. Higher means flow funnels through fewer corridors."
        ),
        "in_partner_hhi": (
            "Herfindahl-Hirschman concentration of inbound tonnage across partners, "
            "same construction as `out_partner_hhi`."
        ),
        **_source_d_descriptions(),
        "log_total_tons": (
            "`log10(outbound + inbound tons)`, clipped at 1. Derived in "
            "`pillar_matrix`. The pillar's headline volume measure."
        ),
        "log_outbound_tons": "`log10(total_outbound_tons)`, clipped at 1. Derived in `pillar_matrix`.",
        "log_inbound_tons": "`log10(total_inbound_tons)`, clipped at 1. Derived in `pillar_matrix`.",
        "as_of_date": "`2022-12-31`. The oldest pillar in the matrix; FAF5 refreshes about every 5 years.",
    },
    status_overrides={
        "total_outbound_tons": "superseded by `log_outbound_tons`",
        "total_inbound_tons": "superseded by `log_inbound_tons`",
    },
    body="""# Source D — Frozen Feature Schema

> Generated by `scripts/export_pillar_schema.py` from
> `data/source_d_faf.parquet` plus the columns `pillar_matrix` derives from it.
> Do not edit the tables by hand. Machine-readable copy in
> `outputs/source_d_feature_schema.csv`.

Keyed on `fips_code`, {row_count:,} rows, **zero nulls anywhere in the file**.
**{ships_count} columns ship**: three log tonnage measures, two partner
concentration indices, and the ten commodity shares.

**Ten columns in the parquet do not ship.** The raw per-commodity tonnages
(`out_sctg*`, `in_sctg*`) ran 0.52–0.97 Spearman against population — they are
levels wearing a commodity label — and moved into `pillar_matrix.SIZE_COLUMNS`
on 2026-08-05 once the downstream target was confirmed rate-shaped. Removing
them cost nothing measurable: matrix-sweep mean lift moved +0.0847 → +0.0851.

## What the shares bought

The ten `share_*` columns are the composition those levels were standing in for,
and they are what surfaced the freight-to-industry link the original proposal
claimed and round 1 could not show: Agriculture LQ moved from
indistinguishable-from-zero to +0.0430 ablated, Manufacturing LQ +0.067 → +0.107
(`source-d-findings.md` §11–§12). Five of the ten fall below the tier-1
threshold where all ten raw tonnages sat, and two are outright size-free — see
the tier table below.

One correction to that findings section while quoting it: its §11 prose calls
those five "size-free," but on the tiering this repo uses, two of the ten are
size-free (|r| < 0.15) and three are partly size. Five clear tier 1, which is
the real and still substantial claim.

**There is no `tons_per_capita` column, deliberately.** It is algebraically
identical to `log_total_tons − log_population`, and both of those already sit in
the matrix, so it adds nothing any size-controlling model cannot already reach.
Measured rather than assumed: D freight against F metro status is −0.036
size-controlled whether the input is raw log tonnage or per-capita tonnage, to
three decimals (`source-d-findings.md` §10). The argument in full is in the
comment block in `scripts/pillar_matrix.py`. It would matter for a consumer
fitting *without* a size control, which is a serving-format question.

## Null semantics

**Nothing in this file is null.** Every crosswalk county has a row, every
county has non-zero outbound and inbound totals, so every share has a defined
denominator (`source-d-findings.md` §11). Source D is the only pillar with no
null policy to state, which is worth saying explicitly so a consumer does not go
looking for one.

The zero-tonnage floor is the one thing to know: the three `log_*` columns clip
at 1 ton before taking the log, so a hypothetical zero-flow county would read as
0.0 rather than negative infinity. No county currently hits the clip.

## Construction

BTS ships the FAF5 Experimental County-Level Estimates as per-state zip files of
origin-destination tables at mixed county and FAF-zone granularity. This pillar
uses domestic flows only (`trade_type = 1`); import and export legs involve a
foreign region rather than a second US county. Both HHIs pool the county-level
and zone-level partner rows into a single distribution per county per direction,
and both were re-derived at market grain on 2026-08-07 so the aggregation
analysis no longer approximates them (`external-target-findings.md` §21).

Two caveats a consumer should carry:

- **The county estimates are BTS's disaggregation, not a direct measurement.**
  Its gravity model assigns near-universal nonzero flow to almost every county
  pair, which is why raw partner *count* was tested and dropped as a feature — it
  does not distinguish a hub from a sink at all (`source-d-findings.md` §2).
- **Partner concentration reverses direction with scale.** A two-state design
  spike found hubs *less* concentrated than rural counties; nationally the
  opposite holds (r = 0.278 between log tonnage and mean HHI), because a
  regional sample cannot see the long-haul corridors that define a real hub
  (§3.2). Anyone who saw the regional result should not carry its sign forward.

## Columns

{columns_table}

## How much of each shipping column is county size

Pearson r against `log10(Census population)` from
`outputs/feature_size_dependence.csv`: tier 1 at ≥0.30, tier 2 at ≥0.15, tier 3
below. **The downstream target is rate-shaped**, so county size is a control and
tier 3 is what transfers cleanly (`docs/downstream_target.md` Part 1).

{size_table}

The three log tonnage columns are tier 1 by construction and ship anyway: freight
volume is the whole of this pillar's signal, and folding it into the size control
would decide Source D's verdict by construction rather than by evidence
(`pillar_matrix.py` module docstring).

## Related

- `analysis-output/source-d/source-d-findings.md` — §9 on every column being a
  level, §10 on the per-capita non-fix, §11–§12 on what the shares bought.
- `data/source_d_partners.parquet` — the partner-tons distribution, shipped so
  both HHIs can be re-derived at any geography.
- `outputs/pillar_vintages.csv` — `as_of_date` for all six pillars.
""",
)


SOURCE_F = PillarSpec(
    pillar="F",
    parquet="source_f_usda_typology.parquet",
    doc_name="source_f_feature_schema.md",
    descriptions={
        "county_name": 'e.g. `"Menominee County, Wisconsin"`, from `county_crosswalk.parquet`.',
        "fips_code": "5-digit state+county FIPS. Join key.",
        "metro_2023": "True where OMB's 2023 delineation puts the county in a metropolitan area.",
        "high_farming": "High concentration of farming earnings or employment.",
        "high_mining": "High concentration of mining earnings or employment.",
        "high_manufacturing": "High concentration of manufacturing earnings or employment.",
        "high_government": (
            "High concentration of government earnings or employment. Note that "
            "Source B's location quotients are private-ownership only, so this flag "
            "has no QCEW counterpart."
        ),
        "high_recreation": "High concentration of recreation earnings or employment.",
        "nonspecialized": "No sector concentrated enough to earn a dependence label.",
        "low_postsecondary_ed": "Demographic risk flag: low share of adults with postsecondary education.",
        "low_employment": "Demographic risk flag: low employment rate among working-age adults.",
        "population_loss": "Demographic risk flag: sustained population decline.",
        "housing_stress": "Demographic risk flag: high share of cost-burdened households.",
        "retirement_destination": (
            "High net in-migration of people aged 60 and over. Restates age structure "
            "by definition, so it is ablated when the target is `median_age`."
        ),
        "persistent_poverty": "Demographic risk flag: poverty above threshold across successive censuses.",
        "industry_dependence_none": "One-hot: no dominant industry. The modal county, at 50.0%.",
        "industry_dependence_farming": "One-hot: farming-dependent.",
        "industry_dependence_mining": "One-hot: mining-dependent.",
        "industry_dependence_manufacturing": "One-hot: manufacturing-dependent.",
        "industry_dependence_government": "One-hot: government-dependent.",
        "industry_dependence_recreation": "One-hot: recreation-dependent.",
        "distress_count": (
            "Sum of the six demographic risk flags, 0–6. Derived in `pillar_matrix`. "
            "Observed maximum is 5; no county carries all six."
        ),
        "as_of_date": (
            "`2025-12-31`, the publication edition rather than a period end — the "
            "codes are built from several upstream series with different windows."
        ),
    },
    status_overrides={},
    body="""# Source F — Frozen Feature Schema

> Generated by `scripts/export_pillar_schema.py` from
> `data/source_f_usda_typology.parquet` plus `distress_count`, derived in
> `pillar_matrix`. Do not edit the tables by hand. Machine-readable copy in
> `outputs/source_f_feature_schema.csv`.

Keyed on `fips_code`, {row_count:,} rows — complete crosswalk coverage, the only
pillar with no missing county. **{ships_count} columns ship.**

**This pillar's slot was contested, and is now settled.** Source F fails the
pairwise hub test: its correlation against Source D freight tonnage, r = 0.495
and the largest raw effect in the 15-pillar-pair sweep, collapses to r = −0.057
once county size is controlled. On the drop-one test it is the **second most
valuable of the six pillars** — withholding this block from a model that already
holds county size and the other five costs +0.0413 mean R² across five external
ACS targets, positive on 5 of 5 and above the shuffled-feature noise floor on
5 of 5.

Both are true and they travel together: the pairwise test was the wrong
instrument for a categorical structural variable, not a wrong number. See
`analysis-output/cross-source/pillar-marginal-findings.md`, and carry the
ablation caveat in its §5 alongside any *internal* figure for this pillar —
seven eighths of F's apparent lift against other pillars' columns is USDA
restating industry composition BLS already measures.

## Null semantics

**Null means "USDA did not classify this county," never `False`.** ERS publishes
sentinel codes — `99` for not classified, `-1` for insufficient data — and
`ingest_source_f.py` maps both to null rather than to `False`, so "no signal" is
never silently conflated with "confirmed absent." This is the pillar's single
most important consumer-facing fact: every flag here is a nullable boolean, not
a plain one.

The 9 counties carrying the not-classified sentinel for industry dependence are
all Connecticut Planning Regions — the same 2022 county abolition that leaves
Source C without GDP for those geographies. Two unrelated federal providers
lagging on the same boundary change is what makes it an upstream issue rather
than an ingestion bug.

`persistent_poverty` (33 nulls) and `population_loss` (17) carry their own
sentinel gaps independent of Connecticut.

**`distress_count` sums with `skipna=True`**, so a county missing two of six
flags is scored on the remaining four rather than penalized or excluded. That is
a slight undercount for the handful of counties affected, documented rather than
corrected (`source-f-findings.md` §5).

## The one-hot block is mutually exclusive; half of it is "none"

The six `industry_dependence_*` columns are mutually exclusive by construction —
verified, every county's six flags sum to exactly 0 or 1, never more. The
distribution is more concentrated than the proposal's five-category framing
implies:

| category | counties | share |
|---|---|---|
| None (nonspecialized) | 1,572 | 50.0% |
| Manufacturing | 706 | 22.5% |
| Farming | 354 | 11.3% |
| Recreation | 267 | 8.5% |
| Government | 146 | 4.6% |
| Mining | 90 | 2.9% |
| Not classified (sentinel) | 9 | 0.3% |

**The modal county contributes no dependence signal at all.** Anyone using this
as a categorical feature should know that before one-hot encoding it again.

## Ablated in cross-pillar work by construction

USDA builds its industry-dependence flags from industry employment and earnings
shares — the same underlying quantity QCEW's location quotients measure. So
`high_manufacturing` predicting the manufacturing LQ is two federal products
restating one fact, not two agencies corroborating each other.

Twelve columns are therefore held out of every cross-pillar test, as
`RESTATEMENT_COLUMNS` in `analyze_pillar_matrix_signal.py`: the six `high_*`
flags and the six `industry_dependence_*` one-hots. The distress flags,
`metro_2023` and `distress_count` are not industry measures and stay in.

The relationship runs the other way too: Source A's `has_metro_attachment` fires
when a Wikipedia intro states the county belongs to a metropolitan statistical
area, which is the OMB delineation `metro_2023` reports directly. Agreement
between those two is bookkeeping.

**This ablation is correct against another pillar and wrong against an external
target.** A downstream model predicting churn may legitimately use every column
here; the restatement rule exists to stop this repo from crediting a pillar for
reciting a label it copied.

## Columns

{columns_table}

## How much of each shipping column is county size

Pearson r against `log10(Census population)` from
`outputs/feature_size_dependence.csv`: tier 1 at ≥0.30, tier 2 at ≥0.15, tier 3
below. **The downstream target is rate-shaped**, so county size is a control and
tier 3 is what transfers cleanly (`docs/downstream_target.md` Part 1).

{size_table}

`metro_2023` is the tier-1 case that matters: at r = +0.592 it is largely a
restatement of county size, and `docs/downstream_target.md` recommends demoting
it along with `population_loss` and `housing_stress`.

## Related

- `analysis-output/cross-source/pillar-marginal-findings.md` — the drop-one test
  that settled this pillar's slot, and its pre-registered decision rule.
- `docs/pillar_status.md` — per-pillar verdicts, including this one.
- `analysis-output/source-f/source-f-findings.md` — the typology breakdown, the
  Connecticut sentinel, and the null result against Source C's velocities.
- `outputs/pillar_vintages.csv` — `as_of_date` for all six pillars.
""",
)


SPECS: tuple[PillarSpec, ...] = (SOURCE_B, SOURCE_C, SOURCE_D, SOURCE_F)


def pillar_frame(spec: PillarSpec) -> pd.DataFrame:
    """Load one pillar's parquet joined with the columns `pillar_matrix` derives.

    The parquet alone understates what a consumer of the matrix receives (Source
    D's shares, Source F's `distress_count`), and the matrix alone understates
    what the parquet holds (every `SIZE_COLUMNS` member). The union is the
    contract.

    Args:
        spec: The pillar being documented.

    Returns:
        DataFrame carrying every column, raw and derived, minus `fips_code`
        duplication.

    Raises:
        FileNotFoundError: If the pillar parquet is absent.
    """
    raw = pd.read_parquet(DATA_DIR / spec.parquet)
    derived = derived_pillar_frames()[spec.pillar]
    extra = [col for col in derived.columns if col not in raw.columns]
    if not extra:
        return raw
    return raw.merge(derived[["fips_code", *extra]], on="fips_code", how="left")


def source_b_null_facts(frame: pd.DataFrame) -> dict[str, object]:
    """Count the three states a Source B county x sector cell can be in.

    The suppression rate quoted in `source-b-findings.md` is suppression among
    cells BLS reports at all, which is not the share of the LQ matrix that
    arrives null. Both are computed here so the document can state each
    correctly.

    Args:
        frame: Source B's frame from `pillar_frame`.

    Returns:
        Cell counts and shares for the suppressed, absent and disclosed states.
    """
    total = len(frame) * len(NAICS2_CODES)
    suppressed = sum(int((frame[f"disclosure_{code}"] == True).sum()) for code in NAICS2_CODES)  # noqa: E712
    absent = sum(int(frame[f"disclosure_{code}"].isna().sum()) for code in NAICS2_CODES)
    disclosed = total - suppressed - absent
    return {
        "suppressed_cells": suppressed,
        "suppressed_share": suppressed / total,
        "absent_cells": absent,
        "absent_share": absent / total,
        "disclosed_cells": disclosed,
        "disclosed_share": disclosed / total,
        "total_null_share": (suppressed + absent) / total,
    }


def render(spec: PillarSpec, schema: pd.DataFrame, frame: pd.DataFrame) -> str:
    """Fill a pillar's markdown template with the generated tables.

    Args:
        spec: The pillar being documented.
        schema: Output of `build_schema`.
        frame: The frame the schema describes.

    Returns:
        Markdown document text.
    """
    fields: dict[str, object] = {
        "columns_table": columns_table(schema),
        "size_table": size_table(schema),
        "row_count": len(frame),
        "ships_count": int((schema["status"] == "ships").sum()),
    }
    if spec.pillar == "B":
        fields.update(source_b_null_facts(frame))
    return spec.body.format(**fields)


def main() -> None:
    """Write a schema CSV and handoff document for each of B, C, D and F."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    _, blocks = build_matrix()
    size_dependence = load_size_dependence()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        frame = pillar_frame(spec)
        schema = build_schema(spec, frame, blocks, size_dependence)

        csv_path = OUTPUTS_DIR / f"source_{spec.pillar.lower()}_feature_schema.csv"
        schema.to_csv(csv_path, index=False)
        (DOCS_DIR / spec.doc_name).write_text(render(spec, schema, frame))

        logger.info(
            "Source %s: %d columns (%d ship) -> %s, %s",
            spec.pillar,
            len(schema),
            int((schema["status"] == "ships").sum()),
            csv_path.name,
            spec.doc_name,
        )


if __name__ == "__main__":
    main()
