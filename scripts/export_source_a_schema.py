"""Publish Source A's frozen feature schema for the feature-store handoff.

`docs/PROJECT_GOAL.md` sets the bar for this repo's output at production
feature-store quality: a stable schema keyed on `fips_code` with documented
coverage and null semantics. Source A's block went from one scalar to 29 shipping
columns in `source-a-findings.md` 13-14, and nothing outside that prose said what
each column means, how often it fires, or what a null denotes.

This script writes that contract from the data rather than by hand, so coverage
figures cannot drift away from the parquet they describe. Every column carries:

- its **status** -- `ships` for the 29 predictors, `diagnostic` for the two
  columns `pillar_matrix` deliberately withholds, and a note where a shipping
  column is ablated in pillar-versus-pillar work only;
- its **null policy**, which is uniform except for `founding_year`;
- its **size tier** under a rate-shaped target, read from
  `outputs/feature_size_dependence.csv`.

Null semantics for this block, stated once because it is the question a consumer
asks first: **absence is `False`, not null.** A county whose article never
mentions manufacturing gets `has_manufacturing = False`, the same value as a
county with no article content at all, because the sparsity pattern is itself the
content signal (`source-a-findings.md` 13.2). `founding_year` is the single
exception: it is genuinely null on the 1,930 counties whose intro states no
founding clause, since there is no numeric value that means "no founding year".

Outputs: `outputs/source_a_feature_schema.csv` and
`docs/source_a_feature_schema.md`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from extract_source_a_features import DIAGNOSTIC_COLUMNS, VARIANT_COLUMNS
from extract_source_a_section_features import (
    SECTION_DIAGNOSTIC_COLUMNS,
    section_output_columns,
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
DOCS_DIR: Path = REPO_ROOT / "docs"

TEXT_FEATURES_PATH: Path = DATA_DIR / "source_a_text_features.parquet"
SIZE_DEPENDENCE_PATH: Path = OUTPUTS_DIR / "feature_size_dependence.csv"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_feature_schema.csv"
OUTPUT_DOC_PATH: Path = DOCS_DIR / "source_a_feature_schema.md"

# Tier boundaries for |r| with county size, matching the tiering published in
# `docs/downstream_target.md` Part 2.
SIZE_TIER_ONE_THRESHOLD: float = 0.30
SIZE_TIER_TWO_THRESHOLD: float = 0.15

# One line per column, in the block's output order. Written by hand because a
# regex is not a definition -- a consumer needs to know what the flag is
# *evidence of*, not which alternation matched.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "content_length": "Characters of lead-section text after boilerplate stripping. The retired incumbent scalar, kept as one column among 29.",
    "has_manufacturing": "Lead names manufacturing, a factory, a mill, or an industrial park.",
    "has_mining": "Lead names mining, a mine, coal, or a quarry.",
    "has_oil_gas": "Lead names oil, petroleum, natural gas, a refinery, or an oilfield.",
    "has_agriculture": "Lead names agriculture, farming, farmland, ranching, crops, or livestock.",
    "has_tourism": "Lead names tourism, a resort, skiing, a casino, or vacationing.",
    "has_timber": "Lead names timber, logging, lumber, forestry, or a sawmill.",
    "has_logistics": "Lead names shipping, freight, a distribution center, or warehousing.",
    "has_university": "Lead names a university or college.",
    "has_military_base": "Lead names a military installation by installation noun (base, station, depot, arsenal, garrison). Tightened after a precision check: bare `Fort X` and `Army` matched city names and Civil War prose.",
    "has_protected_land": "Lead names a National Park, National Forest, National Monument, or Wildlife Refuge.",
    "has_tribal_land": "Lead names present-tense tribal land or sovereignty. Tightened after a precision check: bare `Indian` matched the American Indian Wars and reservations dissolved in the 1830s.",
    "has_river": "Lead names a River or Creek.",
    "has_interstate": "Lead names an Interstate, U.S. Route, or Highway.",
    "has_port": "Lead names a Port of, Harbor/Harbour, or seaport.",
    "n_industry_mentions": "Count of the seven industry flags that fired on the lead. 0 on 92% of counties.",
    "has_metro_attachment": "Lead states the county belongs to a metropolitan or micropolitan statistical area.",
    "has_namesake": "Lead contains 'named for' / 'named after' / 'named in honor of'.",
    "founding_year": "Year captured from a founding clause (created/formed/organized/founded/established/incorporated). Null where the lead states none.",
    "n_distinct_proper_nouns": "Count of distinct capitalized tokens excluding template boilerplate ('County', 'Census', state names). Measures named content rather than prose length.",
    "sec_has_manufacturing": "Economy section names manufacturing.",
    "sec_has_mining": "Economy section names mining.",
    "sec_has_oil_gas": "Economy section names oil or gas.",
    "sec_has_agriculture": "Economy section names agriculture.",
    "sec_has_tourism": "Economy section names tourism.",
    "sec_has_timber": "Economy section names timber.",
    "sec_has_logistics": "Economy section names logistics.",
    "sec_n_industry_mentions": "Count of the seven industry flags that fired on the economy section. Carries 97.6% of the section gain and is effectively size-independent.",
    "has_economy_section": "Article has a body section whose title marks it economic.",
    "n_body_sections": "Count of body sections in the article. DIAGNOSTIC -- a size proxy (r = 0.550 with county size) carrying 2.4% of the block's lift.",
    "has_usda_echo": "Lead restates USDA's own county classification. DIAGNOSTIC -- Source F's `distress_count` is built from those classifications, so this detector must never predict them.",
}

# Shipping columns that pillar-versus-pillar analysis ablates without removing
# them from the block. The distinction matters downstream: the ablation is
# correct against another pillar's feature and wrong against an external target
# (`source-a-findings.md` 17.3).
RESTATEMENT_NOTES: dict[str, str] = {
    "has_metro_attachment": (
        "Ablated in the cross-pillar sweeps as a restatement of Source F's "
        "`metro_2023`. Against an external target it is a legitimate free feature."
    ),
}

logger = logging.getLogger(__name__)


def _size_tier(abs_r: float) -> str:
    """Bucket a feature's correlation with county size into a reporting tier."""
    if abs_r >= SIZE_TIER_ONE_THRESHOLD:
        return "1 (size in disguise)"
    if abs_r >= SIZE_TIER_TWO_THRESHOLD:
        return "2 (partly size)"
    return "3 (size-free)"


def build_schema() -> pd.DataFrame:
    """Assemble one row per Source A column from the parquet it describes.

    Returns:
        DataFrame with columns `column`, `dtype`, `status`, `description`,
        `counties_firing`, `fire_rate`, `nulls`, `null_policy`, `size_tier`,
        `r_with_size`, `notes`.

    Raises:
        FileNotFoundError: If the Source A parquet is absent.
        ValueError: If the parquet is missing a column the schema documents.
    """
    features = pd.read_parquet(TEXT_FEATURES_PATH)
    size_dependence = (
        pd.read_csv(SIZE_DEPENDENCE_PATH).set_index("feature")["r_with_log_size"]
        if SIZE_DEPENDENCE_PATH.exists()
        else pd.Series(dtype="float64")
    )

    diagnostics = {*DIAGNOSTIC_COLUMNS, *SECTION_DIAGNOSTIC_COLUMNS}
    ordered = list(
        dict.fromkeys(
            [*VARIANT_COLUMNS["extracted_full"], *section_output_columns(), *DIAGNOSTIC_COLUMNS]
        )
    )
    missing = [column for column in ordered if column not in features.columns]
    if missing:
        raise ValueError(f"{TEXT_FEATURES_PATH.name} is missing: {missing}")

    rows: list[dict[str, object]] = []
    for column in ordered:
        values = features[column]
        firing = int((values.fillna(0) != 0).sum())
        is_null_bearing = column == "founding_year"
        r = float(size_dependence.get(column, float("nan")))
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "status": "diagnostic" if column in diagnostics else "ships",
                "description": COLUMN_DESCRIPTIONS[column],
                "counties_firing": firing,
                "fire_rate": round(firing / len(features), 4),
                "nulls": int(values.isna().sum()),
                "null_policy": (
                    "null means no founding clause in the lead; there is no numeric "
                    "sentinel for absence"
                    if is_null_bearing
                    else "never null; absence is False/0"
                ),
                "size_tier": _size_tier(abs(r)) if r == r else "unscanned",
                "r_with_size": round(r, 3) if r == r else None,
                "notes": RESTATEMENT_NOTES.get(column, ""),
            }
        )
    return pd.DataFrame(rows)


def render_markdown(schema: pd.DataFrame, county_count: int) -> str:
    """Render the schema as the handoff document.

    Args:
        schema: Output of `build_schema`.
        county_count: Rows in the Source A parquet.

    Returns:
        Markdown document text.
    """
    shipping = schema[schema["status"] == "ships"]
    diagnostic = schema[schema["status"] == "diagnostic"]

    def table(frame: pd.DataFrame) -> str:
        header = (
            "| column | dtype | fires on | nulls | size tier | description |\n"
            "|---|---|---|---|---|---|\n"
        )
        lines = [
            f"| `{row.column}` | {row.dtype} | {row.counties_firing:,} "
            f"({row.fire_rate:.1%}) | {row.nulls:,} | {row.size_tier} | "
            f"{row.description} |"
            for row in frame.itertuples()
        ]
        return header + "\n".join(lines)

    restatements = "\n".join(
        f"- `{row.column}` — {row.notes}"
        for row in schema.itertuples()
        if row.notes
    )

    return f"""# Source A — Frozen Feature Schema

> Generated by `scripts/export_source_a_schema.py` from
> `data/source_a_text_features.parquet`. Do not edit by hand: the coverage
> figures are read from the parquet, so regenerating is how they stay true.
> Machine-readable copy in `outputs/source_a_feature_schema.csv`.

Keyed on `fips_code`, {county_count:,} rows, one per US county and
county-equivalent. **{len(shipping)} columns ship**; {len(diagnostic)} are
diagnostics that `pillar_matrix` withholds from every consumer.

## Null semantics

**Absence is `False`, not null.** A county whose article never mentions
manufacturing carries `has_manufacturing = False`, the same value as a county
with almost no article text, because that sparsity pattern is itself the signal
this block was built around (`source-a-findings.md` §13.2). One uniform schema
covers all {county_count:,} counties, so there is no "not applicable" category to
handle.

`founding_year` is the sole exception. It is null on the counties whose lead
states no founding clause, because no year value can mean "no founding year".
Any imputation is the consuming model's decision.

`as_of_date` carries the scrape date, not a reference period — Wikipedia
describes no fixed window. See `outputs/pillar_vintages.csv` for how that
compares to the other five pillars.

## Shipping columns

{table(shipping)}

## Diagnostic columns — written, never shipped

These are excluded in `pillar_matrix` via `SOURCE_A_DIAGNOSTIC_COLUMNS`, so
every consumer of the matrix inherits the exclusion rather than re-deriving it.

{table(diagnostic)}

## Ablated in pillar-versus-pillar work only

{restatements}

## Size tiers

Tier is `|r|` with `log10(Census population)` from
`outputs/feature_size_dependence.csv`: tier 1 at ≥0.30, tier 2 at ≥0.15, tier 3
below. **The downstream target is rate-shaped** (Comcast FreeWheel Revenue
Science; one row is an impression, request, auction, household, or device —
asserted 2026-08-05, pending written confirmation). So county size is a control
and tier 3 is what transfers cleanly. See `docs/downstream_target.md` Part 1.

Tier 1 is not a cut list. Size-loaded columns are scored on marginal lift over a
`target ~ log_population + density` baseline and kept if they beat it — some
carry urbanicity, which is a real driver rather than a population artifact.

## Related

- `analysis-output/source-a/source-a-findings.md` §13–§17 — how these columns
  were built and what they are worth.
- `docs/downstream_target.md` — the answered rate-versus-count question, and how
  the size tiers should be read under it.
- `outputs/pillar_vintages.csv` — `as_of_date` for all six pillars.
"""


def main() -> None:
    """Write the schema CSV and the handoff document."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    schema = build_schema()
    county_count = len(pd.read_parquet(TEXT_FEATURES_PATH))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    schema.to_csv(OUTPUT_CSV_PATH, index=False)
    OUTPUT_DOC_PATH.write_text(render_markdown(schema, county_count))

    logger.info(
        "%d columns (%d ship, %d diagnostic) -> %s, %s",
        len(schema),
        int((schema["status"] == "ships").sum()),
        int((schema["status"] == "diagnostic").sum()),
        OUTPUT_CSV_PATH,
        OUTPUT_DOC_PATH,
    )


if __name__ == "__main__":
    main()
