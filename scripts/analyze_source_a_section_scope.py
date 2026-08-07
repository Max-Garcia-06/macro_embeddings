"""Test whether reading more of the article -- uniformly -- beats reading the economy section.

`extract_source_a_section_features.py` reads exactly one thing beyond the lead:
sections whose title marks them as economic. That whitelist was a deliberate
precision choice, never a measured one. This module measures it.

Three scopes, same lexicon, same 28 targets, same protocol as
`analyze_source_a_representation.py` (unpenalized size-plus-state baseline,
ridge on its residual with a nested penalty search, 5 folds, seed 42):

- `economy` -- the shipped whitelist, which reproduces `extracted_sections`.
- `economy_plus` -- widened by title to sections that name economic activity
  without narrating it: transportation, infrastructure, government, energy,
  tourism, military. History is deliberately excluded; see below.
- `all_sections` -- every non-reference section in the article.

**Why scope is widened uniformly rather than per tier.** The obvious version of
this experiment reads deeper only for counties whose lead says little, since that
is where the headroom is. It is the wrong design. Tier membership tracks county
size (`content_length` r = 0.359, `n_body_sections` r = 0.550), so a
tier-conditional rule makes `has_agriculture` mean "named in the lead or economy
section" for one county and "named anywhere" for another, and the difference
between those two meanings is correlated with population. That is a size proxy
manufactured inside the feature. Scope stays uniform; sparsity keeps encoding the
tier, exactly as it does today.

**Coverage is not the result.** Reading every section roughly triples the share
of counties with any industry flag, which is precisely the kind of number that
has misled this pillar before -- §13.1's industry base rate was inflated to 19.7%
against a true 6.5% by a case-sensitivity and word-boundary bug, and two lexicon
flags were matching city names and 1830s history until a precision check caught
them. A History section saying agriculture was the mainstay until 1910 is a false
positive for a feature meant to describe the current economy. So this module
reports lift, and writes a sampled-precision file for the hits each widening
adds, flagged for historical framing.

Run after `ingest_source_a.py`, `extract_source_a_features.py`, and
`extract_source_a_section_features.py`. Read-only with respect to the shipped
parquets: nothing here rewrites a feature file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import N_FOLDS, RANDOM_SEED, build_baseline_design
from analyze_source_a_representation import (
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from analyze_source_a_tiers import TIER_LABELS, assign_tiers
from extract_source_a_features import INDUSTRY_LEXICON, VARIANT_COLUMNS
from extract_source_a_section_features import (
    ECONOMY_TITLE_PATTERN,
    SECTION_PREFIX,
    SECTIONS_PARQUET_PATH,
    section_feature_columns,
)
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_section_scope.csv"
OUTPUT_PRECISION_PATH: Path = OUTPUTS_DIR / "source_a_section_scope_precision.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_section_scope_stats.json"

# Titles added to the economy whitelist. Every one names an activity rather than
# a narrative: a Transportation section lists the rail and port infrastructure a
# logistics flag is trying to detect. "History" is the conspicuous omission --
# it is where a defunct industry is most likely to be described in the past
# tense, and the flags are meant to describe the economy as it is.
ECONOMY_PLUS_EXTRA_PATTERN: str = (
    r"^(?:transportation|transport|infrastructure|government|"
    r"government and infrastructure|energy|mining|oil and gas|tourism|"
    r"recreation|military|ports?|airports?|railroads?|utilities|"
    r"major highways|highways|industry and commerce|commerce|"
    r"economy and government|arts and culture)$"
)

# Reference-style sections carry link text and citation boilerplate rather than
# prose about the county, and `ingest_source_a.py` already excludes them from the
# sections parquet. Repeated here only as documentation of what "all" means.
ALL_SECTIONS_PATTERN: str = r".*"

# Everything except the sections that narrate rather than describe. This scope
# exists to separate two explanations of any gain `all_sections` shows: that
# reading more text helps, or that reading *history* helps. A lexicon hit inside
# a History section is usually a defunct industry -- "the South Bronx was a
# manufacturing center" -- and a feature built to describe the current economy
# should not be scoring on it, however well industrial history happens to predict
# present-day industry.
NARRATIVE_TITLE_PATTERN: str = (
    r"^(?:history|early history|notable people|notable natives|"
    r"notable residents|people|culture|in popular culture|etymology|name)$"
)

# Windows of characters kept either side of a lexicon match in the precision
# file, which is sized to show the sentence a reviewer needs without dumping the
# section.
SNIPPET_RADIUS: int = 90

# Markers that a match sits in historical framing. Deliberately crude: this
# flags rows for human review, it does not decide them. A four-digit year before
# 1990 counts, as do the past-tense and cessation markers that show up in
# sentences like "the mill closed in 1974".
HISTORICAL_YEAR_PATTERN: str = r"\b1[5-9]\d{2}\b"
HISTORICAL_PHRASE_PATTERN: str = (
    r"\b(?:was|were|had been|formerly|historically|once|until|"
    r"no longer|closed|ceased|abandoned|declined|defunct|"
    r"in the (?:early|late|mid)[- ]?(?:19|20)th century)\b"
)

# Counties sampled into the precision file per widening step.
PRECISION_SAMPLE_SIZE: int = 60

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scope:
    """One section-selection rule being scored.

    Attributes:
        key: Short identifier used in result columns and stats.
        label: Human-readable description for reports.
        pattern: Case-insensitive full-match regex over the section title.
    """

    key: str
    label: str
    pattern: str


SCOPES: tuple[Scope, ...] = (
    Scope("economy", "economy-titled sections (shipped)", ECONOMY_TITLE_PATTERN),
    Scope(
        "economy_plus",
        "economy + activity-named sections",
        f"(?:{ECONOMY_TITLE_PATTERN}|{ECONOMY_PLUS_EXTRA_PATTERN})",
    ),
    Scope("all_sections", "every non-reference section", ALL_SECTIONS_PATTERN),
    Scope("no_narrative", "every section except history and people", ALL_SECTIONS_PATTERN),
)

# Scopes defined by exclusion rather than by a title whitelist, since a
# full-match pattern cannot express "everything but these".
EXCLUDED_TITLES: dict[str, str] = {"no_narrative": NARRATIVE_TITLE_PATTERN}


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def select_sections(
    sections: pd.DataFrame, pattern: str, exclude_pattern: str | None = None
) -> pd.DataFrame:
    """Keep sections whose title matches a scope's pattern.

    Args:
        sections: Long-format section frame from `ingest_source_a.py`.
        pattern: Full-match regex, applied to the lowercased, stripped title.
        exclude_pattern: Optional full-match regex removed after `pattern`.

    Returns:
        Subset of `sections`.
    """
    titles = sections["section_title"].str.strip().str.lower()
    keep = titles.str.match(pattern, na=False)
    if exclude_pattern is not None:
        keep &= ~titles.str.match(exclude_pattern, na=False)
    return sections[keep]


def sections_for_scope(sections: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    """Apply a scope's include and exclude rules.

    Args:
        sections: Long-format section frame.
        scope: The scope to apply.

    Returns:
        Subset of `sections` this scope reads.
    """
    return select_sections(sections, scope.pattern, EXCLUDED_TITLES.get(scope.key))


def build_scope_features(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Recompute the section feature block from an arbitrary section subset.

    Mirrors `extract_source_a_section_features.build_section_features`, including
    its convention that a county with no matching section gets False and zero
    rather than null. `has_economy_section` keeps its shipped meaning -- economy
    -titled sections only -- so that widening the lexicon's reach does not
    silently redefine a column the feature store already documents.

    Args:
        frame: Matrix rows carrying `fips_code`.
        selected: Sections this scope reads.

    Returns:
        DataFrame aligned to `frame`'s index over `section_feature_columns()`.
    """
    text = selected.groupby("fips_code")["section_text"].agg(" ".join)
    aligned = frame["fips_code"].map(text).fillna("")

    features = pd.DataFrame(index=frame.index)
    for item in INDUSTRY_LEXICON:
        features[f"{SECTION_PREFIX}{item.column}"] = aligned.str.contains(
            item.pattern, regex=True, na=False
        )
    industry_columns = [f"{SECTION_PREFIX}{item.column}" for item in INDUSTRY_LEXICON]
    features[f"{SECTION_PREFIX}n_industry_mentions"] = (
        features[industry_columns].sum(axis=1).astype("int64")
    )
    features["has_economy_section"] = frame["has_economy_section"].to_numpy()
    return features[section_feature_columns()]


def score_scopes(
    matrix: pd.DataFrame,
    scope_features: dict[str, pd.DataFrame],
    targets: list,
) -> pd.DataFrame:
    """Score every scope against every target, sharing one baseline per target.

    The controls never change across scopes, so their out-of-fold predictions are
    computed once per target and reused -- which also guarantees the lifts are
    differences against an identical baseline rather than against three
    separately-fitted ones.

    Args:
        matrix: Feature matrix from `build_matrix`.
        scope_features: Scope key to its recomputed section block.
        targets: Targets to predict.

    Returns:
        One row per (target, scope).
    """
    baseline = build_baseline_design(matrix)
    lead_columns = [c for c in VARIANT_COLUMNS["extracted_full"]]
    records: list[dict[str, float | str | int]] = []

    for target in targets:
        rows = matrix[target.column].notna().to_numpy()
        y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
        base_design = baseline.loc[rows].to_numpy(dtype="float64")
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        baseline_predictions = _baseline_oof_predictions(base_design, y, folds)
        r2_baseline = float(r2_score(y, baseline_predictions))

        for scope in SCOPES:
            block = pd.concat(
                [matrix.loc[rows, lead_columns], scope_features[scope.key].loc[rows]],
                axis=1,
            ).to_numpy(dtype="float64")
            predictions = _residual_oof_predictions(base_design, block, y, folds, None)
            records.append(
                {
                    "pillar": target.pillar,
                    "column": target.column,
                    "label": target.label,
                    "scope": scope.key,
                    "n": int(rows.sum()),
                    "r2_baseline": r2_baseline,
                    "lift": float(r2_score(y, predictions)) - r2_baseline,
                }
            )
        logger.info("scored %s", target.column)

    return pd.DataFrame.from_records(records)


def coverage_by_tier(
    matrix: pd.DataFrame, scope_features: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Report industry-flag coverage per tier under each scope.

    Coverage counts a county as covered if any industry flag is set, whether it
    came from the lead or from a section.

    Args:
        matrix: Feature matrix carrying the lead-derived flags and `tier`.
        scope_features: Scope key to its recomputed section block.

    Returns:
        One row per (scope, tier).
    """
    lead_flags = [item.column for item in INDUSTRY_LEXICON]
    lead_hit = matrix[lead_flags].any(axis=1)
    records: list[dict[str, float | str | int]] = []

    for scope in SCOPES:
        section_flags = [f"{SECTION_PREFIX}{item.column}" for item in INDUSTRY_LEXICON]
        covered = lead_hit | scope_features[scope.key][section_flags].any(axis=1)
        for tier in TIER_LABELS:
            mask = matrix["tier"].to_numpy() == tier
            records.append(
                {
                    "scope": scope.key,
                    "tier": tier,
                    "n": int(mask.sum()),
                    "covered": int(covered[mask].sum()),
                    "coverage": float(covered[mask].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def sample_new_hits(
    matrix: pd.DataFrame, sections: pd.DataFrame, narrow: Scope, wide: Scope
) -> pd.DataFrame:
    """Sample lexicon matches that a widening adds, with their surrounding text.

    Every row is a (county, flag) pair the wider scope sets and the narrower one
    does not, shown as the snippet a reviewer needs to judge whether the match
    describes the county's economy or its history.

    Args:
        matrix: Feature matrix carrying `fips_code`, `county_name`, `tier`.
        sections: Long-format section frame.
        narrow: Scope the hit is absent from.
        wide: Scope the hit is present in.

    Returns:
        Sampled rows, up to `PRECISION_SAMPLE_SIZE`.
    """
    narrow_sections = sections_for_scope(sections, narrow)
    narrow_text = narrow_sections.groupby("fips_code")["section_text"].agg(" ".join)
    added = sections_for_scope(sections, wide)
    added = added[~added.index.isin(narrow_sections.index)]

    rows: list[dict[str, str]] = []
    for item in INDUSTRY_LEXICON:
        compiled = re.compile(item.pattern)
        already = narrow_text.str.contains(item.pattern, regex=True, na=False)
        already_fips = set(already[already].index)
        for fips, title, text in added[
            ["fips_code", "section_title", "section_text"]
        ].itertuples(index=False):
            if fips in already_fips:
                continue
            match = compiled.search(text)
            if match is None:
                continue
            start = max(0, match.start() - SNIPPET_RADIUS)
            snippet = text[start : match.end() + SNIPPET_RADIUS].replace("\n", " ")
            rows.append(
                {
                    "fips_code": fips,
                    "flag": item.column,
                    "section_title": title,
                    "matched": match.group(0),
                    "snippet": snippet,
                }
            )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return frame

    frame = frame.drop_duplicates(subset=["fips_code", "flag"])
    frame["historical_marker"] = frame["snippet"].str.contains(
        HISTORICAL_YEAR_PATTERN, regex=True
    ) | frame["snippet"].str.contains(HISTORICAL_PHRASE_PATTERN, regex=True, case=False)
    frame["widening"] = f"{narrow.key} -> {wide.key}"

    names = matrix.set_index("fips_code")[["county_name", "tier"]]
    frame = frame.join(names, on="fips_code")
    if len(frame) <= PRECISION_SAMPLE_SIZE:
        return frame
    return frame.sample(PRECISION_SAMPLE_SIZE, random_state=RANDOM_SEED)


def summarize(
    scores: pd.DataFrame, coverage: pd.DataFrame, precision: pd.DataFrame
) -> dict[str, object]:
    """Assemble the stats payload.

    Args:
        scores: Per-target, per-scope lifts.
        coverage: Per-scope, per-tier coverage.
        precision: Sampled new hits with historical-marker flags.

    Returns:
        JSON-serializable summary.
    """
    by_scope = scores.groupby("scope")["lift"]
    payload: dict[str, object] = {
        "n_targets": int(scores["column"].nunique()),
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "scopes": {
            scope.key: {
                "label": scope.label,
                "mean_lift": float(by_scope.mean()[scope.key]),
                "median_lift": float(by_scope.median()[scope.key]),
                "n_targets_better_than_economy": int(
                    (
                        scores[scores.scope.eq(scope.key)].set_index("column")["lift"]
                        > scores[scores.scope.eq("economy")].set_index("column")["lift"]
                    ).sum()
                ),
                "coverage_corpus": float(
                    coverage[coverage.scope.eq(scope.key)]["covered"].sum()
                    / coverage[coverage.scope.eq(scope.key)]["n"].sum()
                ),
            }
            for scope in SCOPES
        },
        "coverage_by_tier": coverage.to_dict(orient="records"),
    }
    if not precision.empty:
        payload["precision_sample"] = {
            widening: {
                "n_sampled": int(len(group)),
                "share_with_historical_marker": float(group["historical_marker"].mean()),
            }
            for widening, group in precision.groupby("widening")
        }
    return payload


def main() -> None:
    """Score every section scope and write results."""
    configure_logging()

    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix["tier"] = assign_tiers(matrix["content_length"])
    sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info("scoring %d targets across %d scopes", len(targets), len(SCOPES))

    scope_features = {
        scope.key: build_scope_features(matrix, sections_for_scope(sections, scope))
        for scope in SCOPES
    }

    scores = score_scopes(matrix, scope_features, targets)
    coverage = coverage_by_tier(matrix, scope_features)
    precision = pd.concat(
        [
            sample_new_hits(matrix, sections, SCOPES[0], SCOPES[1]),
            sample_new_hits(matrix, sections, SCOPES[1], SCOPES[3]),
            sample_new_hits(matrix, sections, SCOPES[3], SCOPES[2]),
        ],
        ignore_index=True,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_CSV_PATH, index=False)
    precision.to_csv(OUTPUT_PRECISION_PATH, index=False)
    stats = summarize(scores, coverage, precision)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")

    for scope in SCOPES:
        entry = stats["scopes"][scope.key]
        logger.info(
            "%-13s mean lift %+.5f | coverage %.1f%% | beats economy on %d targets",
            scope.key,
            entry["mean_lift"],
            100 * entry["coverage_corpus"],
            entry["n_targets_better_than_economy"],
        )


if __name__ == "__main__":
    main()
