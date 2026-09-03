"""Can Source A read a county's wages off its Wikipedia article?

Every Source A result so far has been scored against ACS targets. The IRS
county file (Source E) has never been used as a *target* for Source A, only as
a co-predictor, and it is the one economic outcome in this repo that does not
come from the Census: `wage_per_return_thousands` is average wage and salary
income per return that reported any, filed with the IRS. Two sources, no shared
collection instrument, so a relationship between them is not two views of one
survey.

This module asks two questions of that target, using the four content tiers
`analyze_source_a_tiers.py` already defines.

**1. The level.** Median wage per return rises monotonically across the tiers,
stub to rich. That gradient is the obvious finding and it is mostly not about
Wikipedia: article length tracks county population, population tracks wages,
and the tier ordering is largely the size ordering wearing a different label.
So every level result here is reported twice -- raw, and net of
`log_population` -- and the second number is the one that means anything.

**2. The change.** Wage *growth* over the 2018-2022 panel is the more
interesting target precisely because it is size-free: the tier means are flat,
so whatever the industry flags do to growth cannot be the size confound
returning under another name. This is where Source A's lexicon columns behave
like what they claim to be -- statements about a county's economy rather than
about how much someone wrote.

**Four controls the numbers here are reported against.**

- `log_population`, the headcount control. This is the honest baseline for the
  level question and the one the headline quotes.
- **State**, entered as fixed effects in the within-state variant of every
  feature effect. An oil-and-gas flag that only marks Texas and North Dakota
  counties would produce a national correlation that is really a two-state
  correlation; the within-state column says whether the effect survives
  comparing a county to its own neighbours.
- **Centroid latitude and longitude**, the control that erased Source A's
  measured value against the ACS basket (findings §22.2). It is repeated here
  because a wage result that dies under two float columns is a result about
  where a county is, not about what its article says.
- The full three-column `SIZE_FEATURES` baseline is scored and reported, and
  **its result is not usable as a control for this target**. `log_agi` and
  `log_population` are both in it, and their difference is log AGI per head --
  an income-per-person measure that is close to a restatement of wage per
  return. A block adding nothing over that baseline has been told nothing about
  itself. It is here because leaving it out would invite the objection, not
  because it settles anything.

Out-of-fold scoring uses the same harness as the rest of the repo: ridge with
median imputation, five `GroupKFold` folds on state, so no county is predicted
by a model that saw its own state.

Run after `extract_source_a_features.py`, `extract_source_a_section_features.py`
and `ingest_source_e.py`. Read-only with respect to the shipped parquets.

    uv run scripts/analyze_source_a_wage.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import KFold, cross_val_predict

from analyze_external_target import (
    GEO_FEATURES,
    N_FOLDS,
    _pipeline,
    out_of_fold_predictions,
)
from analyze_source_a_tiers import TIER_EDGES, TIER_LABELS, assign_tiers
from pillar_matrix import SIZE_FEATURES, build_matrix
from stats_utils import benjamini_hochberg

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

PANEL_PATH: Path = DATA_DIR / "source_e_irs_soi_panel.parquet"
CENTROIDS_PATH: Path = DATA_DIR / "county_centroids.parquet"
STRUCTURE_PATH: Path = DATA_DIR / "source_a_structure_features.parquet"
SHAPE_PATH: Path = DATA_DIR / "source_a_shape_profile.parquet"

TIER_CSV_PATH: Path = OUTPUTS_DIR / "source_a_wage_by_tier.csv"
EFFECTS_CSV_PATH: Path = OUTPUTS_DIR / "source_a_wage_feature_effects.csv"
SCORES_CSV_PATH: Path = OUTPUTS_DIR / "source_a_wage_scores.csv"
BLOCKS_CSV_PATH: Path = OUTPUTS_DIR / "source_a_wage_block_scores.csv"
STATS_PATH: Path = ANALYSIS_DIR / "source_a_wage_stats.json"

# The two IRS targets. `level` is the cross-section; `growth` is the log change
# across the full span of the panel.
LEVEL_TARGET: str = "wage_per_return_thousands"
GROWTH_TARGET: str = "wage_growth_log"

# Panel endpoints for the growth target. 2018 is the panel's first year and
# 2022 its last; using the endpoints rather than a fitted slope keeps the
# quantity readable as "how much did average wages move over the span", and the
# 2018 level enters every growth model as a mean-reversion control.
GROWTH_START_YEAR: int = 2018
GROWTH_END_YEAR: int = 2022

# Bootstrap replicates behind the contribution intervals. The resample is over
# counties and the out-of-fold predictions are fixed before it runs, so this
# prices the sampling variation in the score, not in the fit.
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_PERCENTILES: tuple[float, float] = (2.5, 97.5)
RANDOM_SEED: int = 42

# A feature effect is called only if its BH q-value clears this. The family is
# every Source A column against one target under one control set, corrected
# together -- 29 columns is enough that the largest few would look significant
# by selection alone.
Q_THRESHOLD: float = 0.05

# Gaussian draws behind each width-matched noise floor. Three is enough to
# place a floor whose sampling noise is small against the gaps being read; it
# is not enough to resolve two blocks whose contributions differ in the fourth
# decimal, and the notebook should not be asked to.
N_NOISE_DRAWS: int = 3

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_wage_panel() -> pd.DataFrame:
    """Build the per-county wage level and growth targets from the IRS panel.

    Returns:
        DataFrame keyed by `fips_code` carrying `wage_start`, `wage_end` and
        `wage_growth_log`, restricted to counties observed in both endpoint
        years. Counties missing an endpoint are dropped rather than
        interpolated: the panel's missing rows are boundary changes
        (Connecticut's planning regions, the Valdez-Cordova split), where a
        filled value would describe a county that did not exist.

    Raises:
        FileNotFoundError: If the IRS panel parquet is absent.
    """
    panel = pd.read_parquet(PANEL_PATH)
    wide = panel.pivot(
        index="fips_code", columns="tax_year", values=LEVEL_TARGET
    )
    endpoints = wide[[GROWTH_START_YEAR, GROWTH_END_YEAR]].dropna()
    dropped = len(wide) - len(endpoints)
    if dropped:
        logger.info(
            "growth target: %d of %d panel counties lack an endpoint year and are dropped",
            dropped,
            len(wide),
        )
    return pd.DataFrame(
        {
            "fips_code": endpoints.index,
            "wage_start": endpoints[GROWTH_START_YEAR].to_numpy(),
            "wage_end": endpoints[GROWTH_END_YEAR].to_numpy(),
            GROWTH_TARGET: np.log(
                endpoints[GROWTH_END_YEAR].to_numpy()
                / endpoints[GROWTH_START_YEAR].to_numpy()
            ),
        }
    )


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    """Join Source A's block, the IRS targets, county size and the tier label.

    Source A ships 3,144 counties and the IRS file 3,143; the inner join drops
    Kalawao County, Hawaii, which has no IRS return file. The join is reported
    rather than silently taken.

    The structure and shape-profile parquets are joined too. Neither ships in
    the feature matrix, so neither is reachable through `build_matrix`, and both
    are needed for the block comparison in `score_blocks`.

    Returns:
        Tuple of (panel, source_a_columns). `panel` carries `fips_code`,
        `state_fips`, `tier`, `log_population`, the `SIZE_FEATURES` columns,
        every Source A feature column, every structure and shape-profile
        column, and both targets.

    Raises:
        FileNotFoundError: If a pillar parquet or the IRS panel is absent.
    """
    matrix, blocks = build_matrix()
    source_a_columns = blocks["A"]

    # `wage_per_return_thousands` is a Source E *feature* in the matrix. It is
    # the target here, so it is pulled out by name and E's block never enters
    # any design below -- a baseline holding E would hold the answer.
    irs = pd.read_parquet(DATA_DIR / "source_e_irs_soi.parquet")[
        ["fips_code", LEVEL_TARGET, "num_returns"]
    ]
    growth = load_wage_panel()
    centroids = pd.read_parquet(CENTROIDS_PATH)[["fips_code", *GEO_FEATURES]]
    structure = pd.read_parquet(STRUCTURE_PATH)
    shape = pd.read_parquet(SHAPE_PATH)

    panel = (
        matrix[["fips_code", "county_name", "state_fips", *SIZE_FEATURES, *source_a_columns]]
        .merge(irs, on="fips_code", how="inner")
        .merge(centroids, on="fips_code", how="left")
        .merge(structure, on="fips_code", how="inner")
        .merge(shape, on="fips_code", how="inner")
        .merge(growth, on="fips_code", how="left")
        .reset_index(drop=True)
    )
    missing_centroid = int(panel[list(GEO_FEATURES)].isna().any(axis=1).sum())
    if missing_centroid:
        raise ValueError(
            f"{missing_centroid} panel counties absent from {CENTROIDS_PATH.name}"
        )
    panel["tier"] = assign_tiers(panel["content_length"])
    panel["log_wage_start"] = np.log(panel["wage_start"])

    logger.info(
        "panel: %d counties, %d Source A columns; %d carry the growth target",
        len(panel),
        len(source_a_columns),
        int(panel[GROWTH_TARGET].notna().sum()),
    )
    return panel, source_a_columns


def summarize_tiers(panel: pd.DataFrame) -> pd.DataFrame:
    """Report both targets, and county size, within each content tier.

    Args:
        panel: Joined panel from `load_panel`.

    Returns:
        One row per tier, in tier order.
    """
    grouped = panel.groupby("tier", observed=True)
    summary = pd.DataFrame(
        {
            "n_counties": grouped.size(),
            "median_content_length": grouped["content_length"].median(),
            "median_population": (10 ** grouped["log_population"].median()).round(0),
            "mean_wage": grouped[LEVEL_TARGET].mean(),
            "median_wage": grouped[LEVEL_TARGET].median(),
            "mean_growth": grouped[GROWTH_TARGET].mean(),
            "median_growth": grouped[GROWTH_TARGET].median(),
            "n_growth": grouped[GROWTH_TARGET].count(),
        }
    )
    return summary.loc[[t for t in TIER_LABELS if t in summary.index]]


def _design(panel: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Stack an intercept onto the named columns, as a float array.

    Args:
        panel: Frame holding `columns`.
        columns: Control column names.

    Returns:
        Array of shape (n, len(columns) + 1).
    """
    block = panel[columns].astype(float).to_numpy()
    return np.column_stack([np.ones(len(panel)), block])


def _state_dummies(panel: pd.DataFrame) -> np.ndarray:
    """One indicator column per state, first level dropped.

    Args:
        panel: Frame carrying `state_fips`.

    Returns:
        Indicator array aligned to `panel`.
    """
    return pd.get_dummies(panel["state_fips"], drop_first=True).to_numpy(dtype=float)


def _residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Return the part of `values` orthogonal to `design`.

    Args:
        values: Vector or column-stacked matrix to residualize.
        design: Control design, intercept included.

    Returns:
        Residuals, same shape as `values`.
    """
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def tier_gradient(panel: pd.DataFrame, target: str) -> dict[str, object]:
    """Measure the tier gradient in `target` before and after the size control.

    Tiers enter as indicators against `stub`, so each coefficient reads as the
    gap between that tier and the stub tier. The controlled fit adds
    `log_population`, and the share of the raw gap that survives it is the
    number the tier story turns on.

    Args:
        panel: Joined panel from `load_panel`.
        target: Target column name.

    Returns:
        Mapping with per-tier raw and size-controlled coefficients, the R2 of
        each fit, and the share of the rich-vs-stub gap that survives.
    """
    usable = panel[panel[target].notna()].reset_index(drop=True)
    y = usable[target].astype(float).to_numpy()
    levels = [t for t in TIER_LABELS if t in set(usable["tier"].dropna())]
    indicators = np.column_stack(
        [(usable["tier"] == level).to_numpy(dtype=float) for level in levels[1:]]
    )

    raw_design = np.column_stack([np.ones(len(usable)), indicators])
    controlled_design = np.column_stack(
        [raw_design, usable["log_population"].to_numpy(dtype=float)]
    )

    def fit(design: np.ndarray) -> tuple[np.ndarray, float]:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients
        r2 = 1.0 - float(residuals @ residuals) / float(((y - y.mean()) ** 2).sum())
        return coefficients, r2

    raw_beta, raw_r2 = fit(raw_design)
    controlled_beta, controlled_r2 = fit(controlled_design)

    raw = {level: float(raw_beta[i + 1]) for i, level in enumerate(levels[1:])}
    controlled = {level: float(controlled_beta[i + 1]) for i, level in enumerate(levels[1:])}
    top = levels[-1]
    return {
        "reference_tier": levels[0],
        "raw_coefficients": raw,
        "size_controlled_coefficients": controlled,
        "raw_r2": raw_r2,
        "size_controlled_r2": controlled_r2,
        "top_tier": top,
        "share_of_top_gap_surviving_size": (
            float(controlled[top] / raw[top]) if raw[top] else float("nan")
        ),
    }


def feature_effects(
    panel: pd.DataFrame, source_a_columns: list[str], target: str
) -> pd.DataFrame:
    """Partial correlation of every Source A column with `target`.

    Two control sets, reported side by side. `size` holds `log_population`;
    `size_state` adds one indicator per state, which is what separates a
    national effect from a handful of states sharing both a flag and a wage
    level. The growth target adds `log_wage_start` to both, so an effect is not
    just mean reversion from a low 2018 base.

    Significance is the analytic t-test on the partial correlation, with
    degrees of freedom reduced by the control rank, then Benjamini-Hochberg
    across all Source A columns within one target and control set.

    `adjusted_effect` is the partial regression slope in the target's own
    units. For a flag it reads as the wage gap, in thousands of dollars, or the
    growth gap, in log points, between counties whose article carries the flag
    and counties otherwise alike in size (and state, under `size_state`).
    `effect_per_sd` rescales it to one standard deviation of the column, which
    is the only unit that puts the 0/1 flags and `content_length` on a shared
    axis -- see the comment at its assignment.

    Args:
        panel: Joined panel from `load_panel`.
        source_a_columns: Source A feature column names.
        target: Target column name.

    Returns:
        One row per column per control set.
    """
    usable = panel[panel[target].notna()].reset_index(drop=True)
    y = usable[target].astype(float).to_numpy()

    base_controls = ["log_population"]
    if target == GROWTH_TARGET:
        base_controls = base_controls + ["log_wage_start"]

    control_sets: dict[str, np.ndarray] = {
        "size": _design(usable, base_controls),
        "size_state": np.column_stack(
            [_design(usable, base_controls), _state_dummies(usable)]
        ),
    }

    rows: list[dict[str, object]] = []
    for scheme, design in control_sets.items():
        rank = int(np.linalg.matrix_rank(design))
        residual_y = _residualize(y, design)
        raw_p: list[float] = []
        scheme_rows: list[dict[str, object]] = []
        for column in source_a_columns:
            values = usable[column].astype(float)
            # `founding_year` is the one documented null-bearing Source A
            # column. Median-fill it here rather than dropping the rows, so
            # every column is measured on the same counties.
            values = values.fillna(values.median()).to_numpy()
            residual_x = _residualize(values, design)
            denominator = np.sqrt(
                float(residual_x @ residual_x) * float(residual_y @ residual_y)
            )
            r = float(residual_x @ residual_y / denominator) if denominator else 0.0
            # The partial regression slope, in the target's own units, so a
            # flag's effect can be read as thousands of dollars (level) or log
            # points of growth rather than only as a correlation.
            variance_x = float(residual_x @ residual_x)
            adjusted = float(residual_x @ residual_y / variance_x) if variance_x else 0.0
            # The same slope rescaled to one standard deviation of the column.
            # Source A mixes 0/1 flags with counts and a character length, whose
            # natural units differ by four orders of magnitude -- a per-unit
            # effect for `content_length` is dollars per character. Charting
            # per-unit effects on one axis renders the continuous columns as
            # invisible slivers, so the comparable quantity is per SD.
            column_sd = float(values.std(ddof=1))
            df = len(usable) - rank - 1
            t = r * np.sqrt(df / max(1e-12, 1 - r**2))
            p = float(2 * stats.t.sf(abs(t), df))
            raw_p.append(p)
            unique = set(np.unique(values))
            is_flag = unique <= {0.0, 1.0}
            scheme_rows.append(
                {
                    "target": target,
                    "control_set": scheme,
                    "column": column,
                    "is_flag": is_flag,
                    "n_true": int(values.sum()) if is_flag else None,
                    "raw_mean_difference": (
                        float(y[values == 1].mean() - y[values == 0].mean())
                        if is_flag and 0 < values.sum() < len(values)
                        else None
                    ),
                    "partial_r": r,
                    "adjusted_effect": adjusted,
                    "column_sd": column_sd,
                    "effect_per_sd": adjusted * column_sd,
                    "p_value": p,
                }
            )
        for row, q in zip(scheme_rows, benjamini_hochberg(raw_p)):
            row["q_value"] = q
            row["significant"] = bool(q < Q_THRESHOLD)
        rows.extend(scheme_rows)

    return pd.DataFrame(rows)


def _r2(y: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination against the mean of `y`.

    Args:
        y: Observed values.
        predicted: Predicted values.

    Returns:
        R2, which is negative when the prediction is worse than the mean.
    """
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return 1.0 - residual / total


def _bootstrap_interval(
    y: np.ndarray, reduced: np.ndarray, full: np.ndarray
) -> dict[str, float]:
    """Percentile interval on the contribution, resampling counties.

    Args:
        y: Observed target.
        reduced: Out-of-fold predictions without Source A.
        full: Out-of-fold predictions with Source A.

    Returns:
        Mapping with `point`, `low` and `high`.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)
    draws = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        index = rng.integers(0, n, n)
        draws[i] = _r2(y[index], full[index]) - _r2(y[index], reduced[index])
    low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
    return {
        "point": _r2(y, full) - _r2(y, reduced),
        "low": float(low),
        "high": float(high),
    }


def score_target(
    panel: pd.DataFrame, source_a_columns: list[str], target: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score Source A out-of-fold against `target`, over three baselines.

    The baselines, weakest control first:

    - `headcount`: `log_population` alone (plus `log_wage_start` for growth).
      The honest control for the tier confound, and the one the headline uses.
    - `geo`: the headcount control plus county centroid latitude and longitude.
      This is the control that erased Source A's measured value in the
      representation round (findings §22.2), so it is the one that decides
      whether a wage result is about a county's economy or about where it sits.
    - `size_full`: the three-column `SIZE_FEATURES`. Documented as
      **uninterpretable for this target** -- `log_agi` minus `log_population`
      is income per head, which nearly restates wage per return. Reported so
      the objection is visible, not as evidence.

    State is deliberately **not** a baseline here. Folds are grouped on state,
    so a test county's state indicator is zero in every training row and the
    dummies can carry nothing -- the arm would price the noise of 50 dead
    columns, not the value of knowing the state. The within-state question is
    answered instead by the `size_state` control set in `feature_effects`,
    which residualizes in sample and has no held-out-state problem.

    Args:
        panel: Joined panel from `load_panel`.
        source_a_columns: Source A feature column names.
        target: Target column name.

    Returns:
        Tuple of (scores, detail). `scores` has one row per baseline; `detail`
        carries the bootstrap intervals and the per-tier breakdown of the
        headcount contribution.
    """
    usable = panel[panel[target].notna()].reset_index(drop=True)
    y = usable[target].astype(float).to_numpy()
    groups = usable["state_fips"].to_numpy()
    tier = usable["tier"].to_numpy()

    base_controls = ["log_population"]
    if target == GROWTH_TARGET:
        base_controls = base_controls + ["log_wage_start"]

    baselines: dict[str, np.ndarray] = {
        "headcount": _design(usable, base_controls)[:, 1:],
        "geo": _design(usable, [*base_controls, *GEO_FEATURES])[:, 1:],
        "size_full": _design(usable, [*base_controls, *SIZE_FEATURES])[:, 1:],
    }
    source_a = usable[source_a_columns].astype(float).to_numpy()

    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {"n_counties": len(usable), "by_baseline": {}}
    for name, control in baselines.items():
        reduced = out_of_fold_predictions(control, y, groups)
        full = out_of_fold_predictions(
            np.column_stack([control, source_a]), y, groups
        )
        reduced_r2, full_r2 = _r2(y, reduced), _r2(y, full)
        rows.append(
            {
                "target": target,
                "baseline": name,
                "n_control_columns": control.shape[1],
                "reduced_r2": reduced_r2,
                "full_r2": full_r2,
                "contribution": full_r2 - reduced_r2,
            }
        )
        entry: dict[str, object] = {
            "n_control_columns": int(control.shape[1]),
            "reduced_r2": reduced_r2,
            "full_r2": full_r2,
            "contribution": full_r2 - reduced_r2,
            "bootstrap": _bootstrap_interval(y, reduced, full),
        }
        if name == "headcount":
            # Scored on the same global out-of-fold predictions, split by tier
            # afterwards. This asks where Source A pays off, not whether a
            # separate model per tier would do better -- which the pillar-worth
            # round already answered "no" for this pillar.
            entry["by_tier"] = {
                level: {
                    "n_counties": int(mask.sum()),
                    "reduced_r2": _r2(y[mask], reduced[mask]),
                    "full_r2": _r2(y[mask], full[mask]),
                    "contribution": _r2(y[mask], full[mask]) - _r2(y[mask], reduced[mask]),
                }
                for level in TIER_LABELS
                if (mask := tier == level).sum() > 0
            }
        detail["by_baseline"][name] = entry

    return pd.DataFrame(rows), detail


def build_blocks(panel: pd.DataFrame, source_a_columns: list[str]) -> dict[str, list[str]]:
    """Group the article's measurable properties into blocks, cheapest first.

    The question these answer is how much of a county's wages can be read off
    properties of its Wikipedia article that require **no reading of the text**
    -- how long it is, how many sections it has, how long those are, which
    titles are present, what order they come in, how many digits they contain.
    The lexicon block is included as the comparator: it is the only one of
    these that actually parses words.

    Blocks are nested on purpose. `length_only` and `sections_only` are single
    columns and are the floor; `counts_and_lengths` adds the section-size
    distribution; `which_sections` is the presence flags alone;
    `section_mix` is how the characters divide across section families;
    `structure` is all of those together; `shape_profile` adds section
    ordering, template conformity and typography; `everything_dumb` is the
    union of the two text-free blocks.

    Args:
        panel: Joined panel from `load_panel`.
        source_a_columns: Source A feature column names.

    Returns:
        Mapping of block key to column list, in reporting order.
    """
    structure = pd.read_parquet(STRUCTURE_PATH).columns.drop("fips_code").tolist()
    shape = pd.read_parquet(SHAPE_PATH).columns.drop("fips_code").tolist()

    which_sections = [c for c in structure if c.startswith("has_section_")]
    section_mix = [c for c in structure if c.startswith("share_chars_")]
    counts_and_lengths = [
        c for c in structure if c not in which_sections and c not in section_mix
    ]

    blocks = {
        "length_only": ["content_length"],
        "sections_only": ["n_body_sections"],
        "counts_and_lengths": counts_and_lengths,
        "which_sections": which_sections,
        "section_mix": section_mix,
        "structure": structure,
        "shape_profile": shape,
        "everything_dumb": structure + shape,
        "typed_lexicon": source_a_columns,
    }
    missing = {
        key: [c for c in columns if c not in panel.columns]
        for key, columns in blocks.items()
    }
    absent = {key: cols for key, cols in missing.items() if cols}
    if absent:
        raise ValueError(f"block columns absent from the panel: {absent}")
    return blocks


def _noise_block(n_rows: int, width: int, draw: int) -> np.ndarray:
    """Gaussian noise of a given width, independent of every row's identity.

    The width-matched noise floor. A 137-column block scored against a
    1-column baseline is not comparable to a 1-column block scored against the
    same baseline -- ridge pays for width, and the price is negative and grows
    with the block. Reporting a real block's contribution without the floor its
    own width scores is the framing error findings §23 exists to document.

    Args:
        n_rows: Rows to generate.
        width: Columns to generate.
        draw: Draw index, mixed into the seed so repeated draws differ.

    Returns:
        Array of shape (n_rows, width).
    """
    rng = np.random.default_rng(RANDOM_SEED + 1000 * draw + width)
    return rng.standard_normal((n_rows, width))


def score_blocks(
    panel: pd.DataFrame, blocks: dict[str, list[str]], target: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score every block against `target`, over the two usable baselines.

    `size_full` is not scored here. It nearly restates the level target (see
    the module docstring), and running nine blocks against it would produce a
    table of small numbers that mean nothing, presented beside numbers that do.

    Each block is reported beside the mean contribution of `N_NOISE_DRAWS`
    Gaussian blocks of its own width, scored through the identical path. A
    block that does not beat its own noise floor has not been shown to carry
    anything.

    Args:
        panel: Joined panel from `load_panel`.
        blocks: Block key to column list, from `build_blocks`.
        target: Target column name.

    Returns:
        Tuple of (scores, detail), one row and one entry per block per baseline.
    """
    usable = panel[panel[target].notna()].reset_index(drop=True)
    y = usable[target].astype(float).to_numpy()
    groups = usable["state_fips"].to_numpy()

    base_controls = ["log_population"]
    if target == GROWTH_TARGET:
        base_controls = base_controls + ["log_wage_start"]
    baselines = {
        "headcount": _design(usable, base_controls)[:, 1:],
        "geo": _design(usable, [*base_controls, *GEO_FEATURES])[:, 1:],
    }

    rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    noise_cache: dict[tuple[str, int], float] = {}
    for baseline_name, control in baselines.items():
        reduced = out_of_fold_predictions(control, y, groups)
        reduced_r2 = _r2(y, reduced)
        for key, columns in blocks.items():
            block = usable[columns].astype(float).to_numpy()
            full = out_of_fold_predictions(np.column_stack([control, block]), y, groups)
            contribution = _r2(y, full) - reduced_r2

            width = len(columns)
            cache_key = (baseline_name, width)
            if cache_key not in noise_cache:
                draws = [
                    _r2(y, out_of_fold_predictions(
                        np.column_stack([control, _noise_block(len(usable), width, d)]),
                        y, groups)) - reduced_r2
                    for d in range(N_NOISE_DRAWS)
                ]
                noise_cache[cache_key] = float(np.mean(draws))
            floor = noise_cache[cache_key]

            interval = _bootstrap_interval(y, reduced, full)
            row = {
                "target": target,
                "baseline": baseline_name,
                "block": key,
                "n_columns": width,
                "reduced_r2": reduced_r2,
                "full_r2": _r2(y, full),
                "contribution": contribution,
                "noise_floor": floor,
                "net_of_noise_floor": contribution - floor,
                "ci_low": interval["low"],
                "ci_high": interval["high"],
                "clears_zero": bool(interval["low"] > 0),
            }
            rows.append(row)
            detail.setdefault(baseline_name, {})[key] = {
                k: v for k, v in row.items()
                if k not in ("target", "baseline", "block")
            }

    return pd.DataFrame(rows), detail


def cv_transfer(
    panel: pd.DataFrame, blocks: dict[str, list[str]], target: str
) -> dict[str, dict[str, float]]:
    """Each block's contribution under random folds versus held-out states.

    Every other score in this module holds out whole states, which is the right
    protocol and also the one that makes several of these blocks look
    catastrophic. This isolates why: the same block is scored a second time
    under a plain shuffled `KFold`, where a county's own state is in the
    training set.

    A block whose contribution is positive under random folds and negative
    under held-out states is not carrying a fact about counties. It is carrying
    a fact about **editors** -- which sections a county article has is largely
    set by whichever WikiProject templated that state's counties, so the
    relationship it learns is calibrated per state and does not transfer to a
    state the model has never seen. Article length and the lexicon flags do not
    behave this way, and the gap between the two protocols is the cleanest
    available evidence for the distinction.

    Args:
        panel: Joined panel from `load_panel`.
        blocks: Block key to column list, from `build_blocks`.
        target: Target column name.

    Returns:
        Mapping of block key to `{"grouped_by_state": R2 gain,
        "random_folds": R2 gain, "transfer_gap": grouped minus random}`.
    """
    usable = panel[panel[target].notna()].reset_index(drop=True)
    y = usable[target].astype(float).to_numpy()
    groups = usable["state_fips"].to_numpy()

    base_controls = ["log_population"]
    if target == GROWTH_TARGET:
        base_controls = base_controls + ["log_wage_start"]
    control = _design(usable, base_controls)[:, 1:]

    random_folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def random_oof(design: np.ndarray) -> np.ndarray:
        return cross_val_predict(_pipeline(), design, y, cv=random_folds)

    grouped_reduced = _r2(y, out_of_fold_predictions(control, y, groups))
    random_reduced = _r2(y, random_oof(control))

    results: dict[str, dict[str, float]] = {}
    for key, columns in blocks.items():
        block = usable[columns].astype(float).to_numpy()
        design = np.column_stack([control, block])
        grouped = _r2(y, out_of_fold_predictions(design, y, groups)) - grouped_reduced
        random = _r2(y, random_oof(design)) - random_reduced
        results[key] = {
            "grouped_by_state": grouped,
            "random_folds": random,
            "transfer_gap": grouped - random,
        }
    return results


def size_recoverability(
    panel: pd.DataFrame, blocks: dict[str, list[str]]
) -> dict[str, float]:
    """How much of `log_population` each block reconstructs out-of-fold.

    The direct answer to "is this just county size?". Findings §24 asked it of
    the shape block against the in-repo basket and found it rebuilds most of
    size unaided; this repeats the measurement on the panel this notebook uses,
    so the wage contributions and the size-recovery numbers are computed on the
    same rows and can be read against each other.

    A block that reconstructs population well and still adds R² over a model
    already holding population is carrying something population does not.

    Args:
        panel: Joined panel from `load_panel`.
        blocks: Block key to column list, from `build_blocks`.

    Returns:
        Mapping of block key to out-of-fold R2 predicting `log_population`.
    """
    y = panel["log_population"].astype(float).to_numpy()
    groups = panel["state_fips"].to_numpy()
    return {
        key: _r2(y, out_of_fold_predictions(
            panel[columns].astype(float).to_numpy(), y, groups))
        for key, columns in blocks.items()
    }


def main() -> None:
    """Run both targets end to end and write the CSVs and the stats artifact."""
    configure_logging()

    panel, source_a_columns = load_panel()

    tiers = summarize_tiers(panel)
    logger.info("wage by content tier:\n%s", tiers.round(4).to_string())

    effects = pd.concat(
        [feature_effects(panel, source_a_columns, target) for target in (LEVEL_TARGET, GROWTH_TARGET)],
        ignore_index=True,
    )
    scores: list[pd.DataFrame] = []
    detail: dict[str, object] = {}
    for target in (LEVEL_TARGET, GROWTH_TARGET):
        frame, target_detail = score_target(panel, source_a_columns, target)
        scores.append(frame)
        detail[target] = target_detail
        logger.info("%s:\n%s", target, frame.round(5).to_string(index=False))

    scores_frame = pd.concat(scores, ignore_index=True)

    blocks = build_blocks(panel, source_a_columns)
    block_frames: list[pd.DataFrame] = []
    block_detail: dict[str, object] = {}
    for target in (LEVEL_TARGET, GROWTH_TARGET):
        frame, target_detail = score_blocks(panel, blocks, target)
        block_frames.append(frame)
        block_detail[target] = target_detail
        logger.info(
            "%s by block:\n%s",
            target,
            frame[frame["baseline"] == "headcount"]
            .drop(columns=["target", "baseline"])
            .round(5)
            .to_string(index=False),
        )
    blocks_frame = pd.concat(block_frames, ignore_index=True)

    transfer = {
        target: cv_transfer(panel, blocks, target)
        for target in (LEVEL_TARGET, GROWTH_TARGET)
    }
    logger.info(
        "%s, contribution under held-out states vs random folds:\n%s",
        LEVEL_TARGET,
        "\n".join(
            f"  {key:22} grouped {v['grouped_by_state']:+.4f}  "
            f"random {v['random_folds']:+.4f}  gap {v['transfer_gap']:+.4f}"
            for key, v in transfer[LEVEL_TARGET].items()
        ),
    )

    recovery = size_recoverability(panel, blocks)
    logger.info(
        "out-of-fold R2 predicting log_population from each block:\n%s",
        "\n".join(f"  {key:22} {value:+.4f}" for key, value in recovery.items()),
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    tiers.to_csv(TIER_CSV_PATH)
    effects.to_csv(EFFECTS_CSV_PATH, index=False)
    scores_frame.to_csv(SCORES_CSV_PATH, index=False)
    blocks_frame.to_csv(BLOCKS_CSV_PATH, index=False)

    STATS_PATH.write_text(
        json.dumps(
            {
                "n_counties": int(len(panel)),
                "n_counties_growth": int(panel[GROWTH_TARGET].notna().sum()),
                "n_source_a_columns": len(source_a_columns),
                "source_a_columns": source_a_columns,
                "level_target": LEVEL_TARGET,
                "growth_target": GROWTH_TARGET,
                "growth_years": [GROWTH_START_YEAR, GROWTH_END_YEAR],
                "tier_edges": list(TIER_EDGES),
                "tier_labels": list(TIER_LABELS),
                "q_threshold": Q_THRESHOLD,
                "tier_summary": json.loads(tiers.to_json(orient="index")),
                "tier_gradient": {
                    target: tier_gradient(panel, target)
                    for target in (LEVEL_TARGET, GROWTH_TARGET)
                },
                "scores": detail,
                "blocks": {
                    "columns": {key: columns for key, columns in blocks.items()},
                    "widths": {key: len(columns) for key, columns in blocks.items()},
                    "scores": block_detail,
                    "size_recoverability": recovery,
                    "cv_transfer": transfer,
                    "n_noise_draws": N_NOISE_DRAWS,
                },
                "n_bootstrap": N_BOOTSTRAP,
            },
            indent=2,
        )
    )

    for path in (TIER_CSV_PATH, EFFECTS_CSV_PATH, SCORES_CSV_PATH, STATS_PATH):
        logger.info("wrote %s", path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
