"""Does the shape of a Wikipedia article know anything county size does not?

`extract_source_a_structure_features.py` builds a block from section counts,
section lengths and section titles -- never from section text. This module
scores it against the same 28-target cross-pillar basket, the same folds and
the same protocol as `analyze_source_a_representation.py`, whose pipeline
helpers are imported rather than reimplemented so the numbers stay directly
comparable to that sweep's.

Five arms:

- `baseline`             -- size (`log_population`, `log_agi`, `log_gdp_latest`)
                            plus state fixed effects, and nothing else
- `structure`            -- baseline plus the structural block
- `typed`                -- baseline plus the shipped 29 typed columns
- `typed_plus_structure` -- baseline plus both
- `size_nonlinear`       -- NULL CONTROL: baseline plus squares, cubes and
                            pairwise products of the baseline's *own* three size
                            columns, which add no information whatsoever

Two comparisons carry the round. `structure` against `baseline` asks what the
skeleton knows. `typed_plus_structure` against `typed` asks whether it knows
anything the shipped block does not already have, which is the fusion-relevant
question and the one most likely to come back at zero.

**What "lift" measures here, stated precisely.** The baseline is linear in
`log_population`, `log_agi` and `log_gdp_latest`, so a positive lift means the
block knows something a *linear-in-logs* size model does not. It does not mean
the block knows something county size does not. Those are different claims, and
until 2026-08-25 this docstring made the stronger one. It was false. The
`size_nonlinear` arm is what makes the difference visible: it is built from the
baseline's own three columns and contains no information the baseline lacks, yet
it scores +0.01748 mean lift on 26 of 28 targets (p = 1.3e-06) -- six and a half
times the structural arm's +0.00269. Any monotone-but-curved relationship with
county size clears this bar, so clearing it is not evidence of content.

**The baseline is doing real work, but less than it looks.** `n_body_sections`
correlates r = 0.550 against log tax returns and was cut from the scored matrix
for exactly that reason. Fitting each arm to the *residuals* of an unpenalized
size-plus-state model does keep a wide block from dragging the controls down,
and a block of pure noise does score approximately zero. What it does not do is
price a *curved* size proxy at zero -- see `size_nonlinear`. Read every number
below against that arm, not against zero.

**Per-pillar is reported beside the aggregate, never instead of it.** Findings
§14.2b established that 20 of the 28 targets are one QCEW table, so a basket-wide
mean is 71% one pillar and reads as a breadth claim the basket does not support.

Run after `extract_source_a_structure_features.py`.

Outputs: `outputs/source_a_structure_scores.csv`,
`outputs/source_a_structure_by_pillar.csv`,
`analysis-output/source-a/source_a_structure_stats.json`.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import (
    N_FOLDS,
    RANDOM_SEED,
    Target,
    build_baseline_design,
)
from analyze_source_a_representation import (
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from extract_source_a_features import VARIANT_COLUMNS
from extract_source_a_section_features import section_feature_columns
from extract_source_a_structure_features import (
    STRUCTURE_FEATURES_PATH,
    structure_feature_columns,
)
from pillar_matrix import SIZE_FEATURES, build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_structure_scores.csv"
OUTPUT_PILLAR_CSV_PATH: Path = OUTPUTS_DIR / "source_a_structure_by_pillar.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_structure_stats.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Arm:
    """One block scored on top of the shared baseline.

    Attributes:
        key: Short identifier used as the result column suffix.
        label: Human-readable description used in reports.
        against: Arm key this one is paired against, or None for a comparison
            made directly against the baseline.
    """

    key: str
    label: str
    against: str | None


ARMS: tuple[Arm, ...] = (
    Arm("structure", "structural block (shape only)", None),
    Arm("typed", "shipped 29 typed columns", None),
    Arm("typed_plus_structure", "typed columns + structural block", "typed"),
    # Last, and deliberately not a finding: this arm exists to price the unit the
    # other three are measured in. Its label leads with NULL CONTROL because the
    # arms table is the one place a reader could mistake it for a result.
    Arm(
        "size_nonlinear",
        "NULL CONTROL: nonlinear transforms of the baseline's own size columns",
        None,
    ),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def typed_columns() -> list[str]:
    """List the shipped Source A typed columns.

    Assembled here rather than imported as one name because neither extraction
    module knows about the other: the lead features live in
    `extract_source_a_features` and the section features in
    `extract_source_a_section_features`.

    Returns:
        The 29 columns `pillar_matrix` exposes as pillar A.
    """
    return [*VARIANT_COLUMNS["extracted_full"], *section_feature_columns()]


# The arm key every other number in this module should be read against.
NULL_ARM_KEY: str = "size_nonlinear"

# Suffixes for the null block's columns. `_sq`/`_cube`/`_x_` are chosen to be
# unmistakable in a results table and to be impossible to confuse with a real
# feature name: nothing in any pillar is named this way.
NULL_SQUARE_SUFFIX: str = "_sq"
NULL_CUBE_SUFFIX: str = "_cube"
NULL_PRODUCT_INFIX: str = "_x_"


def size_nonlinear_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the null arm: an information-free reshaping of the size controls.

    Every column is a deterministic function of `pillar_matrix.SIZE_FEATURES` --
    the same three columns the baseline already holds, unpenalized and in full.
    The block therefore carries *no* information the baseline lacks. Its entire
    purpose is to answer "what is a lift of +0.003 worth?", and the answer it
    gives is "less than an information-free curve on the controls".

    Squares, cubes and pairwise products specifically: those are the cheapest
    basis that spans the monotone-but-curved relationships a linear-in-logs
    baseline cannot represent, which is exactly the gap a size proxy slips
    through. No whitening or residualization is applied -- the arm is supposed
    to be collinear with the baseline, and `_residual_pipeline` already imputes,
    scales and picks a ridge penalty by nested crossvalidation.

    Args:
        frame: Any frame carrying the `SIZE_FEATURES` columns.

    Returns:
        DataFrame on `frame`'s index with nine columns: three squares, three
        cubes and three pairwise products.
    """
    size = frame[list(SIZE_FEATURES)].astype("float64")
    columns: dict[str, pd.Series] = {}
    for name in SIZE_FEATURES:
        columns[f"{name}{NULL_SQUARE_SUFFIX}"] = size[name] ** 2
        columns[f"{name}{NULL_CUBE_SUFFIX}"] = size[name] ** 3
    for left, right in itertools.combinations(SIZE_FEATURES, 2):
        columns[f"{left}{NULL_PRODUCT_INFIX}{right}"] = size[left] * size[right]
    return pd.DataFrame(columns, index=frame.index)


def attach_structure(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge the structural block onto the pillar matrix.

    Args:
        matrix: Feature matrix from `build_matrix`.

    Returns:
        Tuple of (matrix with the structural columns attached, their names).

    Raises:
        FileNotFoundError: If the structural parquet is absent.
        ValueError: If a structural column name already exists in the matrix,
            which would rename both to `_x`/`_y` and score the wrong block; or
            if the merge is not one-to-one, which would put copies of a county
            in both the train and the test side of a fold.
    """
    try:
        features = pd.read_parquet(STRUCTURE_FEATURES_PATH)
    except FileNotFoundError:
        logger.error(
            "Need %s -- run extract_source_a_structure_features.py first.",
            STRUCTURE_FEATURES_PATH,
        )
        raise

    columns = structure_feature_columns(features)
    collisions = sorted(set(columns) & set(matrix.columns))
    if collisions:
        raise ValueError(f"Structural columns already in the matrix: {collisions}")

    # `validate` and the row-count check catch different halves of the same
    # failure, and the NaN check below catches neither: a duplicated `fips_code`
    # on the right silently multiplies matrix rows, passes the NaN check, and
    # puts copies of one county in both the train and the test side of every
    # fold -- which inflates every arm at once and so leaves no arm looking odd.
    try:
        attached = matrix.merge(
            features, on="fips_code", how="left", validate="one_to_one"
        )
    except pd.errors.MergeError as error:
        raise ValueError(
            f"{STRUCTURE_FEATURES_PATH.name} must hold exactly one row per county "
            f"and join one-to-one onto the matrix: {error}"
        ) from error
    if len(attached) != len(matrix):
        raise ValueError(
            f"merging {STRUCTURE_FEATURES_PATH.name} changed the panel from "
            f"{len(matrix)} rows to {len(attached)}; it must be one row per county"
        )

    # Every county in the matrix has a Wikipedia article, so a null here means
    # the two files disagree about the panel rather than that a value is missing.
    missing = int(attached[columns].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} matrix rows have no structural features")
    return attached, columns


def build_arm_blocks(
    matrix: pd.DataFrame, structure_cols: list[str], rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Assemble every arm's feature array for one target's usable rows.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        rows: Boolean mask of rows where the target is observed.

    Every arm is sliced with the same `rows` mask, which is what makes the
    per-target differences paired.

    Returns:
        Mapping of arm key to feature array, one entry per member of `ARMS`.
    """
    typed = typed_columns()
    return {
        "structure": matrix.loc[rows, structure_cols].to_numpy(dtype="float64"),
        "typed": matrix.loc[rows, typed].to_numpy(dtype="float64"),
        "typed_plus_structure": matrix.loc[rows, [*typed, *structure_cols]].to_numpy(
            dtype="float64"
        ),
        "size_nonlinear": size_nonlinear_block(matrix.loc[rows]).to_numpy(dtype="float64"),
    }


def score_target(
    matrix: pd.DataFrame,
    structure_cols: list[str],
    baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score every arm against one target.

    Every arm sees identical folds and identical rows, which is what makes the
    per-target differences paired and the Wilcoxon test across targets legible.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        baseline: Size-plus-state design from `build_baseline_design`.
        target: The column to predict.

    Returns:
        One record with the baseline R2 and each arm's lift over it.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_baseline = float(r2_score(y, _baseline_oof_predictions(base_design, y, folds)))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
    }

    blocks = build_arm_blocks(matrix, structure_cols, rows)
    for arm in ARMS:
        predictions = _residual_oof_predictions(base_design, blocks[arm.key], y, folds, None)
        record[f"lift_{arm.key}"] = float(r2_score(y, predictions)) - r2_baseline
    return record


def run_sweep(
    matrix: pd.DataFrame, structure_cols: list[str], targets: list[Target]
) -> pd.DataFrame:
    """Score every target against every arm.

    Args:
        matrix: Matrix with structural columns attached.
        structure_cols: Structural column names.
        targets: Targets to score.

    Returns:
        Per-target results, sorted by the structural arm's lift.
    """
    baseline = build_baseline_design(matrix)
    records = []
    for target in targets:
        record = score_target(matrix, structure_cols, baseline, target)
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  structure=%+.4f  typed=%+.4f  both=%+.4f  [null size=%+.4f]",
            record["pillar"],
            record["column"],
            record["n"],
            record["lift_structure"],
            record["lift_typed"],
            record["lift_typed_plus_structure"],
            record["lift_size_nonlinear"],
        )
    return pd.DataFrame(records).sort_values("lift_structure", ascending=False).reset_index(drop=True)


def summarize_by_pillar(results: pd.DataFrame) -> pd.DataFrame:
    """Mean lift per arm within each target's owning pillar.

    Args:
        results: Output of `run_sweep`.

    Returns:
        One row per pillar with the target count and each arm's mean lift.
    """
    aggregated = results.groupby("pillar").agg(
        n_targets=("column", "count"),
        **{arm.key: (f"lift_{arm.key}", "mean") for arm in ARMS},
    )
    return aggregated.reset_index()


def _paired_test(results: pd.DataFrame, arm: Arm) -> dict[str, object]:
    """Test one arm against its comparison across every target.

    Args:
        results: Output of `run_sweep`.
        arm: The arm to test. `arm.against` names the arm it is paired with;
            None means it is compared against the baseline, where the lift
            column is already the difference.

    Returns:
        Mean lift, mean paired difference, win count and the Wilcoxon
        signed-rank p-value.
    """
    lifts = results[f"lift_{arm.key}"]
    differences = lifts if arm.against is None else lifts - results[f"lift_{arm.against}"]
    statistic, p_value = wilcoxon(differences)
    return {
        "label": arm.label,
        "compared_against": arm.against or "baseline",
        "mean_lift": float(lifts.mean()),
        "median_lift": float(lifts.median()),
        "mean_paired_difference": float(differences.mean()),
        "n_wins": int((differences > 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def summarize(
    results: pd.DataFrame, n_structure_features: int, n_size_nonlinear_features: int
) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    Args:
        results: Output of `run_sweep`.
        n_structure_features: Width of the structural block.
        n_size_nonlinear_features: Width of the null-control block.

    Returns:
        Target count, block widths, per-arm paired tests, the structural arm's
        lift expressed in units of the null arm's, and the per-pillar breakdown
        as records.
    """
    arms = {arm.key: _paired_test(results, arm) for arm in ARMS}
    null_lift = float(arms[NULL_ARM_KEY]["mean_lift"])
    return {
        "n_targets": int(len(results)),
        "n_structure_features": n_structure_features,
        "n_typed_features": len(typed_columns()),
        "n_size_nonlinear_features": n_size_nonlinear_features,
        "mean_r2_baseline": float(results["r2_baseline"].mean()),
        "null_arm_key": NULL_ARM_KEY,
        # The ratio the notebook and the findings register both quote. Computed
        # here rather than typed anywhere: it is the whole point of the null arm,
        # and a number that has to be recomputed by hand is a number that rots.
        "structure_lift_in_null_arm_units": (
            float(arms["structure"]["mean_lift"] / null_lift) if null_lift else None
        ),
        "arms": arms,
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def main() -> None:
    """Attach the structural block, run every arm, and write the artifacts."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix, structure_cols = attach_structure(matrix)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    n_null_features = len(size_nonlinear_block(matrix.head(1)).columns)
    logger.info(
        "scoring %d targets: %d structural columns against %d shipped typed columns, "
        "calibrated against a %d-column null block built from the baseline's own size columns",
        len(targets),
        len(structure_cols),
        len(typed_columns()),
        n_null_features,
    )

    results = run_sweep(matrix, structure_cols, targets)
    pillar_results = summarize_by_pillar(results)
    stats = summarize(results, len(structure_cols), n_null_features)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    pillar_results.to_csv(OUTPUT_PILLAR_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    # Per-pillar first: the aggregate below is 71% QCEW and reads as a breadth
    # claim the basket does not support unless its composition is visible.
    logger.info("per-pillar mean lift:")
    for row in pillar_results.itertuples():
        logger.info(
            "  pillar %s  %2d targets  structure %+.5f | typed %+.5f | both %+.5f "
            "| [null size %+.5f]",
            row.pillar,
            row.n_targets,
            row.structure,
            row.typed,
            row.typed_plus_structure,
            row.size_nonlinear,
        )

    for arm in ARMS:
        test = stats["arms"][arm.key]
        logger.info(
            "%-22s mean lift %+.5f | vs %-8s mean diff %+.5f | wins %2d/%d | p=%.4f",
            arm.key,
            test["mean_lift"],
            test["compared_against"],
            test["mean_paired_difference"],
            test["n_wins"],
            stats["n_targets"],
            test["wilcoxon_p"],
        )

    # The line that decides how every line above should be read.
    ratio = stats["structure_lift_in_null_arm_units"]
    logger.info(
        "CALIBRATION: the %s null block adds no information the baseline lacks, and scores "
        "%+.5f. The structural arm's %+.5f is %.2fx that -- lift here means 'beyond a "
        "LINEAR-in-logs size model', not 'beyond county size'.",
        NULL_ARM_KEY,
        stats["arms"][NULL_ARM_KEY]["mean_lift"],
        stats["arms"]["structure"]["mean_lift"],
        ratio if ratio is not None else float("nan"),
    )

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
