"""How much can be pulled from article shape -- in all three framings at once.

Round one scored a 64-column structural block and reported +0.00269 mean lift
over a linear size-plus-state baseline. The branch review then showed that an
*information-free* nonlinear reshaping of that baseline's own size columns
scores +0.01748 through the same protocol, and that roughly three quarters of
the structural lift disappears once the baseline is allowed to be curved
(`analysis-output/source-a/source-a-findings.md` §23).

The lesson was not that the number was wrong. It was that a correct number was
quoted in a framing its readers could not see. So this module reports every arm
in three framings and never one without the others:

- `r2_alone_<arm>`  -- out-of-fold R2 with the block as the *only* predictor.
                       No controls. This is the raw-power reading: how much of a
                       county is recoverable from article shape, size and
                       geography included.
- `lift_<arm>`      -- lift over the linear size-plus-state baseline. Comparable
                       to §13 through §23.
- `lift_<arm>_flexbase` -- lift over the curvature-augmented baseline. The
                       strict reading, and the one §23 showed matters.

Five arms, with `shape_v1` present as a regression check rather than a finding:
it re-scores round one's exact block through this module, and its
`lift_shape_v1` must reproduce §23's `lift_structure`. If it does not, something
drifted and the rest of the sweep is not trustworthy.

Run after `extract_source_a_structure_features.py` and
`extract_source_a_shape_profile.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import N_FOLDS, RANDOM_SEED, Target, build_baseline_design
from analyze_source_a_representation import (
    _alone_oof_r2,
    _baseline_oof_predictions,
    _baseline_pipeline,
    _residual_oof_predictions,
    build_non_a_targets,
)
from analyze_source_a_structure import (
    FLEXIBLE_SUFFIX,
    Arm,
    attach_structure,
    build_flexible_baseline_design,
    size_nonlinear_block,
    typed_columns,
)
from extract_source_a_shape_profile import SHAPE_PROFILE_PATH, shape_profile_columns
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_shape_profile_scores.csv"
OUTPUT_PILLAR_CSV_PATH: Path = OUTPUTS_DIR / "source_a_shape_profile_by_pillar.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_shape_profile_stats_scoring.json"

SHAPE_ARMS: tuple[Arm, ...] = (
    # First, and not a finding either: this reproduces round one's block so a
    # drift anywhere in the shared machinery shows up as a changed number here
    # rather than as a silently different result downstream.
    Arm("shape_v1", "REGRESSION CHECK: round one's 64 structural columns", None),
    Arm("shape_v2", "structural block + the four new shape families", "shape_v1"),
    Arm("typed", "shipped 29 typed columns", None),
    Arm("typed_plus_shape_v2", "typed columns + full shape profile", "typed"),
    Arm(
        "size_nonlinear",
        "NULL CONTROL: nonlinear transforms of the baseline's own size columns",
        None,
    ),
)

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# Both learners run for every arm and every framing. Ridge is primary: it is what
# §13 through §23 used, so it is the only reading directly comparable to them.
# Boost is secondary and exists because "as much as possible" is bounded by the
# model class -- a stub-heavy article with 30 sections is a different object from
# a stub-heavy article with 5, and a linear learner cannot say so.
LEARNERS: tuple[str, ...] = ("ridge", "boost")

BOOST_SUFFIX: str = "_boost"

# Fixed rather than searched, deliberately. A nested search would be the honest
# tuned number, but it multiplies runtime by the grid and the point of this arm
# is a ceiling estimate. Fixed settings make the reported ceiling a *lower*
# bound on what boosting could reach, which is the safe direction for a number
# that will be quoted. `min_samples_leaf` is high for the panel size because
# 3,144 counties over 5 folds leaves ~2,500 training rows and a shallow leaf
# would fit fold noise.
BOOST_PARAMS: dict[str, object] = {
    "max_iter": 200,
    "learning_rate": 0.06,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}


def make_booster() -> HistGradientBoostingRegressor:
    """Build the boosting estimator, at fixed hyperparameters.

    Returns:
        An unfitted `HistGradientBoostingRegressor`. It handles NaNs natively,
        so no imputer is needed -- unlike the ridge path, which imputes inside
        `_residual_pipeline`.
    """
    return HistGradientBoostingRegressor(**BOOST_PARAMS)


def boost_residual_oof(
    base_design: np.ndarray, block: np.ndarray, y: np.ndarray, folds: KFold
) -> np.ndarray:
    """Out-of-fold predictions from the controls plus a boosted block.

    Mirrors `_residual_oof_predictions` exactly -- controls fitted unpenalized on
    the training rows, their residuals become the boosting target, the two
    predictions summed on the held-out rows -- with the estimator swapped. It is
    written here rather than reused because the imported routine hardcodes
    `RidgeCV`, and the ridge path must keep calling the import so its numbers
    stay bit-comparable to §23.

    Args:
        base_design: Control array.
        block: Feature block for the same rows.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        Out-of-fold prediction per row.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(base_design):
        controls = _baseline_pipeline().fit(base_design[train_idx], y[train_idx])
        residuals = y[train_idx] - controls.predict(base_design[train_idx])
        model = make_booster().fit(block[train_idx], residuals)
        predictions[test_idx] = controls.predict(base_design[test_idx]) + model.predict(
            block[test_idx]
        )
    return predictions


def boost_alone_oof_r2(block: np.ndarray, y: np.ndarray, folds: KFold) -> float:
    """Out-of-fold R2 of a boosted block with no controls at all.

    Args:
        block: Feature block.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        R2 over the concatenated out-of-fold predictions.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(block):
        model = make_booster().fit(block[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(block[test_idx])
    return float(r2_score(y, predictions))


def arm_record_keys(arm_key: str, learner_suffix: str = "") -> tuple[str, str, str]:
    """The three per-arm, per-learner column names `score_target` emits.

    This is the single source of truth for how an arm's key and a learner's
    suffix combine into a column name. `empty_record_keys` and `score_target`
    both call it instead of each independently formatting the same f-strings,
    so a change to the naming scheme cannot leave one of them stale relative to
    the other -- which is exactly the drift a hand-maintained parallel copy
    could not prevent.

    Args:
        arm_key: One of `SHAPE_ARMS`' `.key` values.
        learner_suffix: `""` for ridge (the primary learner, unsuffixed to stay
            bit-comparable to earlier rounds' column names) or `BOOST_SUFFIX`.

    Returns:
        Tuple of (raw-power column, baseline-lift column, flexible-lift column).
    """
    return (
        f"r2_alone_{arm_key}{learner_suffix}",
        f"lift_{arm_key}{learner_suffix}",
        f"lift_{arm_key}{FLEXIBLE_SUFFIX}{learner_suffix}",
    )


def empty_record_keys() -> dict[str, float]:
    """The full column set one scored target produces, with zeros.

    Exists so a test can assert that every arm carries every framing under every
    learner without running a sweep. A row missing a framing is exactly how round
    one's number came to be quoted in a reading its audience could not see.

    Returns:
        Mapping of every per-arm result column to 0.0.
    """
    keys: dict[str, float] = {}
    for arm in SHAPE_ARMS:
        for suffix in ("", BOOST_SUFFIX):
            for key in arm_record_keys(arm.key, suffix):
                keys[key] = 0.0
    return keys


def attach_blocks(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge round one's structural block and the new shape profile onto the matrix.

    Round one's merge is delegated to `analyze_source_a_structure.attach_structure`,
    which already carries the collision check, the `validate="one_to_one"` guard
    and the row-count assertion. The profile merge repeats those guards for the
    same reasons.

    Args:
        matrix: Feature matrix from `build_matrix`.

    Returns:
        Tuple of (matrix with both blocks attached, round-one column names, shape
        profile column names).

    Raises:
        FileNotFoundError: If either parquet is absent.
        ValueError: On a column collision, a non-one-to-one merge, or a row whose
            profile is missing.
    """
    attached, v1_columns = attach_structure(matrix)

    try:
        profile = pd.read_parquet(SHAPE_PROFILE_PATH)
    except FileNotFoundError:
        logger.error(
            "Need %s -- run extract_source_a_shape_profile.py first.", SHAPE_PROFILE_PATH
        )
        raise

    profile_columns = shape_profile_columns(profile)
    collisions = sorted(set(profile_columns) & set(attached.columns))
    if collisions:
        raise ValueError(f"Shape profile columns already in the matrix: {collisions}")

    merged = attached.merge(profile, on="fips_code", how="left", validate="one_to_one")
    if len(merged) != len(attached):
        raise ValueError(
            f"Profile merge changed the row count: {len(attached)} -> {len(merged)}"
        )
    missing = int(merged[profile_columns].isna().any(axis=1).sum())
    if missing:
        raise ValueError(f"{missing} matrix rows have no shape profile")
    return merged, v1_columns, profile_columns


def build_arm_blocks(
    matrix: pd.DataFrame, v1_cols: list[str], profile_cols: list[str], rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Assemble every arm's feature array for one target's usable rows.

    Every arm is sliced with the same `rows` mask, which is what makes the
    per-target differences paired.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        rows: Boolean mask of rows where the target is observed.

    Returns:
        Mapping of arm key to feature array, one entry per member of `SHAPE_ARMS`.
    """
    typed = typed_columns()
    v2 = [*v1_cols, *profile_cols]
    return {
        "shape_v1": matrix.loc[rows, v1_cols].to_numpy(dtype="float64"),
        "shape_v2": matrix.loc[rows, v2].to_numpy(dtype="float64"),
        "typed": matrix.loc[rows, typed].to_numpy(dtype="float64"),
        "typed_plus_shape_v2": matrix.loc[rows, [*typed, *v2]].to_numpy(dtype="float64"),
        "size_nonlinear": size_nonlinear_block(matrix.loc[rows]).to_numpy(dtype="float64"),
    }


def score_target(
    matrix: pd.DataFrame,
    v1_cols: list[str],
    profile_cols: list[str],
    baseline: pd.DataFrame,
    flexible_baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score every arm against one target, in all three framings.

    One splitter is constructed here and handed to every fit, so every arm sees
    identical folds and identical rows under both baselines and in the
    no-baseline reading.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        baseline: Linear design from `build_baseline_design`.
        flexible_baseline: The curvature-augmented design.
        target: The column to predict.

    Returns:
        One record with both baselines' R2 and, per arm, the raw R2 and the lift
        over each baseline.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")
    flexible_design = flexible_baseline.loc[rows].to_numpy(dtype="float64")

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_baseline = float(r2_score(y, _baseline_oof_predictions(base_design, y, folds)))
    r2_flexible = float(r2_score(y, _baseline_oof_predictions(flexible_design, y, folds)))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
        "r2_baseline_flexible": r2_flexible,
    }

    blocks = build_arm_blocks(matrix, v1_cols, profile_cols, rows)
    for arm in SHAPE_ARMS:
        block = blocks[arm.key]

        r2_alone_key, lift_key, lift_flex_key = arm_record_keys(arm.key)
        record[r2_alone_key] = _alone_oof_r2(block, y, folds, None)
        record[lift_key] = (
            float(r2_score(y, _residual_oof_predictions(base_design, block, y, folds, None)))
            - r2_baseline
        )
        record[lift_flex_key] = (
            float(
                r2_score(
                    y, _residual_oof_predictions(flexible_design, block, y, folds, None)
                )
            )
            - r2_flexible
        )

        boost_r2_alone_key, boost_lift_key, boost_lift_flex_key = arm_record_keys(
            arm.key, BOOST_SUFFIX
        )
        record[boost_r2_alone_key] = boost_alone_oof_r2(block, y, folds)
        record[boost_lift_key] = (
            float(r2_score(y, boost_residual_oof(base_design, block, y, folds)))
            - r2_baseline
        )
        record[boost_lift_flex_key] = (
            float(r2_score(y, boost_residual_oof(flexible_design, block, y, folds)))
            - r2_flexible
        )
    return record


def run_sweep(
    matrix: pd.DataFrame, v1_cols: list[str], profile_cols: list[str], targets: list[Target]
) -> pd.DataFrame:
    """Score every target against every arm.

    Args:
        matrix: Matrix with both blocks attached.
        v1_cols: Round-one structural column names.
        profile_cols: Shape-profile column names.
        targets: Targets to score.

    Returns:
        Per-target results, sorted by the `shape_v2` arm's raw R2.
    """
    baseline = build_baseline_design(matrix)
    flexible_baseline = build_flexible_baseline_design(matrix, baseline)

    records = []
    for target in targets:
        record = score_target(
            matrix, v1_cols, profile_cols, baseline, flexible_baseline, target
        )
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  alone=%.4f  lift=%+.4f  flex=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["r2_alone_shape_v2"],
            record["lift_shape_v2"],
            record[f"lift_shape_v2{FLEXIBLE_SUFFIX}"],
        )
    return (
        pd.DataFrame(records)
        .sort_values("r2_alone_shape_v2", ascending=False)
        .reset_index(drop=True)
    )
