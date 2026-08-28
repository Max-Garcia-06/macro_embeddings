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
from sklearn.impute import SimpleImputer
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
from pillar_matrix import SIZE_FEATURES, build_matrix

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


def size_recoverability(
    matrix: pd.DataFrame, blocks_by_key: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """How much of county size each block can reconstruct.

    §23 closed on an open problem: the per-column size audit comes back clean
    while the block as a whole carries size, because the dependence is *joint*
    across columns and no per-column statistic can see it. This inverts the
    question. Predicting size *from* shape measures the joint channel directly,
    in one number, and bounds how much of any reported lift could be size in
    disguise.

    Both learners run, because the channel §23 found is curved and a linear
    reading of it would understate it -- which is the same mistake round one made
    one level up.

    Args:
        matrix: Any frame carrying `SIZE_FEATURES`.
        blocks_by_key: Feature arrays to test, keyed by name. Every array must
            have one row per row of `matrix`.

    Returns:
        Nested mapping of block key to `{"<size measure>_<learner>": R2}`.
    """
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sizes = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(matrix[list(SIZE_FEATURES)]),
        columns=list(SIZE_FEATURES),
        index=matrix.index,
    )

    recovery: dict[str, dict[str, float]] = {}
    for key, block in blocks_by_key.items():
        scores: dict[str, float] = {}
        for measure in SIZE_FEATURES:
            y = sizes[measure].to_numpy(dtype="float64")
            scores[f"{measure}_ridge"] = _alone_oof_r2(block, y, folds, None)
            scores[f"{measure}_boost"] = boost_alone_oof_r2(block, y, folds)
        recovery[key] = scores
    return recovery


def summarize_by_pillar(results: pd.DataFrame) -> pd.DataFrame:
    """Mean lift per arm within each target's owning pillar.

    Reported beside the aggregate and never instead of it: 20 of the 28 targets
    are one QCEW table, so a basket-wide mean is 71% one pillar.

    Args:
        results: Output of `run_sweep`.

    Returns:
        One row per pillar with the target count and each arm's mean lift under
        each learner.
    """
    aggregations: dict[str, tuple[str, str]] = {"n_targets": ("column", "count")}
    for arm in SHAPE_ARMS:
        for suffix in ("", BOOST_SUFFIX):
            aggregations[f"{arm.key}{suffix}"] = (f"lift_{arm.key}{suffix}", "mean")
    return results.groupby("pillar").agg(**aggregations).reset_index()


def _paired_test(results: pd.DataFrame, arm: Arm, column: str) -> dict[str, object]:
    """Test one arm's lift column against its comparison across every target.

    Args:
        results: Output of `run_sweep`.
        arm: The arm to test. `arm.against` names the arm it is paired with;
            None means the comparison is against the baseline, where the lift
            column is already the difference.
        column: Full lift column name, carrying whichever suffixes apply.

    Returns:
        Mean lift, mean paired difference, win count and the Wilcoxon
        signed-rank p-value.
    """
    lifts = results[column]
    if arm.against is None:
        differences = lifts
    else:
        differences = lifts - results[column.replace(arm.key, arm.against, 1)]
    statistic, p_value = wilcoxon(differences)
    return {
        "mean_lift": float(lifts.mean()),
        "median_lift": float(lifts.median()),
        "mean_paired_difference": float(differences.mean()),
        "compared_against": arm.against or "baseline",
        "n_wins": int((differences > 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def summarize(
    results: pd.DataFrame,
    size_recovery: dict[str, dict[str, float]],
    n_v1: int,
    n_profile: int,
) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    Args:
        results: Output of `run_sweep`.
        size_recovery: Output of `size_recoverability`.
        n_v1: Width of round one's block.
        n_profile: Width of the new shape profile.

    Returns:
        Target counts, block widths, the size diagnostic, and per-arm results in
        all three framings under both learners.
    """
    arms: dict[str, object] = {}
    for arm in SHAPE_ARMS:
        for learner, suffix in (("ridge", ""), ("boost", BOOST_SUFFIX)):
            arms[f"{arm.key}_{learner}"] = {
                "label": arm.label,
                "learner": learner,
                "mean_r2_alone": float(results[f"r2_alone_{arm.key}{suffix}"].mean()),
                "linear": _paired_test(results, arm, f"lift_{arm.key}{suffix}"),
                "flexible": _paired_test(
                    results, arm, f"lift_{arm.key}{FLEXIBLE_SUFFIX}{suffix}"
                ),
            }
    return {
        "n_targets": int(len(results)),
        "n_shape_v1_features": n_v1,
        "n_shape_profile_features": n_profile,
        "n_shape_v2_features": n_v1 + n_profile,
        "n_typed_features": len(typed_columns()),
        "mean_r2_baseline": float(results["r2_baseline"].mean()),
        "mean_r2_baseline_flexible": float(results["r2_baseline_flexible"].mean()),
        "size_recoverability": size_recovery,
        "arms": arms,
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def main() -> None:
    """Attach both blocks, run the sweep and the diagnostic, write the artifacts."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix, v1_cols, profile_cols = attach_blocks(matrix)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets: shape_v1=%d, profile=%d, shape_v2=%d, typed=%d",
        len(targets),
        len(v1_cols),
        len(profile_cols),
        len(v1_cols) + len(profile_cols),
        len(typed_columns()),
    )

    results = run_sweep(matrix, v1_cols, profile_cols, targets)

    all_rows = np.ones(len(matrix), dtype=bool)
    diagnostic_blocks = build_arm_blocks(matrix, v1_cols, profile_cols, all_rows)
    size_recovery = size_recoverability(
        matrix, {key: diagnostic_blocks[key] for key in ("shape_v1", "shape_v2")}
    )

    pillar_results = summarize_by_pillar(results)
    stats = summarize(results, size_recovery, len(v1_cols), len(profile_cols))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    pillar_results.to_csv(OUTPUT_PILLAR_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("size recoverable from the shape block (out-of-fold R2):")
    for key, scores in size_recovery.items():
        for measure in SIZE_FEATURES:
            logger.info(
                "  %-9s %-16s ridge %.4f | boost %.4f",
                key,
                measure,
                scores[f"{measure}_ridge"],
                scores[f"{measure}_boost"],
            )

    for name, arm_stats in stats["arms"].items():
        logger.info(
            "%-26s alone %.4f | linear %+.5f (p=%.4f) | flexible %+.5f (p=%.4f)",
            name,
            arm_stats["mean_r2_alone"],
            arm_stats["linear"]["mean_lift"],
            arm_stats["linear"]["wilcoxon_p"],
            arm_stats["flexible"]["mean_lift"],
            arm_stats["flexible"]["wilcoxon_p"],
        )

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
