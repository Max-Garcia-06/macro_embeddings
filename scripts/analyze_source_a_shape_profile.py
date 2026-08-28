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


# Width of the noise block scored by `boost_floor_lift`. Arbitrary: boosting
# cannot extract structure from independent Gaussian noise regardless of how
# many columns it has, so this is fixed at a small value rather than searched.
NOISE_BLOCK_WIDTH: int = 3


def boost_floor_lift(
    matrix: pd.DataFrame,
    baseline: pd.DataFrame,
    flexible_baseline: pd.DataFrame,
    targets: list[Target],
) -> pd.DataFrame:
    """Score a block that carries no information at all through the boost path.

    Fix-round finding: boost lifts were published on an uncalibrated scale --
    `lift_shape_v1_boost = -0.0587` and `lift_shape_v1 = +0.0027` under the same
    word "lift" with no offset stated -- and the arm meant to anchor that scale,
    `size_nonlinear` (built from the baseline's own size columns), turned out to
    be the *least* negative boost arm rather than the most, so it does not
    bracket the real arms and cannot serve as a floor. This scores pure
    Gaussian noise, independent of every target and every row's identity,
    through the exact `boost_residual_oof` path every arm uses, under the same
    per-target folds and both baselines. Its lift is what a block with zero
    signal looks like on this scale, so every other boost arm's lift can be
    read as a distance from it instead of as a bare, unanchored number.

    Args:
        matrix: Matrix with both blocks attached. Used only for row count and
            alignment -- the noise block carries none of its columns.
        baseline: Linear baseline design.
        flexible_baseline: Curvature-augmented baseline design.
        targets: Targets to score, matched one-for-one with `run_sweep`'s.

    Returns:
        Per-target frame with `column`, `lift_boost_floor` (vs the linear
        baseline) and `lift_boost_floor_flexbase` (vs the flexible baseline),
        named so it can merge onto `run_sweep`'s output without colliding with
        any arm's columns.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(size=(len(matrix), NOISE_BLOCK_WIDTH))

    records: list[dict[str, float | str]] = []
    for target in targets:
        rows = matrix[target.column].notna().to_numpy()
        y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
        base_design = baseline.loc[rows].to_numpy(dtype="float64")
        flexible_design = flexible_baseline.loc[rows].to_numpy(dtype="float64")
        block = noise[rows]

        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        r2_baseline = float(r2_score(y, _baseline_oof_predictions(base_design, y, folds)))
        r2_flexible = float(
            r2_score(y, _baseline_oof_predictions(flexible_design, y, folds))
        )

        records.append(
            {
                "column": target.column,
                "lift_boost_floor": float(
                    r2_score(y, boost_residual_oof(base_design, block, y, folds))
                )
                - r2_baseline,
                "lift_boost_floor_flexbase": float(
                    r2_score(y, boost_residual_oof(flexible_design, block, y, folds))
                )
                - r2_flexible,
            }
        )
    return pd.DataFrame(records)


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


def _signed_rank(differences: pd.Series) -> dict[str, object]:
    """Wilcoxon signed-rank test of one series of paired differences against zero.

    Args:
        differences: Paired differences to test.

    Returns:
        Mean paired difference, win count and the Wilcoxon statistic/p-value.
    """
    statistic, p_value = wilcoxon(differences)
    return {
        "mean_paired_difference": float(differences.mean()),
        "n_wins": int((differences > 0).sum()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def _paired_test(results: pd.DataFrame, arm: Arm, column: str) -> dict[str, object]:
    """Test one arm's lift column against the baseline and, if paired, its arm.

    Fix-round finding: the previous version always reported `mean_lift` as the
    lift over the baseline, but for a paired arm its `wilcoxon_p` tested a
    different quantity -- the difference against the other arm, not against
    zero. Published side by side (e.g. `shape_v2`'s +0.00260 mean lift next to
    p=0.4515), a reader had no way to see that the p-value answered "does
    shape_v2 beat shape_v1?" rather than "does shape_v2 beat the baseline?" (the
    true against-baseline p was 0.0013). This version always runs the
    against-baseline test on the lift column itself, under `vs_baseline`, and --
    only when `arm.against` is set -- separately runs the against-arm test on
    the paired difference, under `vs_arm`, a name that cannot be mistaken for
    the baseline reading.

    Args:
        results: Output of `run_sweep`.
        arm: The arm to test. `arm.against` names the arm it is paired with, or
            None when the only comparison is the baseline.
        column: Full lift column name, carrying whichever suffixes apply.

    Returns:
        Mean and median lift over the baseline, a `vs_baseline` signed-rank
        reading of that same lift column and -- only when `arm.against` is set
        -- a `vs_arm` signed-rank reading of the paired difference against that
        arm's matching column, carrying its own `compared_against` name.
    """
    lifts = results[column]
    record: dict[str, object] = {
        "mean_lift": float(lifts.mean()),
        "median_lift": float(lifts.median()),
        "vs_baseline": _signed_rank(lifts),
    }
    if arm.against is not None:
        other_column = column.replace(arm.key, arm.against, 1)
        record["vs_arm"] = {
            "compared_against": arm.against,
            **_signed_rank(lifts - results[other_column]),
        }
    return record


def summarize(
    results: pd.DataFrame,
    size_recovery: dict[str, dict[str, float]],
    n_v1: int,
    n_profile: int,
) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    `results` must already carry `lift_boost_floor` and `lift_boost_floor_flexbase`
    -- `main` merges `boost_floor_lift`'s output onto `run_sweep`'s output by
    `column` before calling this. Every boost-learner arm entry gets a
    `vs_boost_floor` reading under each framing, so no boost lift is published
    without its floor visible in the same artifact (fix-round finding 2).

    Args:
        results: Output of `run_sweep`, merged with `boost_floor_lift`'s output.
        size_recovery: Output of `size_recoverability`.
        n_v1: Width of round one's block.
        n_profile: Width of the new shape profile.

    Returns:
        Target counts, block widths, the size diagnostic, the boost floor, and
        per-arm results in all three framings under both learners -- each
        carrying a `vs_baseline` reading, a `vs_arm` reading when paired, and
        (boost learner only) a `vs_boost_floor` reading.
    """
    arms: dict[str, object] = {}
    for arm in SHAPE_ARMS:
        for learner, suffix in (("ridge", ""), ("boost", BOOST_SUFFIX)):
            linear = _paired_test(results, arm, f"lift_{arm.key}{suffix}")
            flexible = _paired_test(
                results, arm, f"lift_{arm.key}{FLEXIBLE_SUFFIX}{suffix}"
            )
            if learner == "boost":
                linear["vs_boost_floor"] = _signed_rank(
                    results[f"lift_{arm.key}{suffix}"] - results["lift_boost_floor"]
                )
                flexible["vs_boost_floor"] = _signed_rank(
                    results[f"lift_{arm.key}{FLEXIBLE_SUFFIX}{suffix}"]
                    - results["lift_boost_floor_flexbase"]
                )
            arms[f"{arm.key}_{learner}"] = {
                "label": arm.label,
                "learner": learner,
                "mean_r2_alone": float(results[f"r2_alone_{arm.key}{suffix}"].mean()),
                "linear": linear,
                "flexible": flexible,
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
        "boost_floor": {
            "linear": {
                "mean_lift": float(results["lift_boost_floor"].mean()),
                **_signed_rank(results["lift_boost_floor"]),
            },
            "flexible": {
                "mean_lift": float(results["lift_boost_floor_flexbase"].mean()),
                **_signed_rank(results["lift_boost_floor_flexbase"]),
            },
        },
        "arms": arms,
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def _format_framing(framing: dict[str, object]) -> str:
    """Render one framing's lift and its test(s) for a single log line.

    Fix-round finding 1 was a mean printed beside a p-value that tested a
    different comparison. This renders the against-baseline p right next to the
    lift it actually measures, and adds the against-arm and against-floor
    readings -- when present -- each labeled with what it is against, so the
    log line cannot be misread the way the JSON's old flat fields were.

    Args:
        framing: One of `_paired_test`'s return values, as stored in
            `summarize`'s `arms` entries (optionally carrying `vs_boost_floor`).

    Returns:
        A compact string for one log line.
    """
    text = (
        f"{framing['mean_lift']:+.5f} "
        f"(vs_baseline p={framing['vs_baseline']['wilcoxon_p']:.4f}"
    )
    if "vs_arm" in framing:
        vs_arm = framing["vs_arm"]
        text += f", vs {vs_arm['compared_against']} p={vs_arm['wilcoxon_p']:.4f}"
    if "vs_boost_floor" in framing:
        vs_floor = framing["vs_boost_floor"]
        text += (
            f", vs floor Δ={vs_floor['mean_paired_difference']:+.5f}"
            f" p={vs_floor['wilcoxon_p']:.4f}"
        )
    return text + ")"


def main() -> None:
    """Attach both blocks, run the sweep and the diagnostics, write the artifacts."""
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

    # Fix-round finding 2: calibrate the boost lift scale with an
    # information-free block scored through the identical path, under the same
    # two baselines this sweep already built inside `run_sweep`. Rebuilding them
    # here is safe -- both are pure functions of `matrix` with no randomness --
    # and merging the floor onto `results` makes it a computed artifact sitting
    # beside the arms it calibrates, in the same CSV, rather than a number that
    # only exists in prose.
    baseline = build_baseline_design(matrix)
    flexible_baseline = build_flexible_baseline_design(matrix, baseline)
    floor = boost_floor_lift(matrix, baseline, flexible_baseline, targets)
    results = results.merge(floor, on="column", how="left", validate="one_to_one")
    if results[["lift_boost_floor", "lift_boost_floor_flexbase"]].isna().any().any():
        raise ValueError("boost floor is missing a target that the sweep scored")

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

    floor_stats = stats["boost_floor"]
    logger.info(
        "boost floor (information-free block, out-of-fold lift): "
        "linear %+.5f (p=%.4f) | flexible %+.5f (p=%.4f)",
        floor_stats["linear"]["mean_lift"],
        floor_stats["linear"]["wilcoxon_p"],
        floor_stats["flexible"]["mean_lift"],
        floor_stats["flexible"]["wilcoxon_p"],
    )

    for name, arm_stats in stats["arms"].items():
        logger.info(
            "%-26s alone %.4f | linear %s | flexible %s",
            name,
            arm_stats["mean_r2_alone"],
            _format_framing(arm_stats["linear"]),
            _format_framing(arm_stats["flexible"]),
        )

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
