"""Per-pillar marginal contribution: what does each block add that the rest do not?

`analyze_pillar_matrix_signal.py` asks whether the other five pillars together
predict one pillar's column. This asks the question `docs/pillar_status.md` puts
to Source F, and puts it to every pillar rather than only the one under
suspicion:

    for each target column t owned by pillar Q
        for each pillar P, P != Q
            reduced = size + state fixed effects + every block except Q and P
            full    = reduced + block P
            lift_P(t) = R2_out-of-fold(full) - R2_out-of-fold(reduced)

A positive lift means block P knows something about t that county size, state
geography, and the *other* three pillars do not already supply. That is a
strictly harder bar than the pairwise correlation Source F was judged on and
failed: F's collapse from r = 0.495 to r = -0.057 against Source D tonnage is a
statement about one pair of columns under a size control, and it says nothing
about whether the block explains residual variance once the rest of the matrix
is in the model. This measures that directly.

**What this arm cannot do is cut a pillar**, for the reason
`analyze_pillar_matrix_signal.py` already records: every target here is another
pillar's own feature, so low lift is equally consistent with "useless" and with
"genuinely independent information source", which is what a feature store wants.
Read it as corroboration for the external-target arm in
`analyze_external_target.py`, which scores the same drop-one design against five
public outcomes and is the arm the verdict rests on.

Every lift is reported twice, raw and with `RESTATEMENT_COLUMNS` removed from
the predictors. That ablation matters more here than anywhere else in the repo:
USDA derives Source F's industry-dependence flags from industry employment and
earnings shares, which is what Source B's location quotients measure, so an
unablated F lift against a Source B target would credit F for restating BLS.

The null is the same row shuffle the sibling script uses -- block P's rows are
permuted, breaking county alignment while preserving each column's marginal
distribution -- and one Benjamini-Hochberg correction runs across the whole
sweep.

Outputs: `outputs/pillar_block_marginal.csv` and
`analysis-output/cross-source/pillar_block_marginal_stats.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import (
    FDR_ALPHA,
    MIN_TARGET_OBSERVATIONS,
    N_FOLDS,
    RANDOM_SEED,
    RESTATEMENT_COLUMNS,
    Target,
    _oof_r2,
    _ridge_pipeline,
    _selected_alpha,
    build_baseline_design,
    build_targets,
    configure_logging,
)
from pillar_matrix import build_matrix
from stats_utils import benjamini_hochberg

# Fewer reps than the sibling sweep's 99. This design fits five blocks per
# target rather than one, so the same rep count would multiply a 15-minute run
# into an hour and a half for a null whose only job is to certify the sign of a
# lift. 49 gives a resolution of 0.02 on the p-value, which is enough at
# FDR_ALPHA = 0.05 for lifts that are not borderline, and borderline lifts are
# not what this arm is being asked to settle.
N_NULL_REPS: int = 49

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "pillar_block_marginal.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "pillar_block_marginal_stats.json"

logger = logging.getLogger(__name__)


def _context_columns(
    blocks: dict[str, list[str]], target_pillar: str, block_pillar: str
) -> list[str]:
    """List the predictor columns that sit in the reduced design.

    Args:
        blocks: Pillar-to-columns mapping from `build_matrix`.
        target_pillar: Pillar owning the target column, excluded to stop a block
            predicting itself.
        block_pillar: Pillar whose marginal contribution is being measured, also
            excluded -- it is what the full design adds back.

    Returns:
        Feature column names from the remaining pillars, in pillar order.
    """
    return [
        column
        for pillar, columns in blocks.items()
        if pillar not in (target_pillar, block_pillar)
        for column in columns
    ]


def score_block(
    matrix: pd.DataFrame,
    blocks: dict[str, list[str]],
    baseline: pd.DataFrame,
    target: Target,
    block_pillar: str,
) -> dict[str, float | str | int] | None:
    """Measure one pillar block's marginal lift on one target.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping.
        baseline: Size-plus-state design from `build_baseline_design`.
        target: The column to predict.
        block_pillar: The pillar whose block is added to the reduced design.

    Returns:
        Result record, or None if the target has too few observations.
    """
    rows = matrix[target.column].notna()
    n = int(rows.sum())
    if n < MIN_TARGET_OBSERVATIONS:
        return None

    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    context = _context_columns(blocks, target.pillar, block_pillar)
    added = blocks[block_pillar]

    base_design = baseline.loc[rows].to_numpy(dtype="float64")
    context_design = np.hstack(
        [base_design, matrix.loc[rows, context].to_numpy(dtype="float64")]
    )
    added_design = matrix.loc[rows, added].to_numpy(dtype="float64")
    full_design = np.hstack([context_design, added_design])

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_reduced = _oof_r2(context_design, y, _ridge_pipeline(), folds)
    r2_full = _oof_r2(full_design, y, _ridge_pipeline(), folds)
    lift = r2_full - r2_reduced

    # Definitional overlap removed from both sides. The reduced design has to
    # lose its restatements too, or the ablation would measure the restatement's
    # absence from the context rather than from the block under test.
    context_ablated = [col for col in context if col not in RESTATEMENT_COLUMNS]
    added_ablated = [col for col in added if col not in RESTATEMENT_COLUMNS]
    reduced_ablated_design = np.hstack(
        [base_design, matrix.loc[rows, context_ablated].to_numpy(dtype="float64")]
    )
    if added_ablated:
        full_ablated_design = np.hstack(
            [reduced_ablated_design, matrix.loc[rows, added_ablated].to_numpy(dtype="float64")]
        )
        lift_ablated = _oof_r2(full_ablated_design, y, _ridge_pipeline(), folds) - _oof_r2(
            reduced_ablated_design, y, _ridge_pipeline(), folds
        )
    else:
        lift_ablated = 0.0

    alpha = _selected_alpha(full_design, y)
    rng = np.random.default_rng(RANDOM_SEED)
    null_lifts = np.empty(N_NULL_REPS)
    for rep in range(N_NULL_REPS):
        shuffled = np.hstack([context_design, added_design[rng.permutation(n)]])
        null_lifts[rep] = _oof_r2(shuffled, y, _ridge_pipeline(alpha), folds) - r2_reduced

    p_value = (np.sum(null_lifts >= lift) + 1) / (N_NULL_REPS + 1)

    return {
        "target_pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "block": block_pillar,
        "n": n,
        "n_added_columns": len(added),
        "r2_reduced": r2_reduced,
        "r2_full": r2_full,
        "lift": lift,
        "lift_ablated": lift_ablated,
        "null_lift_mean": float(null_lifts.mean()),
        "null_lift_p95": float(np.percentile(null_lifts, 95)),
        "p": float(p_value),
    }


def run_sweep(
    matrix: pd.DataFrame, blocks: dict[str, list[str]], targets: list[Target]
) -> pd.DataFrame:
    """Score every (target, block) pair and apply one FDR correction.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping.
        targets: Targets to score.

    Returns:
        One row per scored pair, sorted by ablated lift descending.
    """
    baseline = build_baseline_design(matrix)
    records: list[dict[str, float | str | int]] = []
    for target in targets:
        for block_pillar in blocks:
            if block_pillar == target.pillar:
                continue
            record = score_block(matrix, blocks, baseline, target, block_pillar)
            if record is None:
                logger.warning("skipping %s: too few non-null rows", target.column)
                break
            records.append(record)
            logger.info(
                "%s -> %-26s  reduced R2=%+.3f  lift=%+.4f  ablated=%+.4f  p=%.3f",
                block_pillar,
                target.column,
                record["r2_reduced"],
                record["lift"],
                record["lift_ablated"],
                record["p"],
            )

    results = pd.DataFrame(records)
    results["q"] = benjamini_hochberg(results["p"].tolist())
    results["significant"] = results["q"] < FDR_ALPHA
    results["carries_signal"] = (
        results["significant"] & (results["lift"] > 0) & (results["lift_ablated"] > 0)
    )
    return results.sort_values("lift_ablated", ascending=False).reset_index(drop=True)


def summarize(results: pd.DataFrame) -> dict[str, object]:
    """Collapse per-pair results to the per-block verdict table.

    Args:
        results: Output of `run_sweep`.

    Returns:
        JSON-serializable summary, keyed by the block whose contribution is
        being measured rather than by the pillar owning the target.
    """
    by_block = results.groupby("block").agg(
        n_targets=("column", "size"),
        n_carrying_signal=("carries_signal", "sum"),
        n_positive_ablated=("lift_ablated", lambda s: int((s > 0).sum())),
        mean_lift=("lift", "mean"),
        mean_lift_ablated=("lift_ablated", "mean"),
        median_lift_ablated=("lift_ablated", "median"),
        max_lift_ablated=("lift_ablated", "max"),
    )
    return {
        "n_pairs": int(len(results)),
        "n_targets": int(results["column"].nunique()),
        "n_folds": N_FOLDS,
        "n_null_reps": N_NULL_REPS,
        "random_seed": RANDOM_SEED,
        "fdr_alpha": FDR_ALPHA,
        "by_block": json.loads(by_block.to_json(orient="index")),
        "best_pair": {
            "block": results.iloc[0]["block"],
            "column": results.iloc[0]["column"],
            "lift_ablated": float(results.iloc[0]["lift_ablated"]),
        },
    }


def main() -> None:
    """Build the matrix, run the drop-one sweep, and write CSV plus stats JSON."""
    configure_logging()

    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    targets = build_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets x %d blocks (minus self-pairs)", len(targets), len(blocks)
    )

    results = run_sweep(matrix, blocks, targets)
    stats = summarize(results)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)
    for block, row in sorted(stats["by_block"].items()):
        logger.info(
            "block %s: mean ablated lift %+.4f over %d targets, %d carrying signal",
            block,
            row["mean_lift_ablated"],
            row["n_targets"],
            row["n_carrying_signal"],
        )


if __name__ == "__main__":
    main()
