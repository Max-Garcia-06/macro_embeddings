"""Does the shape of a Wikipedia article know anything county size does not?

`extract_source_a_structure_features.py` builds a block from section counts,
section lengths and section titles -- never from section text. This module
scores it against the same 28-target cross-pillar basket, the same folds and
the same protocol as `analyze_source_a_representation.py`, whose pipeline
helpers are imported rather than reimplemented so the numbers stay directly
comparable to that sweep's.

Four arms:

- `baseline`             -- size (`log_population`, `log_agi`, `log_gdp_latest`)
                            plus state fixed effects, and nothing else
- `structure`            -- baseline plus the structural block
- `typed`                -- baseline plus the shipped 29 typed columns
- `typed_plus_structure` -- baseline plus both

Two comparisons carry the round. `structure` against `baseline` asks what the
skeleton knows. `typed_plus_structure` against `typed` asks whether it knows
anything the shipped block does not already have, which is the fusion-relevant
question and the one most likely to come back at zero.

**The baseline is doing real work here, not decoration.** `n_body_sections`
correlates r = 0.550 against log tax returns and was cut from the scored matrix
for exactly that reason. Fitting each arm to the *residuals* of an unpenalized
size-plus-state model is what makes a pure size proxy worth approximately
nothing instead of worth a headline.

**Per-pillar is reported beside the aggregate, never instead of it.** Findings
§14.2b established that 20 of the 28 targets are one QCEW table, so a basket-wide
mean is 71% one pillar and reads as a breadth claim the basket does not support.

Run after `extract_source_a_structure_features.py`.

Outputs: `outputs/source_a_structure_scores.csv`,
`outputs/source_a_structure_by_pillar.csv`,
`analysis-output/source-a/source_a_structure_stats.json`.
"""

from __future__ import annotations

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
from pillar_matrix import build_matrix

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


def attach_structure(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge the structural block onto the pillar matrix.

    Args:
        matrix: Feature matrix from `build_matrix`.

    Returns:
        Tuple of (matrix with the structural columns attached, their names).

    Raises:
        FileNotFoundError: If the structural parquet is absent.
        ValueError: If a structural column name already exists in the matrix,
            which would rename both to `_x`/`_y` and score the wrong block.
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

    attached = matrix.merge(features, on="fips_code", how="left")
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

    Returns:
        Mapping of arm key to feature array.
    """
    typed = typed_columns()
    return {
        "structure": matrix.loc[rows, structure_cols].to_numpy(dtype="float64"),
        "typed": matrix.loc[rows, typed].to_numpy(dtype="float64"),
        "typed_plus_structure": matrix.loc[rows, [*typed, *structure_cols]].to_numpy(
            dtype="float64"
        ),
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
            "%s %-28s n=%4d  structure=%+.4f  typed=%+.4f  both=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["lift_structure"],
            record["lift_typed"],
            record["lift_typed_plus_structure"],
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


def summarize(results: pd.DataFrame, n_structure_features: int) -> dict[str, object]:
    """Assemble the stats artifact the notebook reads.

    Args:
        results: Output of `run_sweep`.
        n_structure_features: Width of the structural block.

    Returns:
        Target count, block widths, per-arm paired tests, and the per-pillar
        breakdown as records.
    """
    return {
        "n_targets": int(len(results)),
        "n_structure_features": n_structure_features,
        "n_typed_features": len(typed_columns()),
        "mean_r2_baseline": float(results["r2_baseline"].mean()),
        "arms": {arm.key: _paired_test(results, arm) for arm in ARMS},
        "by_pillar": summarize_by_pillar(results).to_dict(orient="records"),
    }


def main() -> None:
    """Attach the structural block, run the four arms, and write the artifacts."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    matrix, structure_cols = attach_structure(matrix)
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets: %d structural columns against %d shipped typed columns",
        len(targets),
        len(structure_cols),
        len(typed_columns()),
    )

    results = run_sweep(matrix, structure_cols, targets)
    pillar_results = summarize_by_pillar(results)
    stats = summarize(results, len(structure_cols))

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
            "  pillar %s  %2d targets  structure %+.5f | typed %+.5f | both %+.5f",
            row.pillar,
            row.n_targets,
            row.structure,
            row.typed,
            row.typed_plus_structure,
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

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
