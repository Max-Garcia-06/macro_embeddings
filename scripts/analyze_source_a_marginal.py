"""Does Source A still earn its slot once a model already has the other pillars?

Two analyses in this repo disagree about Source A's typed features, and the
disagreement is structural rather than contradictory:

- `analyze_source_a_representation.py` scores Source A against a **size + state**
  baseline and finds its 29 typed columns worth +0.0032 mean R2 lift, beating
  both the shipped `content_length` scalar and the cut 1024-dim embedding.
- `analyze_pillar_matrix_signal.py` scores the **other five pillars together**
  and finds Source A's expansion adds nothing, slightly diluting the sweep.

Neither settles the fusion decision, because neither matches what a downstream
consumer experiences. The first uses a baseline far weaker than reality: a model
at Comcast will not have only county size and state, it will have every pillar
E_macro publishes. The second never isolates Source A at all -- it measures the
other five pillars as a bloc, and asks them to predict a pillar's own features,
where the target's sibling columns dominate by construction.

This script runs the missing configuration:

    baseline  = size + state + every pillar except Source A and the target's own
    variant   = baseline + Source A's block
    lift      = R2_out-of-fold(variant) - R2_out-of-fold(baseline)

That is the fusion-relevant question: **given everything else the feature store
already carries, does Source A add anything?** A feature block that lifts over
size and state but vanishes against a crowded baseline is carrying information
some other pillar already supplies more directly, and it is a liability in a
production feature store rather than an asset.

The target's own pillar is excluded from the baseline for the same reason
`analyze_pillar_matrix_signal.py` excludes it: predicting one QCEW sector from
nineteen other QCEW sectors is a within-pillar task that no outside source can
contribute to, and including it would measure Source A against a ceiling it
cannot reach for reasons unrelated to its content.

Design points carried over from the representation harness, for comparability:

- **The baseline is fitted separately from the block being tested**, and the
  block is fitted to the baseline's residuals. Stacking both under one shared
  ridge penalty forces a single penalty to serve controls and candidate alike,
  which is the artifact documented at length in that script's header.
- **The baseline here is penalized**, unlike that script's, because it now spans
  70+ pillar columns with real missingness rather than three size measures and
  state dummies. Median imputation carries a missingness indicator, since BLS
  suppression is itself informative.
- **Penalties are chosen by nested crossvalidation** inside each training fold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_pillar_matrix_signal import (
    N_FOLDS,
    RANDOM_SEED,
    Target,
    build_baseline_design,
)
from analyze_source_a_representation import RESIDUAL_ALPHAS, build_non_a_targets
from extract_source_a_features import VARIANT_COLUMNS
from paired_power import diagnostics
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_marginal.csv"
OUTPUT_PILLAR_CSV_PATH: Path = OUTPUTS_DIR / "source_a_marginal_by_pillar.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_marginal_stats.json"

# Penalty grid for the crowded baseline. Wider than the pillar sweep's because
# the design now carries every pillar's columns plus state dummies.
BASELINE_ALPHAS: tuple[float, ...] = tuple(10.0**k for k in range(-1, 5))

INNER_FOLDS: int = 5

# The two Source A representations worth comparing here: what the pipeline
# shipped before this experiment line, and what it ships now.
SCORED_VARIANTS: tuple[str, ...] = ("length", "extracted_sections")

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def _pipeline(alphas: tuple[float, ...]) -> Pipeline:
    """Build the impute-scale-ridge pipeline used for both stages.

    Args:
        alphas: Penalty grid handed to RidgeCV.

    Returns:
        Unfitted sklearn Pipeline. Median imputation adds a missingness
        indicator, because BLS suppresses roughly 35% of the location-quotient
        cells and "suppressed" is informative about county size and structure.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=alphas, cv=INNER_FOLDS)),
        ]
    )


def build_crowded_baseline(
    matrix: pd.DataFrame,
    blocks: dict[str, list[str]],
    thin_baseline: pd.DataFrame,
    target: Target,
) -> np.ndarray:
    """Assemble size, state, and every pillar except Source A and the target's own.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping.
        thin_baseline: Size-plus-state design from `build_baseline_design`.
        target: The column being predicted.

    Returns:
        Baseline design array for every row, including rows the target is null
        on; callers subset it.
    """
    columns = [
        column
        for pillar, pillar_columns in blocks.items()
        if pillar not in ("A", target.pillar)
        for column in pillar_columns
    ]
    return np.hstack(
        [
            thin_baseline.to_numpy(dtype="float64"),
            matrix[columns].to_numpy(dtype="float64"),
        ]
    )


def _oof_predictions(design: np.ndarray, y: np.ndarray, folds: KFold) -> np.ndarray:
    """Out-of-fold predictions of the baseline alone.

    Args:
        design: Baseline design array.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        One prediction per row.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(design):
        model = _pipeline(BASELINE_ALPHAS).fit(design[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(design[test_idx])
    return predictions


def _residual_oof_predictions(
    design: np.ndarray, block: np.ndarray, y: np.ndarray, folds: KFold
) -> np.ndarray:
    """Out-of-fold predictions of the baseline plus one Source A block.

    Args:
        design: Baseline design array.
        block: Source A representation for the same rows.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        One prediction per row.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(design):
        baseline = _pipeline(BASELINE_ALPHAS).fit(design[train_idx], y[train_idx])
        residuals = y[train_idx] - baseline.predict(design[train_idx])
        block_model = _pipeline(RESIDUAL_ALPHAS).fit(block[train_idx], residuals)
        predictions[test_idx] = baseline.predict(design[test_idx]) + block_model.predict(
            block[test_idx]
        )
    return predictions


def score_target(
    matrix: pd.DataFrame,
    blocks: dict[str, list[str]],
    thin_baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score Source A's variants against both a thin and a crowded baseline.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping.
        thin_baseline: Size-plus-state design.
        target: The column to predict.

    Returns:
        Result record carrying both baselines' R2 and each variant's lift over
        each, so the two framings can be compared row by row.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    thin_design = thin_baseline.loc[rows].to_numpy(dtype="float64")
    crowded_design = build_crowded_baseline(matrix, blocks, thin_baseline, target)[rows]

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_thin = float(r2_score(y, _oof_predictions(thin_design, y, folds)))
    r2_crowded = float(r2_score(y, _oof_predictions(crowded_design, y, folds)))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "n_baseline_columns": int(crowded_design.shape[1]),
        "r2_thin_baseline": r2_thin,
        "r2_crowded_baseline": r2_crowded,
    }

    for key in SCORED_VARIANTS:
        columns = ["content_length"] if key == "length" else list(VARIANT_COLUMNS[key])
        block = matrix.loc[rows, columns].to_numpy(dtype="float64")
        record[f"lift_thin_{key}"] = (
            float(r2_score(y, _residual_oof_predictions(thin_design, block, y, folds))) - r2_thin
        )
        record[f"lift_crowded_{key}"] = (
            float(r2_score(y, _residual_oof_predictions(crowded_design, block, y, folds)))
            - r2_crowded
        )

    return record


def run_sweep(
    matrix: pd.DataFrame, blocks: dict[str, list[str]], targets: list[Target]
) -> pd.DataFrame:
    """Score every target under both baselines.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping.
        targets: Targets to score.

    Returns:
        One row per target, sorted by crowded-baseline lift descending.
    """
    thin_baseline = build_baseline_design(matrix)
    records = []
    for target in targets:
        record = score_target(matrix, blocks, thin_baseline, target)
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  R2 thin=%.3f crowded=%.3f | A lift thin=%+.4f crowded=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["r2_thin_baseline"],
            record["r2_crowded_baseline"],
            record["lift_thin_extracted_sections"],
            record["lift_crowded_extracted_sections"],
        )
    return (
        pd.DataFrame(records)
        .sort_values("lift_crowded_extracted_sections", ascending=False)
        .reset_index(drop=True)
    )


def summarize_by_pillar(results: pd.DataFrame) -> pd.DataFrame:
    """Per-pillar lift and retention, which is the primary reportable result.

    Retention varies by an order of magnitude across the basket: Source C's
    velocity series keep most of their thin-baseline lift because no other
    pillar measures them, while Source E's capital-to-wage ratio keeps almost
    none because Source E measures it directly. Since 20 of the 28 targets are
    QCEW location quotients -- the worst-retaining large block -- the single
    aggregate figure is as much a property of the target mix as of Source A,
    and publishing it without this table is the most likely way to mislead a
    downstream reader.

    Args:
        results: Output of `run_sweep`.

    Returns:
        One row per target pillar with each variant's mean lift under both
        baselines, plus the share of thin-baseline lift that survives.
    """
    grouped = results.groupby("pillar")
    columns = {
        f"{stage}_{key}": grouped[f"lift_{stage}_{key}"].mean()
        for key in SCORED_VARIANTS
        for stage in ("thin", "crowded")
    }
    frame = pd.DataFrame({"n_targets": grouped.size(), **columns})
    for key in SCORED_VARIANTS:
        frame[f"retained_{key}"] = frame[f"crowded_{key}"] / frame[f"thin_{key}"]
    return frame.reset_index()


def summarize(results: pd.DataFrame) -> dict[str, object]:
    """Collapse to the numbers that decide whether Source A survives fusion.

    Args:
        results: Output of `run_sweep`.

    Returns:
        JSON-serializable summary, including how much of each variant's
        thin-baseline lift survives the crowded baseline.
    """
    summary: dict[str, object] = {
        "n_targets": int(len(results)),
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "mean_r2_thin_baseline": float(results["r2_thin_baseline"].mean()),
        "mean_r2_crowded_baseline": float(results["r2_crowded_baseline"].mean()),
        "mean_baseline_columns": float(results["n_baseline_columns"].mean()),
        "variants": {},
    }

    for key in SCORED_VARIANTS:
        thin = results[f"lift_thin_{key}"]
        crowded = results[f"lift_crowded_{key}"]
        statistic, p_value = wilcoxon(crowded)
        summary["variants"][key] = {
            "mean_lift_thin": float(thin.mean()),
            "mean_lift_crowded": float(crowded.mean()),
            "median_lift_crowded": float(crowded.median()),
            "share_retained": float(crowded.mean() / thin.mean()) if thin.mean() else float("nan"),
            "n_positive_crowded": int((crowded > 0).sum()),
            "wilcoxon_p_crowded": float(p_value),
            "wilcoxon_statistic_crowded": float(statistic),
            **diagnostics(crowded, results["pillar"]),
        }

    typed = results["lift_crowded_extracted_sections"]
    scalar = results["lift_crowded_length"]
    statistic, p_value = wilcoxon(typed - scalar)
    summary["typed_vs_scalar_crowded"] = {
        "mean_difference": float((typed - scalar).mean()),
        "n_wins": int((typed > scalar).sum()),
        "wilcoxon_p": float(p_value),
        "wilcoxon_statistic": float(statistic),
        **diagnostics(typed - scalar, results["pillar"]),
    }
    summary["by_pillar"] = json.loads(summarize_by_pillar(results).to_json(orient="records"))
    return summary


def main() -> None:
    """Run the marginal-value test and write the CSV plus stats JSON."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info(
        "scoring %d targets: does Source A add over a baseline that already has B-F?",
        len(targets),
    )

    results = run_sweep(matrix, blocks, targets)
    pillar_results = summarize_by_pillar(results)
    stats = summarize(results)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    pillar_results.to_csv(OUTPUT_PILLAR_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_PILLAR_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)
    logger.info(
        "baseline R2 %.3f (size+state) -> %.3f (plus B-F, %.0f columns)",
        stats["mean_r2_thin_baseline"],
        stats["mean_r2_crowded_baseline"],
        stats["mean_baseline_columns"],
    )

    # Per-pillar first. Retention swings from near-total to near-zero depending
    # on whether another pillar already measures the target, so the aggregate
    # underneath is meaningless without this.
    logger.info("per-pillar, extracted_sections (the primary breakout):")
    for row in pillar_results.itertuples():
        logger.info(
            "  pillar %s  %2d targets  thin %+.5f -> crowded %+.5f (%.0f%% retained)",
            row.pillar,
            row.n_targets,
            row.thin_extracted_sections,
            row.crowded_extracted_sections,
            100 * row.retained_extracted_sections,
        )

    for key in SCORED_VARIANTS:
        test = stats["variants"][key]
        logger.info(
            "%-20s lift %+.5f thin -> %+.5f crowded (%.0f%% retained) | positive on %2d/%d | p=%.4f",
            key,
            test["mean_lift_thin"],
            test["mean_lift_crowded"],
            100 * test["share_retained"],
            test["n_positive_crowded"],
            stats["n_targets"],
            test["wilcoxon_p_crowded"],
        )
        logger.info(
            "%-20s   dz=%.3f power=%.2f | effective n %.1f of %d (ICC %.3f), pillar-blocked p=%.3f",
            "",
            test["effect"]["dz"],
            test["effect"]["power"],
            test["clustering"]["n_effective"],
            test["clustering"]["n_nominal"],
            test["clustering"]["icc"],
            test["clustering"]["cluster_mean_p"],
        )

    versus = stats["typed_vs_scalar_crowded"]
    logger.info(
        "typed block beats content_length on %d/%d targets under the crowded baseline "
        "(mean diff %+.5f, p=%.4f) -- dz=%.3f, power %.2f, would need %d targets for 80%%",
        versus["n_wins"],
        stats["n_targets"],
        versus["mean_difference"],
        versus["wilcoxon_p"],
        versus["effect"]["dz"],
        versus["effect"]["power"],
        versus["effect"]["n_for_80"],
    )


if __name__ == "__main__":
    main()
