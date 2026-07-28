"""Head-to-head: does the bge-m3 embedding beat `content_length` as a predictor?

Source A's 1024-dim embedding was cut on three numbers -- Mantel r = 0.041
against economic distance, k-means silhouette 0.028, and `len(intro_text)`
outperforming it against metro status. Two of those route through Source C or
through a single bivariate correlation, and Section 8 of the key-findings
notebook showed Source C's velocity is the one target in the matrix that
nothing predicts. A verdict resting partly on a broken yardstick is worth
re-testing on a better one.

Better one: hold Source A fixed as the *predictor* and ask what it buys against
the rest of the matrix. For every continuous target in pillars B through F,

    baseline   = county size + state fixed effects
    + length   = baseline + content_length                    (1 feature)
    + pca50    = baseline + 50 principal components of the embedding
    + full     = baseline + all 1024 embedding dimensions

and compare out-of-fold R2 lift over the baseline. This is the same protocol as
`analyze_pillar_matrix_signal.py` with the predictor side narrowed to Source A
alone, so the numbers are directly comparable to that sweep's.

Four design points worth stating, because they decide whether the answer is
trustworthy. The third one reversed this script's result once already:

- **PCA is fitted inside each fold.** Fitting it on the full matrix first would
  let the held-out rows shape the components that predict them.
- **The comparison is paired.** Every variant sees identical folds and
  identical rows for a given target, so per-target differences are meaningful
  and a Wilcoxon signed-rank test across targets is the headline statistic
  rather than any single row.
- **The baseline is never penalized.** Stacking 1024 embedding dimensions
  beside the controls and fitting one ridge over the whole design forces a
  single penalty to serve both, and the penalty large enough to tame 1024
  correlated dimensions also crushes the size and state controls that carry
  most of the fit. That design reports large negative lifts for the embedding
  on every target -- an artifact of the shared penalty, not a property of the
  embedding. Here the baseline is fitted unpenalized, and each Source A
  representation is fitted to the baseline's *residuals*, so a representation
  that knows nothing costs approximately nothing instead of dragging the
  controls down with it.
- **The ridge penalty is chosen by nested crossvalidation.** An inner split of
  each training fold selects it, so no variant is scored at a penalty picked by
  looking at the rows it is being scored on. Selecting the penalty on the
  out-of-fold metric instead would flatter the 1024-dim variant most, since it
  has the most to gain from a lucky choice.

`content_length` sits inside the embedding by construction -- longer text moves
the vector -- so the interesting quantity is not whether the embedding knows
length. It is whether it knows anything *else* that the other pillars care
about.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_pillar_matrix_signal import (
    N_FOLDS,
    RANDOM_SEED,
    STATIC_TARGETS,
    Target,
    build_baseline_design,
)
from pillar_matrix import DATA_DIR, build_matrix

# Principal components retained for the reduced variant. 50 keeps roughly the
# same order of magnitude as Source B's 20-column block, so "the embedding" is
# not judged solely at a width no other pillar is asked to compete at.
N_COMPONENTS: int = 50

# Penalty grid for the residual model. It runs far higher than the pillar-matrix
# sweep's grid because 1024 correlated dimensions against ~2,500 rows need it;
# the scalar variant lands at the low end and is unaffected by the extra range.
RESIDUAL_ALPHAS: tuple[float, ...] = tuple(10.0**k for k in range(0, 7))

# Inner splits used to select that penalty inside each training fold.
INNER_FOLDS: int = 5

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_representation.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_representation_stats.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    """One Source A representation being scored.

    Attributes:
        key: Short identifier used as the result column suffix.
        label: Human-readable description used in reports.
        n_features: Width of the representation, for the cost comparison.
    """

    key: str
    label: str
    n_features: int


VARIANTS: tuple[Variant, ...] = (
    Variant("length", "content_length (scalar)", 1),
    Variant("pca50", f"bge-m3, {N_COMPONENTS} principal components", N_COMPONENTS),
    Variant("full", "bge-m3, all 1024 dimensions", 1024),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_embeddings(fips_order: pd.Series) -> np.ndarray:
    """Load the legacy bge-m3 embeddings aligned to the matrix's row order.

    Args:
        fips_order: `fips_code` column of the assembled matrix, defining row
            order.

    Returns:
        Array of shape (len(fips_order), 1024). Counties absent from the
        embeddings parquet get a row of NaN.

    Raises:
        FileNotFoundError: If the embeddings parquet is absent.
    """
    path = DATA_DIR / "source_a_embeddings.parquet"
    try:
        frame = pd.read_parquet(path, columns=["fips_code", "embedding"])
    except FileNotFoundError:
        logger.error("Missing embeddings parquet: %s", path)
        raise

    lookup = dict(zip(frame["fips_code"], frame["embedding"]))
    missing = 1024 * [np.nan]
    rows = [np.asarray(lookup.get(fips, missing), dtype="float64") for fips in fips_order]
    matrix = np.vstack(rows)
    n_missing = int(np.isnan(matrix[:, 0]).sum())
    if n_missing:
        logger.warning("%d counties have no embedding row", n_missing)
    return matrix


def build_non_a_targets(blocks: dict[str, list[str]], naics_labels: dict[str, str]) -> list[Target]:
    """List every continuous target outside Source A.

    Args:
        blocks: Pillar-to-columns mapping from `build_matrix`.
        naics_labels: NAICS 2-digit code to sector name.

    Returns:
        Targets in pillars B through F.
    """
    lq_targets = [
        Target("B", col, f"{naics_labels.get(col.replace('lq_emp_', ''), col)} LQ")
        for col in blocks["B"]
        if col.startswith("lq_emp_")
    ]
    return lq_targets + [t for t in STATIC_TARGETS if t.pillar not in ("A", "B")]


def _baseline_pipeline() -> Pipeline:
    """Build the unpenalized control model.

    Ordinary least squares, not ridge: the controls are three size measures and
    ~50 state dummies against thousands of rows, so there is nothing to
    regularize, and leaving them unpenalized is what keeps a wide Source A
    representation from degrading them.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _residual_pipeline(n_components: int | None) -> Pipeline:
    """Build the impute-scale-(reduce)-ridge pipeline fitted to residuals.

    Args:
        n_components: PCA components to retain, or None to skip reduction.

    Returns:
        Unfitted sklearn Pipeline whose ridge penalty is selected by an inner
        crossvalidation over RESIDUAL_ALPHAS.
    """
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)))
    steps.append(("model", RidgeCV(alphas=RESIDUAL_ALPHAS, cv=INNER_FOLDS)))
    return Pipeline(steps)


def _baseline_oof_r2(base_design: np.ndarray, y: np.ndarray, folds: KFold) -> float:
    """Out-of-fold R2 of the controls alone.

    Args:
        base_design: Size-plus-state control array.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        R2 over the concatenated out-of-fold predictions.
    """
    return float(r2_score(y, cross_val_predict(_baseline_pipeline(), base_design, y, cv=folds)))


def _residual_oof_r2(
    base_design: np.ndarray,
    block: np.ndarray,
    y: np.ndarray,
    folds: KFold,
    n_components: int | None,
) -> float:
    """Out-of-fold R2 of controls plus one Source A representation.

    Within each fold the controls are fitted on the training rows, their
    residuals become the target for the ridge, and the two predictions are
    summed on the held-out rows. The controls therefore never see the ridge
    penalty, and a representation carrying no information shrinks toward zero
    rather than displacing them.

    Args:
        base_design: Size-plus-state control array.
        block: Source A representation for the same rows.
        y: Target vector.
        folds: Crossvalidation splitter.
        n_components: PCA components to retain, or None to skip reduction.

    Returns:
        R2 over the concatenated out-of-fold predictions.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(base_design):
        controls = _baseline_pipeline().fit(base_design[train_idx], y[train_idx])
        residuals = y[train_idx] - controls.predict(base_design[train_idx])
        residual_model = _residual_pipeline(n_components).fit(block[train_idx], residuals)
        predictions[test_idx] = controls.predict(base_design[test_idx]) + residual_model.predict(
            block[test_idx]
        )
    return float(r2_score(y, predictions))


def score_target(
    matrix: pd.DataFrame,
    embeddings: np.ndarray,
    baseline: pd.DataFrame,
    target: Target,
) -> dict[str, float | str | int]:
    """Score every Source A variant against one target.

    Each representation is fitted to the controls' residuals, so the size and
    state terms are identical across all three variants and none of them can
    degrade the controls.

    Args:
        matrix: Feature matrix from `build_matrix`.
        embeddings: Aligned embedding array.
        baseline: Size-plus-state design.
        target: The column to predict.

    Returns:
        Result record with one lift column per variant.
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")
    length = matrix.loc[rows, ["content_length"]].to_numpy(dtype="float64")
    embedding = embeddings[rows]

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_baseline = _baseline_oof_r2(base_design, y, folds)

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
    }

    blocks = {"length": (length, None), "pca50": (embedding, N_COMPONENTS), "full": (embedding, None)}
    for variant in VARIANTS:
        block, n_components = blocks[variant.key]
        record[f"lift_{variant.key}"] = (
            _residual_oof_r2(base_design, block, y, folds, n_components) - r2_baseline
        )

    return record


def run_sweep(
    matrix: pd.DataFrame, embeddings: np.ndarray, targets: list[Target]
) -> pd.DataFrame:
    """Score every target against every Source A variant.

    Args:
        matrix: Feature matrix from `build_matrix`.
        embeddings: Aligned embedding array.
        targets: Targets to score.

    Returns:
        One row per target, sorted by the best embedding lift descending.
    """
    baseline = build_baseline_design(matrix)
    records = []
    for target in targets:
        record = score_target(matrix, embeddings, baseline, target)
        records.append(record)
        logger.info(
            "%s %-28s n=%4d  length=%+.4f  pca50=%+.4f  full=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["lift_length"],
            record["lift_pca50"],
            record["lift_full"],
        )

    results = pd.DataFrame(records)
    results["best_embedding_lift"] = results[["lift_pca50", "lift_full"]].max(axis=1)
    results["embedding_beats_length"] = results["best_embedding_lift"] > results["lift_length"]
    return results.sort_values("best_embedding_lift", ascending=False).reset_index(drop=True)


def summarize(results: pd.DataFrame) -> dict[str, object]:
    """Run the paired comparison and collapse results to reportable numbers.

    Args:
        results: Output of `run_sweep`.

    Returns:
        JSON-serializable summary including the Wilcoxon signed-rank test of
        embedding lift against `content_length` lift across targets.
    """
    differences = results["best_embedding_lift"] - results["lift_length"]
    statistic, p_value = wilcoxon(differences)
    return {
        "n_targets": int(len(results)),
        "n_folds": N_FOLDS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "mean_lift_length": float(results["lift_length"].mean()),
        "mean_lift_pca50": float(results["lift_pca50"].mean()),
        "mean_lift_full": float(results["lift_full"].mean()),
        "mean_lift_best_embedding": float(results["best_embedding_lift"].mean()),
        "median_lift_length": float(results["lift_length"].median()),
        "median_lift_best_embedding": float(results["best_embedding_lift"].median()),
        "n_embedding_wins": int(results["embedding_beats_length"].sum()),
        "mean_difference": float(differences.mean()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
        "best_target": results.iloc[0]["column"],
        "best_embedding_lift": float(results.iloc[0]["best_embedding_lift"]),
    }


def main() -> None:
    """Build the matrix, run the head-to-head, and write CSV plus stats JSON."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    embeddings = load_embeddings(matrix["fips_code"])
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info("scoring %d non-Source-A targets against %d variants", len(targets), len(VARIANTS))

    results = run_sweep(matrix, embeddings, targets)
    stats = summarize(results)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)
    logger.info(
        "embedding beats content_length on %d of %d targets | mean lift %+.4f (best "
        "embedding) vs %+.4f (length) | Wilcoxon p = %.4f",
        stats["n_embedding_wins"],
        stats["n_targets"],
        stats["mean_lift_best_embedding"],
        stats["mean_lift_length"],
        stats["wilcoxon_p"],
    )


if __name__ == "__main__":
    main()
