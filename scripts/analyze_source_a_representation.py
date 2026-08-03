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

**Extended 2026-08-03** with the typed features from `extract_source_a_features.py`,
which are the actual candidate replacement for the cut embedding: they cost one
regex pass rather than a 2.2GB model, and they stay interpretable at the feature
store. Three nested widths are scored (`extracted_min` / `_mid` / `_full`) so the
reduction question is answered as "which columns earn their slot" rather than "how
many principal components" -- PCA on this corpus is a measured dead end, retaining
only 42% of the full embedding's advantage, because its highest-variance direction
is the Texas founding-narrative artifact documented in §3.2 of the findings.

Two reporting additions, both requested as decision inputs for the open question of
whether county size is a control or part of the target:

- **Raw alongside size-controlled.** Every variant is additionally scored alone,
  with no size or state controls, so the raw explanatory power is visible next to
  the controlled lift rather than inferred from it.
- **Lift broken out per content tier.** Source A's corpus is extremely uneven --
  25.2% of the richest quartile names an industry against 1.1% of the thin tier --
  so a mean lift across all counties can hide a gain that exists only where there
  was text to read. The breakout reuses the same out-of-fold predictions rather
  than refitting per tier, so it adds no model and no extra penalty selection.
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
from analyze_source_a_tiers import TIER_LABELS, assign_tiers
from extract_source_a_features import VARIANT_COLUMNS
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
OUTPUT_TIER_CSV_PATH: Path = OUTPUTS_DIR / "source_a_representation_by_tier.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_representation_stats.json"

# Minimum rows a tier must contribute to a target before its lift is reported.
# Below this an out-of-fold R2 on a subset is dominated by which rows happened to
# land in it, and reporting it would invite reading noise as a tier effect.
MIN_TIER_OBSERVATIONS: int = 150

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
    Variant("extracted_min", "typed features, 4 columns", len(VARIANT_COLUMNS["extracted_min"])),
    Variant("extracted_mid", "typed features, 8 columns", len(VARIANT_COLUMNS["extracted_mid"])),
    Variant("extracted_full", "typed features, all columns", len(VARIANT_COLUMNS["extracted_full"])),
    Variant("pca50", f"bge-m3, {N_COMPONENTS} principal components", N_COMPONENTS),
    Variant("full", "bge-m3, all 1024 dimensions", 1024),
)

# The variant the pillar is judged against. `content_length` is what Source A
# currently ships, so beating it is the whole question.
INCUMBENT: str = "length"

# Variants built from the typed extraction rather than the embedding. Kept
# separate because the headline comparison is extraction against the incumbent;
# the embedding columns are retained for reference at a cost the pillar no longer
# pays.
EXTRACTED_KEYS: tuple[str, ...] = ("extracted_min", "extracted_mid", "extracted_full")


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


def _baseline_oof_predictions(base_design: np.ndarray, y: np.ndarray, folds: KFold) -> np.ndarray:
    """Out-of-fold predictions from the controls alone.

    Predictions rather than a score, because the per-tier breakout needs to
    re-evaluate the same predictions on row subsets. Refitting per tier would
    change both the training set and the selected penalty, which would confound
    "this tier is more predictable" with "this tier got its own model."

    Args:
        base_design: Size-plus-state control array.
        y: Target vector.
        folds: Crossvalidation splitter.

    Returns:
        Out-of-fold prediction per row.
    """
    return cross_val_predict(_baseline_pipeline(), base_design, y, cv=folds)


def _residual_oof_predictions(
    base_design: np.ndarray,
    block: np.ndarray,
    y: np.ndarray,
    folds: KFold,
    n_components: int | None,
) -> np.ndarray:
    """Out-of-fold predictions from controls plus one Source A representation.

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
        Out-of-fold prediction per row.
    """
    predictions = np.empty(len(y))
    for train_idx, test_idx in folds.split(base_design):
        controls = _baseline_pipeline().fit(base_design[train_idx], y[train_idx])
        residuals = y[train_idx] - controls.predict(base_design[train_idx])
        residual_model = _residual_pipeline(n_components).fit(block[train_idx], residuals)
        predictions[test_idx] = controls.predict(base_design[test_idx]) + residual_model.predict(
            block[test_idx]
        )
    return predictions


def _alone_oof_r2(block: np.ndarray, y: np.ndarray, folds: KFold, n_components: int | None) -> float:
    """Out-of-fold R2 of one representation with no size or state controls.

    This is the "raw" number: what Source A explains before any confound is
    removed. It is reported beside the controlled lift because whether county
    size is a control or part of the target is still an open decision for this
    project, and the two framings give very different pictures of the same
    pillar.

    Args:
        block: Source A representation.
        y: Target vector.
        folds: Crossvalidation splitter.
        n_components: PCA components to retain, or None to skip reduction.

    Returns:
        R2 over the concatenated out-of-fold predictions.
    """
    return float(r2_score(y, cross_val_predict(_residual_pipeline(n_components), block, y, cv=folds)))


def build_variant_blocks(
    matrix: pd.DataFrame, embeddings: np.ndarray, rows: np.ndarray
) -> dict[str, tuple[np.ndarray, int | None]]:
    """Assemble every variant's feature array for one target's usable rows.

    Args:
        matrix: Feature matrix from `build_matrix`.
        embeddings: Aligned embedding array.
        rows: Boolean mask of rows where the target is observed.

    Returns:
        Mapping of variant key to (feature array, PCA components or None).
    """
    embedding = embeddings[rows]
    blocks: dict[str, tuple[np.ndarray, int | None]] = {
        "length": (matrix.loc[rows, ["content_length"]].to_numpy(dtype="float64"), None),
        "pca50": (embedding, N_COMPONENTS),
        "full": (embedding, None),
    }
    for key in EXTRACTED_KEYS:
        columns = list(VARIANT_COLUMNS[key])
        blocks[key] = (matrix.loc[rows, columns].to_numpy(dtype="float64"), None)
    return blocks


def score_target(
    matrix: pd.DataFrame,
    embeddings: np.ndarray,
    baseline: pd.DataFrame,
    tiers: pd.Series,
    target: Target,
) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    """Score every Source A variant against one target, overall and per tier.

    Each representation is fitted to the controls' residuals, so the size and
    state terms are identical across variants and none of them can degrade the
    controls. Tier lifts re-score the same out-of-fold predictions on row
    subsets, so every tier is judged by a model trained on the whole corpus.

    Args:
        matrix: Feature matrix from `build_matrix`.
        embeddings: Aligned embedding array.
        baseline: Size-plus-state design.
        tiers: Content tier per county, aligned to `matrix`.
        target: The column to predict.

    Returns:
        Tuple of (overall record, per-tier records).
    """
    rows = matrix[target.column].notna().to_numpy()
    y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
    base_design = baseline.loc[rows].to_numpy(dtype="float64")
    row_tiers = tiers[rows].to_numpy()

    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    baseline_predictions = _baseline_oof_predictions(base_design, y, folds)
    r2_baseline = float(r2_score(y, baseline_predictions))

    record: dict[str, float | str | int] = {
        "pillar": target.pillar,
        "column": target.column,
        "label": target.label,
        "n": int(rows.sum()),
        "r2_baseline": r2_baseline,
    }
    tier_records: list[dict[str, float | str | int]] = []

    blocks = build_variant_blocks(matrix, embeddings, rows)
    for variant in VARIANTS:
        block, n_components = blocks[variant.key]
        predictions = _residual_oof_predictions(base_design, block, y, folds, n_components)
        record[f"lift_{variant.key}"] = float(r2_score(y, predictions)) - r2_baseline
        record[f"r2_alone_{variant.key}"] = _alone_oof_r2(block, y, folds, n_components)

        for tier in TIER_LABELS:
            mask = row_tiers == tier
            if mask.sum() < MIN_TIER_OBSERVATIONS:
                continue
            tier_records.append(
                {
                    "pillar": target.pillar,
                    "column": target.column,
                    "variant": variant.key,
                    "tier": tier,
                    "n": int(mask.sum()),
                    "r2_baseline": float(r2_score(y[mask], baseline_predictions[mask])),
                    "lift": float(r2_score(y[mask], predictions[mask]))
                    - float(r2_score(y[mask], baseline_predictions[mask])),
                }
            )

    return record, tier_records


def run_sweep(
    matrix: pd.DataFrame, embeddings: np.ndarray, tiers: pd.Series, targets: list[Target]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every target against every Source A variant.

    Args:
        matrix: Feature matrix from `build_matrix`.
        embeddings: Aligned embedding array.
        tiers: Content tier per county, aligned to `matrix`.
        targets: Targets to score.

    Returns:
        Tuple of (per-target results sorted by best extracted lift, per-tier
        results in long format).
    """
    baseline = build_baseline_design(matrix)
    records = []
    tier_records: list[dict[str, float | str | int]] = []
    for target in targets:
        record, target_tier_records = score_target(matrix, embeddings, baseline, tiers, target)
        records.append(record)
        tier_records.extend(target_tier_records)
        logger.info(
            "%s %-28s n=%4d  length=%+.4f  ext_min=%+.4f  ext_full=%+.4f  emb_full=%+.4f",
            record["pillar"],
            record["column"],
            record["n"],
            record["lift_length"],
            record["lift_extracted_min"],
            record["lift_extracted_full"],
            record["lift_full"],
        )

    results = pd.DataFrame(records)
    results["best_embedding_lift"] = results[["lift_pca50", "lift_full"]].max(axis=1)
    results["embedding_beats_length"] = results["best_embedding_lift"] > results["lift_length"]
    results["best_extracted_lift"] = results[[f"lift_{k}" for k in EXTRACTED_KEYS]].max(axis=1)
    results["extracted_beats_length"] = results["best_extracted_lift"] > results["lift_length"]
    results = results.sort_values("best_extracted_lift", ascending=False).reset_index(drop=True)
    return results, pd.DataFrame(tier_records)


def _paired_test(results: pd.DataFrame, key: str) -> dict[str, float | int]:
    """Compare one variant against the incumbent across every target.

    The paired test across targets is the headline rather than any single row:
    with 28 targets and six variants, some row wins by chance, and the question
    is whether a representation is better in general.

    Args:
        results: Output of `run_sweep`.
        key: Variant key to test.

    Returns:
        Mean lift, mean paired difference, win count, and the Wilcoxon
        signed-rank p-value against the incumbent.
    """
    differences = results[f"lift_{key}"] - results[f"lift_{INCUMBENT}"]
    # The incumbent against itself is all zeros, which Wilcoxon cannot rank.
    statistic, p_value = (float("nan"), float("nan")) if key == INCUMBENT else wilcoxon(differences)
    return {
        "mean_lift": float(results[f"lift_{key}"].mean()),
        "median_lift": float(results[f"lift_{key}"].median()),
        "mean_r2_alone": float(results[f"r2_alone_{key}"].mean()),
        "n_wins_vs_incumbent": int((differences > 0).sum()),
        "mean_difference": float(differences.mean()),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def summarize(results: pd.DataFrame, tier_results: pd.DataFrame) -> dict[str, object]:
    """Run the paired comparisons and collapse results to reportable numbers.

    Args:
        results: Per-target output of `run_sweep`.
        tier_results: Per-tier output of `run_sweep`.

    Returns:
        JSON-serializable summary. Every variant is tested against the incumbent
        `content_length`; the embedding keys are retained so the numbers stay
        comparable to the 2026-07-27 run that decided the cut.
    """
    differences = results["best_embedding_lift"] - results["lift_length"]
    statistic, p_value = wilcoxon(differences)

    tier_means = (
        tier_results.groupby(["variant", "tier"], observed=True)["lift"].mean().unstack()
        if len(tier_results)
        else pd.DataFrame()
    )

    return {
        "n_targets": int(len(results)),
        "n_folds": N_FOLDS,
        "n_components": N_COMPONENTS,
        "random_seed": RANDOM_SEED,
        "incumbent": INCUMBENT,
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
        "best_embedding_lift": float(results["best_embedding_lift"].max()),
        "mean_lift_best_extracted": float(results["best_extracted_lift"].mean()),
        "n_extracted_wins": int(results["extracted_beats_length"].sum()),
        "variants": {variant.key: _paired_test(results, variant.key) for variant in VARIANTS},
        "mean_lift_by_tier": json.loads(tier_means.to_json(orient="index")) if len(tier_means) else {},
    }


def main() -> None:
    """Build the matrix, run the head-to-head, and write CSV plus stats JSON."""
    configure_logging()

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, blocks = build_matrix()
    if "n_industry_mentions" not in matrix.columns:
        raise ValueError("Extracted columns absent -- run extract_source_a_features.py first.")

    embeddings = load_embeddings(matrix["fips_code"])
    tiers = assign_tiers(matrix["content_length"])
    targets = build_non_a_targets(blocks, NAICS2_LABELS)
    logger.info("scoring %d non-Source-A targets against %d variants", len(targets), len(VARIANTS))

    results, tier_results = run_sweep(matrix, embeddings, tiers, targets)
    stats = summarize(results, tier_results)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    tier_results.to_csv(OUTPUT_TIER_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))

    logger.info("wrote %s", OUTPUT_CSV_PATH)
    logger.info("wrote %s", OUTPUT_TIER_CSV_PATH)
    logger.info("wrote %s", OUTPUT_STATS_PATH)
    for variant in VARIANTS:
        test = stats["variants"][variant.key]
        logger.info(
            "%-15s (%4d cols) mean lift %+.5f | raw R2 alone %.4f | beats %s on %2d/%d | p=%.4f",
            variant.key,
            variant.n_features,
            test["mean_lift"],
            test["mean_r2_alone"],
            INCUMBENT,
            test["n_wins_vs_incumbent"],
            stats["n_targets"],
            test["wilcoxon_p"],
        )


if __name__ == "__main__":
    main()
