"""Does Source A's marginal contribution depend on how Source A is represented?

`analyze_external_target.py` measures what each pillar adds to a model that
already holds county size and the other five pillars. Source A scores
**-0.0000** there -- the only block negative in both the internal and the
external arm. That number was measured with Source A represented as its 29 typed
columns, because that is what the pillar ships.

`analyze_source_a_tiered_embedding.py` then found that a 384-dimension MiniLM
embedding of the full article text is a statistical tie with those typed columns
on cross-pillar lift (mean +0.00044 in the embedding's favour, median -0.00001,
14 of 28 targets, p = 0.76). Two representations that tie on one measurement need
not tie on another, and the drop-one is the measurement Source A's slot actually
depends on.

So this asks the narrow question directly: **is Source A's -0.0000 a fact about
the pillar, or about its representation?** The reduced model is identical in
every arm -- size plus the other five pillars -- so the only thing that varies is
what Source A contributes with.

Representations of Source A, scored against the same five public ACS
targets, out-of-fold on held-out states, with the same restatement ablation:

- `typed` -- the 29 shipped columns. Reproduces the published -0.0000.
- `typed_transformed` -- the 29 shipped columns plus the pre-registered
  capacity pass (`source_a_typed_transform.py`): log1p on count columns and a
  `sec_n_industry_mentions` x tier interaction. Both arms are scored under
  ridge, so 29 raw columns against 384 dense dimensions was not an
  equal-capacity comparison; this arm equalizes it without consulting any
  target's score to choose the transform.
- `minilm_uniform` -- MiniLM over lead plus every non-narrative section,
  identically for every county, mean-pooled.
- `minilm_uniform_l2` -- the same vectors, row-normalized. This was the best arm
  in the representation sweep.
- `minilm_uniform_pca29` / `minilm_uniform_pca64` -- the same vectors, reduced
  by PCA fitted inside each fold to 29 and 64 dimensions respectively, so the
  width-driven part of the embedding's measured penalty against the 29-column
  typed block is controlled for.

**A note on what a negative contribution means here.** Contribution is
R2(full) - R2(reduced), so a block that carries nothing useful lands near zero
and can go slightly negative through fold noise and the cost of extra columns.
The question is not whether the embedding makes Source A positive but whether it
moves it enough to change the conclusion that Source A adds nothing marginal.

Run after `ingest_source_a.py`, `extract_source_a_section_features.py` and
`analyze_external_target.py`. Read-only with respect to the shipped parquets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from analyze_external_target import (
    EXTERNAL_TARGETS,
    N_FOLDS,
    TARGET_RESTATEMENTS,
    _pipeline,
    load_panel,
    out_of_fold_predictions,
)
from analyze_pillar_matrix_signal import RANDOM_SEED
from analyze_source_a_tiered_embedding import (
    ENCODER_NAME,
    EMBEDDINGS_PARQUET_PATH,
    SECTIONS_PARQUET_PATH,
    TEXT_VARIANTS,
    build_variant_texts,
    encode_variant,
)
from analyze_source_a_tiers import assign_tiers
from pillar_matrix import SIZE_FEATURES
from source_a_typed_transform import transform_typed

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_representation_marginal.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_representation_marginal_stats.json"

# The text variant to encode. `uniform` reads the lead plus every non-narrative
# section identically for every county -- the arm that beat both tier-conditional
# rules once the chunk cap stopped truncating it.
VARIANT_KEY: str = "uniform"

logger = logging.getLogger(__name__)

# Embedding arms entering the marginal comparison, and the width each is reduced
# to. `None` means the native 384 dimensions. The 29-dimension arm exists so the
# comparison against the typed block is width-matched: findings §21.2 states that
# an unknown share of the embedding's penalty is width rather than content, and
# names this arm as the missing control.
EMBEDDING_ARMS: dict[str, int | None] = {
    "minilm_uniform": None,
    "minilm_uniform_l2": None,
    "minilm_uniform_pca29": 29,
    "minilm_uniform_pca64": 64,
}


def fit_reduction(train_vectors: np.ndarray, n_components: int) -> PCA:
    """Fit a PCA reduction on training rows only.

    Fitting on the full matrix would let the reduction see the held-out states
    the fold is scored on, which inflates the arm it is meant to control.

    Args:
        train_vectors: Rows in the fold's training split.
        n_components: Target width.

    Returns:
        The fitted reducer, ready to transform both splits.
    """
    reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reducer.fit(train_vectors)
    return reducer


def configure_logging() -> None:
    """Send INFO-level progress to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_source_a_embedding(fips_order: pd.Series) -> dict[str, np.ndarray]:
    """Encode Source A's text and return vectors aligned to `fips_order`.

    Args:
        fips_order: The `fips_code` sequence the panel is in.

    Returns:
        Mapping of representation key to a vector array whose rows align to
        `fips_order`. Counties with no text get a zero vector, which is what the
        encoder harness already does for them. The `_pca29`/`_pca64` keys carry
        the same unreduced 384-dimension vectors as `minilm_uniform` -- the
        reduction itself happens per fold in `out_of_fold_predictions_with_reduction`,
        never here, so it never sees held-out rows.
    """
    from sentence_transformers import SentenceTransformer

    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    matrix["tier"] = assign_tiers(matrix["content_length"])
    text = pd.read_parquet(EMBEDDINGS_PARQUET_PATH)[["fips_code", "embedding_text"]]
    matrix = matrix.merge(text, on="fips_code", how="left")
    sections = pd.read_parquet(SECTIONS_PARQUET_PATH)

    variant = next(v for v in TEXT_VARIANTS if v.key == VARIANT_KEY)
    logger.info("loading %s", ENCODER_NAME)
    model = SentenceTransformer(ENCODER_NAME)
    texts = build_variant_texts(matrix, sections, variant)
    vectors, diagnostics = encode_variant(model, texts, matrix["tier"])
    logger.info("%s: %s", variant.key, diagnostics)

    normed = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    # Reindex onto the panel's row order. `build_matrix` and the panel are both
    # keyed on fips_code but the panel is an inner join against the targets, so
    # it is a subset in its own order rather than the matrix's.
    by_fips = pd.DataFrame({"fips_code": matrix["fips_code"]})
    position = by_fips.reset_index().set_index("fips_code")["index"]
    rows = fips_order.map(position).to_numpy()
    missing = int(pd.isna(rows).sum())
    if missing:
        raise ValueError(f"{missing} panel counties absent from the encoded matrix")
    rows = rows.astype(int)
    base = vectors[rows]
    return {
        "minilm_uniform": base,
        "minilm_uniform_l2": normed[rows],
        "minilm_uniform_pca29": base,
        "minilm_uniform_pca64": base,
    }


def out_of_fold_predictions_with_reduction(
    size_and_others: np.ndarray,
    vectors: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Out-of-fold predictions with the embedding reduced inside each fold.

    Args:
        size_and_others: The reduced-model design, unreduced.
        vectors: Full-width embedding rows aligned to `size_and_others`.
        y: Target values.
        groups: State FIPS per row.
        n_components: Width to reduce the embedding to.

    Returns:
        Out-of-fold predictions, one per row.
    """
    predictions = np.zeros_like(y, dtype=float)
    splitter = GroupKFold(n_splits=N_FOLDS)
    for train_idx, test_idx in splitter.split(size_and_others, y, groups):
        reducer = fit_reduction(vectors[train_idx], n_components)
        design_train = np.hstack([size_and_others[train_idx], reducer.transform(vectors[train_idx])])
        design_test = np.hstack([size_and_others[test_idx], reducer.transform(vectors[test_idx])])
        model = _pipeline()
        model.fit(design_train, y[train_idx])
        predictions[test_idx] = model.predict(design_test)
    return predictions


def _ensure_tier_column(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach Source A's content-length tier to the panel if it lacks one.

    `load_panel` (`analyze_external_target.py`) builds the panel from
    `pillar_matrix.build_matrix`, which does not carry `tier` -- it is not a
    shipped feature, only a housekeeping cut on `content_length`.
    `transform_typed`'s industry-mentions interaction needs it.

    The panel is an inner join against the external targets, so it is a
    *subset* of the matrix, in the matrix's own row order rather than the
    panel's -- alignment must go through `fips_code`, never positional index.

    Args:
        panel: Joined panel from `load_panel`.

    Returns:
        `panel`, with a `tier` column added if it was missing.

    Raises:
        ValueError: If any panel county is absent from the tier assignment.
    """
    if "tier" in panel.columns:
        return panel

    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    tiers = pd.DataFrame(
        {"fips_code": matrix["fips_code"], "tier": assign_tiers(matrix["content_length"])}
    )
    merged = panel.merge(tiers, on="fips_code", how="left")
    missing = int(merged["tier"].isna().sum())
    if missing:
        raise ValueError(f"{missing} panel counties absent from the tier assignment")
    return merged


def score_representation(
    panel: pd.DataFrame,
    pillar_columns: list[str],
    a_columns: list[str],
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Score Source A's marginal contribution under each representation.

    For every target the reduced model is identical -- size plus the other five
    pillars -- so the contributions are directly comparable and differ only in
    what Source A brings.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        a_columns: Source A's column names within `pillar_columns`.
        embeddings: Representation key to vectors aligned to `panel`.

    Returns:
        One row per (target, representation).
    """
    panel = _ensure_tier_column(panel)
    other_columns = [column for column in pillar_columns if column not in a_columns]
    rows: list[dict[str, object]] = []

    for target in EXTERNAL_TARGETS:
        mask = panel[target.column].notna().to_numpy()
        usable = panel[mask].reset_index(drop=True)
        y = usable[target.column].astype(float).to_numpy()
        groups = usable["state_fips"].to_numpy()
        ablate = set(TARGET_RESTATEMENTS.get(target.column, ()))

        others = [column for column in other_columns if column not in ablate]
        size_and_others = usable[list(SIZE_FEATURES) + others].astype(float).to_numpy()
        reduced_r2 = float(
            r2_score(y, out_of_fold_predictions(size_and_others, y, groups))
        )

        typed = [column for column in a_columns if column not in ablate]
        transformed, _ = transform_typed(usable, typed, usable["tier"])
        designs = {
            "typed": np.hstack([size_and_others, usable[typed].astype(float).to_numpy()]),
            "typed_transformed": np.hstack([size_and_others, transformed]),
        }
        # Arms whose EMBEDDING_ARMS width is not None are reduced inside each
        # fold rather than assembled into a fixed design up front -- fitting
        # PCA on the full matrix would let it see the held-out states each fold
        # is scored on.
        reduced_predictions: dict[str, tuple[np.ndarray, int]] = {}
        for key, vectors in embeddings.items():
            width = EMBEDDING_ARMS.get(key)
            block = vectors[mask]
            if width is None:
                designs[key] = np.hstack([size_and_others, block])
            else:
                predicted = out_of_fold_predictions_with_reduction(
                    size_and_others, block, y, groups, width
                )
                reduced_predictions[key] = (predicted, width)

        representations: dict[str, tuple[np.ndarray, int]] = {
            key: (out_of_fold_predictions(design, y, groups), design.shape[1] - size_and_others.shape[1])
            for key, design in designs.items()
        }
        representations.update(reduced_predictions)

        for key, (predicted, n_columns) in representations.items():
            full_r2 = float(r2_score(y, predicted))
            rows.append(
                {
                    "target": target.column,
                    "label": target.label,
                    "representation": key,
                    "n": int(mask.sum()),
                    "n_columns": int(n_columns),
                    "r2_reduced": reduced_r2,
                    "r2_full": full_r2,
                    "contribution": full_r2 - reduced_r2,
                }
            )
            logger.info(
                "%-24s %-18s contribution %+.5f (full %.4f, reduced %.4f)",
                target.column,
                key,
                full_r2 - reduced_r2,
                full_r2,
                reduced_r2,
            )

    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame) -> dict[str, object]:
    """Collapse per-target contributions to one figure per representation.

    Args:
        scores: Output of `score_representation`.

    Returns:
        JSON-serializable summary.
    """
    by_representation: dict[str, object] = {}
    for key, group in scores.groupby("representation"):
        by_representation[str(key)] = {
            "mean_contribution": float(group["contribution"].mean()),
            "median_contribution": float(group["contribution"].median()),
            "n_positive": int((group["contribution"] > 0).sum()),
            "n_targets": int(len(group)),
            "n_columns": int(group["n_columns"].iloc[0]),
            "by_target": {
                str(row.target): float(row.contribution) for row in group.itertuples()
            },
        }
    return {
        "question": (
            "Is Source A's near-zero marginal contribution a fact about the "
            "pillar or about its representation?"
        ),
        "encoder": ENCODER_NAME,
        "text_variant": VARIANT_KEY,
        "n_targets": int(scores["target"].nunique()),
        "by_representation": by_representation,
    }


def main() -> None:
    """Score every Source A representation's marginal contribution."""
    configure_logging()
    panel, pillar_columns, blocks = load_panel()
    a_columns = list(blocks["A"])
    logger.info("Source A ships %d columns in the matrix", len(a_columns))

    embeddings = build_source_a_embedding(panel["fips_code"])
    scores = score_representation(panel, pillar_columns, a_columns, embeddings)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_CSV_PATH, index=False)
    stats = summarize(scores)
    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")

    logger.info("--- Source A marginal contribution by representation ---")
    for key, summary in stats["by_representation"].items():
        logger.info(
            "%-18s (%4d cols) mean %+.5f | median %+.5f | positive on %d/%d",
            key,
            summary["n_columns"],
            summary["mean_contribution"],
            summary["median_contribution"],
            summary["n_positive"],
            summary["n_targets"],
        )
    logger.info("wrote %s and %s", OUTPUT_CSV_PATH, OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
