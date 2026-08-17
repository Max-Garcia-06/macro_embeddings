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

Three representations of Source A, scored against the same five public ACS
targets, out-of-fold on held-out states, with the same restatement ablation:

- `typed` -- the 29 shipped columns. Reproduces the published -0.0000.
- `minilm_uniform` -- MiniLM over lead plus every non-narrative section,
  identically for every county, mean-pooled.
- `minilm_uniform_l2` -- the same vectors, row-normalized. This was the best arm
  in the representation sweep.

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
from sklearn.metrics import r2_score

from analyze_external_target import (
    EXTERNAL_TARGETS,
    TARGET_RESTATEMENTS,
    load_panel,
    out_of_fold_predictions,
)
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
        encoder harness already does for them.
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
    return {"minilm_uniform": vectors[rows], "minilm_uniform_l2": normed[rows]}


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
        designs = {
            "typed": np.hstack([size_and_others, usable[typed].astype(float).to_numpy()]),
        }
        for key, vectors in embeddings.items():
            designs[key] = np.hstack([size_and_others, vectors[mask]])

        for key, design in designs.items():
            full_r2 = float(r2_score(y, out_of_fold_predictions(design, y, groups)))
            rows.append(
                {
                    "target": target.column,
                    "label": target.label,
                    "representation": key,
                    "n": int(mask.sum()),
                    "n_columns": int(design.shape[1] - size_and_others.shape[1]),
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
