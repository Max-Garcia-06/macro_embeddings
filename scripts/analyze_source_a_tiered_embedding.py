"""Score a small embedding whose input text does -- or does not -- depend on content tier.

Two questions in one harness, because they are cheap to answer together and
easy to confuse:

1. **Does a smaller encoder work here?** The cut embedding was bge-m3 at 1024
   dimensions. This uses `all-MiniLM-L6-v2` at 384, which is a natively small
   model rather than a projection of a large one -- a distinction the earlier
   PCA-50 result (+0.00164 against the full 1024-d's +0.00281) makes worth
   drawing.
2. **Should the text fed to the encoder depend on the tier?** §15 asked the
   analogous question of the *model* and answered no. This asks it of the
   *input*, and answers it by measuring rather than by argument.

Three text variants, one encoder, same 28 targets and protocol as
`analyze_source_a_representation.py`:

- `lead_only` -- the cleaned lead section, which is what the retired embedding
  encoded. The control: it isolates the change of model and width from any
  change in what is read.
- `uniform` -- lead plus every non-narrative section, identically for every
  county.
- `tier_conditional` -- the requested design. Depth is spent where the lead says
  least: stub and thin counties get lead plus non-narrative sections, mid
  counties get lead plus economy-titled sections, rich counties get the lead
  alone.

**The objection this is built to test, stated plainly.** An embedding's value is
that it is a shared metric space: two counties are comparable because their
coordinates mean the same thing. Under `tier_conditional` they do not -- a stub
county's vector summarizes its whole article and a rich county's summarizes one
paragraph -- and tier tracks county size (`content_length` r = 0.359,
`n_body_sections` r = 0.550), so the difference in meaning is correlated with
population. Where typed flags would corrupt one boolean, this corrupts all 384
dimensions at once. That is an argument, not a measurement, which is why
`uniform` is scored beside it at identical width.

Long inputs are chunked rather than truncated: MiniLM's window is 256 tokens, so
a rich county's article would otherwise lose most of itself silently. Chunks are
mean-pooled per county, and the per-county chunk cap is logged rather than
applied quietly.

Run after `ingest_source_a.py` and `extract_source_a_section_features.py`.
Read-only with respect to the shipped parquets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from analyze_pillar_matrix_signal import N_FOLDS, RANDOM_SEED, build_baseline_design
from analyze_source_a_representation import (
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from analyze_source_a_section_scope import (
    NARRATIVE_TITLE_PATTERN,
    select_sections,
)
from analyze_source_a_tiers import assign_tiers
from extract_source_a_features import VARIANT_COLUMNS
from extract_source_a_section_features import (
    ECONOMY_TITLE_PATTERN,
    SECTIONS_PARQUET_PATH,
    section_feature_columns,
)
from ingest_source_a import normalize_article_text, strip_self_reference
from pillar_matrix import build_matrix

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"

EMBEDDINGS_PARQUET_PATH: Path = DATA_DIR / "source_a_embeddings.parquet"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_tiered_embedding.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_tiered_embedding_stats.json"

# 384 dimensions against bge-m3's 1024, and 90MB against 2.2GB. Chosen as a
# natively small model rather than a reduction of a large one.
ENCODER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Characters per chunk. The encoder's window is 256 word-pieces; ~900 characters
# lands under it for ordinary English prose with room to spare, so chunks are
# not silently truncated inside the encoder.
CHUNK_CHARS: int = 900

# Chunks kept per county. Ten chunks is ~9,000 characters, which covers the
# median county's entire non-narrative body. The cap exists so one 40,000-word
# article cannot dominate runtime, and what it drops is reported.
MAX_CHUNKS_PER_COUNTY: int = 10

# Reduced width scored alongside the native one, since "smaller" was part of the
# question and 64 is where a projection starts being meaningfully cheaper to
# serve than the typed block it competes with.
PCA_COMPONENTS: int = 64

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextVariant:
    """One rule for assembling a county's encoder input.

    Attributes:
        key: Short identifier used in result columns and stats.
        label: Human-readable description for reports.
        tier_scope: Tier to section-selection regex, or None for lead only.
    """

    key: str
    label: str
    tier_scope: dict[str, str | None]


ALL_TITLES: str = r".*"

TEXT_VARIANTS: tuple[TextVariant, ...] = (
    TextVariant(
        "lead_only",
        "cleaned lead section only",
        {tier: None for tier in ("stub", "thin", "mid", "rich")},
    ),
    TextVariant(
        "uniform",
        "lead + every non-narrative section, same rule for all",
        {tier: ALL_TITLES for tier in ("stub", "thin", "mid", "rich")},
    ),
    TextVariant(
        "tier_conditional",
        "depth by tier: thin reads the article, rich reads its lead",
        {
            "stub": ALL_TITLES,
            "thin": ALL_TITLES,
            "mid": ECONOMY_TITLE_PATTERN,
            "rich": None,
        },
    ),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def clean_section_text(text: str, county_name: str) -> str:
    """Normalize a section's text the way the lead's text was normalized.

    The retired embedding encoded `embedding_text`, which has the county and
    state name stripped so the model cannot key on them. Section text gets the
    same treatment, so that the only difference between variants is which part
    of the article is read.

    Args:
        text: Raw section text.
        county_name: County name to strip.

    Returns:
        Normalized, self-reference-stripped text.
    """
    return strip_self_reference(normalize_article_text(text), county_name)


def build_variant_texts(
    frame: pd.DataFrame, sections: pd.DataFrame, variant: TextVariant
) -> pd.Series:
    """Assemble every county's encoder input under one variant.

    Args:
        frame: Rows carrying `fips_code`, `county_name`, `embedding_text`, `tier`.
        sections: Long-format section frame.
        variant: The rule to apply.

    Returns:
        Series of input text, aligned to `frame`'s index.
    """
    texts = frame["embedding_text"].fillna("")
    for tier, pattern in variant.tier_scope.items():
        if pattern is None:
            continue
        selected = select_sections(sections, pattern, NARRATIVE_TITLE_PATTERN)
        joined = selected.groupby("fips_code")["section_text"].agg(" ".join)
        tier_rows = frame["tier"].to_numpy() == tier
        extra = frame.loc[tier_rows, "fips_code"].map(joined).fillna("")
        cleaned = [
            clean_section_text(text, name)
            for text, name in zip(extra, frame.loc[tier_rows, "county_name"])
        ]
        texts.loc[tier_rows] = texts.loc[tier_rows] + " " + pd.Series(
            cleaned, index=extra.index
        )
    return texts.str.strip()


def chunk_text(text: str) -> list[str]:
    """Split text into encoder-sized chunks, applying the per-county cap.

    Args:
        text: Assembled input text.

    Returns:
        Up to `MAX_CHUNKS_PER_COUNTY` chunks; empty list for empty text.
    """
    if not text:
        return []
    chunks = [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    return chunks[:MAX_CHUNKS_PER_COUNTY]


def encode_variant(model, texts: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
    """Encode every county's chunks and mean-pool them into one vector.

    Args:
        model: Loaded SentenceTransformer.
        texts: Assembled input text per county.

    Returns:
        Tuple of (vectors aligned to `texts`, diagnostics about chunking).
    """
    chunked = [chunk_text(text) for text in texts]
    flat = [chunk for chunks in chunked for chunk in chunks]
    logger.info("encoding %d chunks over %d counties", len(flat), len(chunked))
    encoded = model.encode(flat, batch_size=256, show_progress_bar=False,
                           normalize_embeddings=True)

    vectors = np.zeros((len(chunked), encoded.shape[1]), dtype="float64")
    cursor = 0
    for row, chunks in enumerate(chunked):
        if not chunks:
            cursor += 0
            continue
        vectors[row] = encoded[cursor : cursor + len(chunks)].mean(axis=0)
        cursor += len(chunks)

    capped = sum(1 for text, chunks in zip(texts, chunked)
                 if len(text) > CHUNK_CHARS * MAX_CHUNKS_PER_COUNTY)
    dropped = sum(max(0, len(text) - CHUNK_CHARS * MAX_CHUNKS_PER_COUNTY) for text in texts)
    diagnostics = {
        "n_chunks": float(len(flat)),
        "mean_chars": float(texts.str.len().mean()),
        "counties_hitting_cap": float(capped),
        "chars_dropped_by_cap": float(dropped),
    }
    return vectors, diagnostics


def score_blocks(
    matrix: pd.DataFrame, blocks: dict[str, tuple[np.ndarray, int | None]], targets: list
) -> pd.DataFrame:
    """Score every representation against every target on a shared baseline.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Representation key to (array, PCA components or None).
        targets: Targets to predict.

    Returns:
        One row per (target, representation).
    """
    baseline = build_baseline_design(matrix)
    records: list[dict[str, float | str | int]] = []

    for target in targets:
        rows = matrix[target.column].notna().to_numpy()
        y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
        base_design = baseline.loc[rows].to_numpy(dtype="float64")
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        baseline_predictions = _baseline_oof_predictions(base_design, y, folds)
        r2_baseline = float(r2_score(y, baseline_predictions))

        for key, (block, n_components) in blocks.items():
            predictions = _residual_oof_predictions(
                base_design, block[rows], y, folds, n_components
            )
            records.append(
                {
                    "pillar": target.pillar,
                    "column": target.column,
                    "label": target.label,
                    "representation": key,
                    "n": int(rows.sum()),
                    "r2_baseline": r2_baseline,
                    "lift": float(r2_score(y, predictions)) - r2_baseline,
                }
            )
        logger.info("scored %s", target.column)

    return pd.DataFrame.from_records(records)


def main() -> None:
    """Encode every text variant, score it, and write results."""
    configure_logging()
    from sentence_transformers import SentenceTransformer

    from analyze_source_b_industry_mix import NAICS2_LABELS

    matrix, pillar_blocks = build_matrix()
    matrix["tier"] = assign_tiers(matrix["content_length"])
    embeddings = pd.read_parquet(EMBEDDINGS_PARQUET_PATH)[["fips_code", "embedding_text"]]
    matrix = matrix.merge(embeddings, on="fips_code", how="left")
    sections = pd.read_parquet(SECTIONS_PARQUET_PATH)
    targets = build_non_a_targets(pillar_blocks, NAICS2_LABELS)

    logger.info("loading %s", ENCODER_NAME)
    model = SentenceTransformer(ENCODER_NAME)

    blocks: dict[str, tuple[np.ndarray, int | None]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for variant in TEXT_VARIANTS:
        texts = build_variant_texts(matrix, sections, variant)
        vectors, stats = encode_variant(model, texts)
        blocks[variant.key] = (vectors, None)
        blocks[f"{variant.key}_pca{PCA_COMPONENTS}"] = (vectors, PCA_COMPONENTS)
        diagnostics[variant.key] = stats
        logger.info("%s: %s", variant.key, stats)

    typed = matrix[list(VARIANT_COLUMNS["extracted_full"]) + section_feature_columns()]
    blocks["typed_sections"] = (typed.to_numpy(dtype="float64"), None)

    scores = score_blocks(matrix, blocks, targets)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_CSV_PATH, index=False)

    by_rep = scores.groupby("representation")["lift"]
    reference = scores[scores.representation.eq("typed_sections")].set_index("column")["lift"]
    payload = {
        "encoder": ENCODER_NAME,
        "n_dimensions": int(blocks["lead_only"][0].shape[1]),
        "pca_components": PCA_COMPONENTS,
        "chunk_chars": CHUNK_CHARS,
        "max_chunks_per_county": MAX_CHUNKS_PER_COUNTY,
        "n_targets": int(scores["column"].nunique()),
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
        "text_diagnostics": diagnostics,
        "representations": {
            key: {
                "mean_lift": float(by_rep.mean()[key]),
                "median_lift": float(by_rep.median()[key]),
                "n_targets_beating_typed": int(
                    (
                        scores[scores.representation.eq(key)].set_index("column")["lift"]
                        > reference
                    ).sum()
                ),
            }
            for key in blocks
        },
    }
    OUTPUT_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    for key, entry in payload["representations"].items():
        logger.info("%-26s mean lift %+.5f", key, entry["mean_lift"])


if __name__ == "__main__":
    main()
