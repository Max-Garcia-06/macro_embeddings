"""Offline re-embedding of Source A from stored intro texts.

Reads raw_intro_text from source_a_embeddings.parquet (no Wikimedia API
access), applies a cleaning variant, re-embeds with BAAI/bge-m3, and writes
source_a_embeddings_{variant}.parquet including the embedding_text column so
the exact embedded text is auditable.

Variants:
  raw: no cleaning -- embeds raw_intro_text unchanged. Used to reconstruct
      the true pre-cleaning baseline for gate comparisons.
  v2: strip_self_reference + strip_boilerplate_phrasing (incl. the new
      eponym / metro-area / formation patterns), with empty-text fallback.
  v3: v2, then drop sentences whose masked template appears in >=5% of
      counties (boilerplate_frequency).
  v4: v2, then drop sentences whose masked template appears in >=5% of
      counties, UNLESS doing so would leave a county's text below
      MIN_CONTENT_LENGTH -- in that case the v2 text is kept unfiltered for
      that county (see analysis-output/source-a-findings.md section 12's
      rural-county over-stripping finding).
  v5: v2, then frequency-filter drops are reviewed by a local Gemma model
      (llm_boilerplate_review) and restored where Gemma judges they carry
      county-specific content the kept text lacks (see
      analysis-output/source-a-findings.md section 12's rural-county
      over-stripping finding, and docs/rounds/2026-07-08-llm-boilerplate-
      review/DESIGN.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from analyze_source_a_similarity import MIN_CONTENT_LENGTH
from boilerplate_frequency import (
    DEFAULT_MIN_COUNTY_FRACTION,
    drop_common_sentences,
    find_common_templates,
    find_dropped_sentences,
    mask_sentence_template,
    split_sentences,
)
from ingest_source_a import BgeM3EmbeddingGenerator, configure_logging
from llm_boilerplate_review import DEFAULT_CACHE_PATH, GemmaClient, review_dropped_sentences
from text_cleaning import clean_for_embedding

INPUT_PARQUET_PATH: Path = Path(__file__).resolve().parent / "source_a_embeddings.parquet"
OUTPUT_TEMPLATE: str = "source_a_embeddings_{variant}.parquet"
LOG_EVERY: int = 250

logger = logging.getLogger(__name__)


def _restore_ordered(text: str, common_templates: set[str], restored: set[str]) -> str:
    """Rebuild text keeping non-common sentences plus any Gemma-restored ones.

    Preserves the original sentence order (restorations are not appended at
    the end). Falls back to the unfiltered text if nothing would remain,
    matching drop_common_sentences's own fallback.

    Args:
        text: Original (v2) county text, before frequency filtering.
        common_templates: Output of find_common_templates.
        restored: Sentences Gemma judged should be restored.

    Returns:
        Reconstructed text.
    """
    sentences = split_sentences(text)
    kept = [
        s for s in sentences if mask_sentence_template(s) not in common_templates or s in restored
    ]
    if not kept:
        return text
    return " ".join(kept)


def build_embedding_texts(
    df: pd.DataFrame,
    variant: str,
    gemma_client: GemmaClient | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> list[str]:
    """Produce the per-county text to embed for a cleaning variant.

    Args:
        df: DataFrame with county_name and raw_intro_text columns.
        variant: "raw" (no cleaning), "v2" (regex cleaning), "v3" (v2 +
            frequency filter), "v4" (v2 + outcome-gated frequency filter),
            or "v5" (v2 + frequency filter, with a local Gemma model
            reviewing and potentially restoring each dropped sentence).
        gemma_client: Client used for "v5" reviews; a default GemmaClient()
            is constructed if not given. Ignored for other variants.
        cache_path: Verdict cache path used for "v5". Ignored for other
            variants.

    Returns:
        One non-empty text per row, in row order.

    Raises:
        ValueError: If variant is not one of the supported names.
    """
    if variant not in ("raw", "v2", "v3", "v4", "v5"):
        raise ValueError(f"Unknown variant: {variant!r}")
    if variant == "raw":
        return df["raw_intro_text"].tolist()
    texts = [
        clean_for_embedding(raw, name)
        for name, raw in zip(df["county_name"], df["raw_intro_text"])
    ]
    if variant in ("v3", "v4", "v5"):
        templates = find_common_templates(texts, DEFAULT_MIN_COUNTY_FRACTION)
        logger.info("Frequency filter: %d common templates found", len(templates))
        filtered = [drop_common_sentences(t, templates) for t in texts]
        if variant == "v3":
            texts = filtered
        elif variant == "v4":
            texts = [
                candidate if len(candidate) >= MIN_CONTENT_LENGTH else original
                for original, candidate in zip(texts, filtered)
            ]
        else:  # v5
            client = gemma_client if gemma_client is not None else GemmaClient()
            texts = [
                _restore_ordered(
                    original,
                    templates,
                    set(
                        review_dropped_sentences(
                            candidate,
                            find_dropped_sentences(original, templates),
                            client,
                            cache_path,
                        )
                    ),
                )
                for original, candidate in zip(texts, filtered)
            ]
    return texts


def main() -> None:
    """Re-embed the full corpus for one cleaning variant."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["raw", "v2", "v3", "v4", "v5"], required=True)
    args = parser.parse_args()

    df = pd.read_parquet(INPUT_PARQUET_PATH)
    logger.info("Loaded %d counties from %s", len(df), INPUT_PARQUET_PATH)
    texts = build_embedding_texts(df, args.variant)

    embedder = BgeM3EmbeddingGenerator(device="cpu")
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        if i % LOG_EVERY == 0:
            logger.info("Embedding %d/%d", i, len(texts))
        vector = embedder.l2_normalize(embedder.encode(text))
        embeddings.append(vector.tolist())

    out = df[["county_name", "fips_code", "raw_intro_text"]].copy()
    out["embedding_text"] = texts
    out["embedding"] = embeddings
    output_path = Path(__file__).resolve().parent / OUTPUT_TEMPLATE.format(variant=args.variant)
    out.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(out), output_path)


if __name__ == "__main__":
    main()
