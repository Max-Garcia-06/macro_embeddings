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

import argparse
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
    MIN_TIER_OBSERVATIONS,
    _baseline_oof_predictions,
    _residual_oof_predictions,
    build_non_a_targets,
)
from analyze_source_a_section_scope import (
    NARRATIVE_TITLE_PATTERN,
    select_sections,
)
from analyze_source_a_tiers import TIER_LABELS, assign_tiers
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
OUTPUT_TIER_CSV_PATH: Path = OUTPUTS_DIR / "source_a_tiered_embedding_by_tier.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_tiered_embedding_stats.json"

# 384 dimensions against bge-m3's 1024, and 90MB against 2.2GB. Chosen as a
# natively small model rather than a reduction of a large one.
ENCODER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Characters per chunk. The encoder's window is 256 word-pieces; ~900 characters
# lands under it for ordinary English prose with room to spare, so chunks are
# not silently truncated inside the encoder.
CHUNK_CHARS: int = 900

# Chunks kept per county. The cap exists so one 40,000-word article cannot
# dominate runtime, and what it drops is reported.
#
# **Raised from 10 to 64 on 2026-08-17.** Ten chunks is ~9,000 characters, which
# covers the median county's non-narrative body but *not* the mean one: at that
# setting the `uniform` arm hit the cap for 1,871 of 3,144 counties (59.5%) and
# discarded 17.1M characters, roughly 43% of its own input. That made the
# headline result -- "the small encoder still loses to the typed block" -- a
# claim about an arm that had read at most 9,000 characters per county, which is
# a weaker claim than it appeared. 64 chunks is ~57,600 characters; the
# diagnostic below reports what still binds.
MAX_CHUNKS_PER_COUNTY: int = 64

# Reduced width scored alongside the native one, since "smaller" was part of the
# question and 64 is where a projection starts being meaningfully cheaper to
# serve than the typed block it competes with.
PCA_COMPONENTS: int = 64

# Batched encoding is not composition-invariant: `sentence-transformers` buckets by
# length and pads per batch, so a differently sized chunk list perturbs the numerics.
# Measured at ~1.1e-7 on both `mps` and `cpu` -- it is padding, not the GPU, and
# forcing CPU does not remove it. Identical composition reproduces bitwise, which is
# why the pipeline's own reruns stay exact. Bitwise equality across *arms* was never
# achievable and is not the claim.
VECTOR_TOLERANCE: float = 1e-5

# Every lift in the notebook is printed at 1e-5. Requiring agreement an order below
# that means the splice cannot change a digit anyone reads.
LIFT_TOLERANCE: float = 1e-6

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
    # The mirror image, and the sharper test of why `tier_conditional` loses.
    # If the explanation is "the rich tier is where economic content lives and
    # that rule reads least from it", then inverting the rule should recover most
    # of `uniform`'s gain. If it recovers all of it, the thin tier's sections are
    # contributing nothing and could be skipped for free; if it recovers only
    # part, they carry signal the lead does not.
    TextVariant(
        "tier_conditional_inverse",
        "depth by tier, inverted: rich reads the article, thin reads its lead",
        {
            "stub": None,
            "thin": None,
            "mid": ECONOMY_TITLE_PATTERN,
            "rich": ALL_TITLES,
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


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving direction untouched.

    Args:
        vectors: Pooled county vectors.

    Returns:
        Row-normalized copy.
    """
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def vector_tier_diagnostics(
    vectors: np.ndarray, tiers: pd.Series
) -> tuple[float, dict[str, float]]:
    """Measure how much of a vector set's variance is explained by tier alone.

    Args:
        vectors: Pooled county vectors.
        tiers: Content tier per county, aligned to `vectors`.

    Returns:
        Tuple of (between-tier variance share, mean vector norm per tier).
    """
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    total_ss = float((centred**2).sum())
    between_ss = 0.0
    tier_array = tiers.to_numpy()
    for tier in TIER_LABELS:
        mask = tier_array == tier
        between_ss += float(mask.sum()) * float((centred[mask].mean(axis=0) ** 2).sum())
    norms = np.linalg.norm(vectors, axis=1)
    return between_ss / total_ss, {
        tier: float(norms[tier_array == tier].mean()) for tier in TIER_LABELS
    }


def encode_variant(
    model, texts: pd.Series, tiers: pd.Series
) -> tuple[np.ndarray, dict[str, float]]:
    """Encode every county's chunks and mean-pool them into one vector.

    Chunk vectors are unit-norm, so mean-pooling k of them yields a vector whose
    length falls roughly as 1/sqrt(k) once the chunks stop pointing the same way.
    That makes the pooled norm a direct readout of how much text a county
    contributed -- and under a tier-conditional rule, a readout of its tier. The
    per-tier norms are reported for exactly that reason.

    Args:
        model: Loaded SentenceTransformer.
        texts: Assembled input text per county.
        tiers: Content tier per county, aligned to `texts`.

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
    tier_variance_share, mean_norm_by_tier = vector_tier_diagnostics(vectors, tiers)
    chunk_counts = np.array([len(chunks) for chunks in chunked], dtype="float64")
    diagnostics = {
        "n_chunks": float(len(flat)),
        # Share of the vector set's total variance that is explained purely by
        # which tier a county is in. Under a uniform rule this is whatever the
        # corpus genuinely has; under a tier-conditional rule it also absorbs the
        # construction rule itself, and tier is a size proxy the baseline already
        # controls for -- so any variance spent on it is variance not spent on
        # anything the model can use.
        "tier_variance_share": tier_variance_share,
        "mean_norm_by_tier": mean_norm_by_tier,
        "mean_chunks_by_tier": {
            tier: float(chunk_counts[tiers.to_numpy() == tier].mean())
            for tier in TIER_LABELS
        },
        "mean_chars": float(texts.str.len().mean()),
        "counties_hitting_cap": float(capped),
        "chars_dropped_by_cap": float(dropped),
    }
    return vectors, diagnostics


def score_blocks(
    matrix: pd.DataFrame, blocks: dict[str, tuple[np.ndarray, int | None]], targets: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every representation against every target on a shared baseline.

    The fit stays pooled and only the out-of-fold predictions are sliced by tier.
    Fitting per tier as well would change two things at once and make the lift
    incomparable to the pooled arms.

    Each per-tier row carries its own `r2_baseline` because R2 does not decompose
    across subsets: every slice computes its total sum of squares against its own
    mean, so a tier's raw R2 is not comparable to another tier's. The *lift* is,
    since it differences two R2 over identical rows and the denominator cancels.

    Args:
        matrix: Feature matrix from `build_matrix`.
        blocks: Representation key to (array, PCA components or None).
        targets: Targets to predict.

    Returns:
        Tuple of (one row per (target, representation), one row per
        (target, representation, tier) clearing `MIN_TIER_OBSERVATIONS`).
    """
    baseline = build_baseline_design(matrix)
    records: list[dict[str, float | str | int]] = []
    tier_records: list[dict[str, float | str | int]] = []

    for target in targets:
        rows = matrix[target.column].notna().to_numpy()
        y = matrix.loc[rows, target.column].to_numpy(dtype="float64")
        base_design = baseline.loc[rows].to_numpy(dtype="float64")
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        baseline_predictions = _baseline_oof_predictions(base_design, y, folds)
        r2_baseline = float(r2_score(y, baseline_predictions))
        row_tiers = matrix.loc[rows, "tier"].to_numpy()

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
            for tier in TIER_LABELS:
                mask = row_tiers == tier
                if mask.sum() < MIN_TIER_OBSERVATIONS:
                    continue
                tier_baseline = float(r2_score(y[mask], baseline_predictions[mask]))
                tier_records.append(
                    {
                        "pillar": target.pillar,
                        "column": target.column,
                        "label": target.label,
                        "representation": key,
                        "tier": tier,
                        "n": int(mask.sum()),
                        "r2_baseline": tier_baseline,
                        "lift": float(r2_score(y[mask], predictions[mask])) - tier_baseline,
                    }
                )
        logger.info("scored %s", target.column)

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(tier_records)


def verify_splice(model, matrix: pd.DataFrame, sections: pd.DataFrame, targets) -> None:
    """Assert a spliced arm equals a genuinely encoded one.

    The sweep arms are built by copying rows between two existing vector sets
    rather than by encoding new text. That is only valid because a county's
    vector depends on no other county. This encodes `drop_rich` for real and
    compares, so the claim is tested rather than asserted.

    Args:
        model: Loaded SentenceTransformer.
        matrix: Feature matrix carrying `tier`, `embedding_text`, `county_name`.
        sections: Long-format section frame.
        targets: Targets to score, for the published-lift comparison.

    Raises:
        AssertionError: If the spliced and encoded arms differ.
    """
    uniform = TextVariant(
        "uniform", "", {tier: ALL_TITLES for tier in TIER_LABELS}
    )
    lead_only = TextVariant("lead_only", "", {tier: None for tier in TIER_LABELS})
    drop_rich = TextVariant(
        "drop_rich",
        "",
        {"stub": ALL_TITLES, "thin": ALL_TITLES, "mid": ALL_TITLES, "rich": None},
    )
    uniform_vectors, _ = encode_variant(
        model, build_variant_texts(matrix, sections, uniform), matrix["tier"]
    )
    lead_vectors, _ = encode_variant(
        model, build_variant_texts(matrix, sections, lead_only), matrix["tier"]
    )
    encoded, _ = encode_variant(
        model, build_variant_texts(matrix, sections, drop_rich), matrix["tier"]
    )

    tier_array = matrix["tier"].to_numpy()
    spliced = uniform_vectors.copy()
    spliced[tier_array == "rich"] = lead_vectors[tier_array == "rich"]

    largest = float(np.abs(spliced - encoded).max())
    logger.info("splice vs encode, largest absolute difference: %.3e", largest)
    assert largest < VECTOR_TOLERANCE, (
        f"splice and encode disagree by {largest:.3e}, above {VECTOR_TOLERANCE:.0e}"
    )

    # Vector agreement is necessary but not sufficient. What licenses the design
    # is that no number this notebook prints can move: every lift is reported at
    # 1e-5, so score both arms and require agreement an order below that.
    scores, _ = score_blocks(
        matrix, {"spliced": (spliced, None), "encoded": (encoded, None)}, targets
    )
    wide = scores.pivot(index="column", columns="representation", values="lift")
    lift_gap = float((wide["spliced"] - wide["encoded"]).abs().max())
    logger.info(
        "largest lift difference across %d targets: %.3e", len(wide), lift_gap
    )
    assert lift_gap < LIFT_TOLERANCE, (
        f"splice changes a published lift by {lift_gap:.3e}, "
        f"above {LIFT_TOLERANCE:.0e}"
    )
    logger.info("splice verified: vectors within tolerance, published lifts unmoved")


def main(verify_only: bool = False) -> None:
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

    if verify_only:
        verify_splice(model, matrix, sections, targets)
        return

    blocks: dict[str, tuple[np.ndarray, int | None]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for variant in TEXT_VARIANTS:
        texts = build_variant_texts(matrix, sections, variant)
        vectors, stats = encode_variant(model, texts, matrix["tier"])
        blocks[variant.key] = (vectors, None)
        blocks[f"{variant.key}_pca{PCA_COMPONENTS}"] = (vectors, PCA_COMPONENTS)
        # Row-normalized twin. `StandardScaler` centres and scales each dimension
        # across counties, which does nothing to a magnitude gradient that runs
        # across *rows* -- so a pooled norm that varies with tier survives into
        # every column as a tier-correlated component. Scoring the direction
        # alone isolates that channel: if a variant recovers here, its loss was
        # the magnitude gradient rather than the text it read.
        blocks[f"{variant.key}_l2"] = (l2_normalize(vectors), None)
        diagnostics[variant.key] = stats
        logger.info("%s: %s", variant.key, stats)

    # Drop-one sweep. A county's vector depends only on its own tier's text rule
    # -- `build_variant_texts` selects per county and `encode_variant` mean-pools
    # each county independently -- so "uniform everywhere except tier T reads its
    # lead" is a row-wise splice of two arrays already in hand, and is identical
    # to what a fresh encode would produce. `--verify-splice` asserts that rather
    # than asking the reader to take it on faith.
    #
    # The splice is scored raw AND row-normalized because it manufactures a norm
    # discontinuity at the tier boundary: `lead_only` rows carry norm ~1.0 (one
    # chunk, nothing to cancel) against `uniform`'s 0.63-0.71. A model can read
    # the tier straight off magnitude without reading any text, so the raw arm is
    # confounded by construction and the `_l2` twin is the one to trust. The gap
    # between them measures the size of that artifact.
    uniform_vectors = blocks["uniform"][0]
    lead_vectors = blocks["lead_only"][0]
    tier_array = matrix["tier"].to_numpy()
    for tier in TIER_LABELS:
        spliced = uniform_vectors.copy()
        spliced[tier_array == tier] = lead_vectors[tier_array == tier]
        blocks[f"drop_{tier}"] = (spliced, None)
        blocks[f"drop_{tier}_l2"] = (l2_normalize(spliced), None)
        share, norms = vector_tier_diagnostics(spliced, matrix["tier"])
        diagnostics[f"drop_{tier}"] = {
            "tier_variance_share": share,
            "mean_norm_by_tier": norms,
        }
        logger.info("spliced drop_%s: tier_variance_share %.4f", tier, share)

    typed = matrix[list(VARIANT_COLUMNS["extracted_full"]) + section_feature_columns()]
    blocks["typed_sections"] = (typed.to_numpy(dtype="float64"), None)

    scores, tier_scores = score_blocks(matrix, blocks, targets)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_CSV_PATH, index=False)
    tier_scores.to_csv(OUTPUT_TIER_CSV_PATH, index=False)

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
        "mean_lift_by_tier": json.loads(
            tier_scores.groupby(["representation", "tier"])["lift"]
            .mean()
            .unstack()
            .to_json(orient="index")
        ),
    }
    OUTPUT_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    for key, entry in payload["representations"].items():
        logger.info("%-26s mean lift %+.5f", key, entry["mean_lift"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-splice",
        action="store_true",
        help="Encode one sweep arm for real and assert it equals the splice, then exit.",
    )
    main(verify_only=parser.parse_args().verify_splice)
