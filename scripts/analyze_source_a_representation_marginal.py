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
  identically for every county, mean-pooled. Scored here as the **unselected
  reference**: it is the width-matched twin of the leakage-carrying arm from
  the Part 3 sweep, kept in the comparison precisely because it lost scope
  selection -- it is informative as a contrast, not eligible to represent A.
- `minilm_uniform_l2` -- the same vectors, row-normalized.
- `minilm_uniform_pca29` / `minilm_uniform_pca64` -- the same vectors, reduced
  by PCA fitted inside each fold to 29 and 64 dimensions respectively, so the
  width-driven part of the embedding's measured penalty against the 29-column
  typed block is controlled for.
- `minilm_prose_plus_history_ccr_pca29` -- **the selected embedding arm**,
  fixed by the Task 9 scope selection (`docs/source_a_representation_decision.md`):
  `prose_plus_history` text (lead + substantive prose + history/notable
  people, no census tables/lists/highways), common-component-removed
  (`remove_common_component` -- corpus mean subtracted, then row-normalized),
  then PCA-reduced to 29 dimensions inside each fold to match the typed arm's
  width. Selection ran on the in-repo 28-target basket, disjoint from this
  script's external decision basket, so the arm was fixed before it was ever
  scored here.

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
    EXCLUDED_TARGETS as _EXCLUDED_TARGETS,
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
    l2_normalize,
    remove_common_component,
)
from analyze_source_a_tiers import assign_tiers
from pillar_matrix import SIZE_FEATURES
from source_a_typed_transform import transform_typed

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "source-a"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "source_a_representation_marginal.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "source_a_representation_marginal_stats.json"
CENTROIDS_PATH: Path = REPO_ROOT / "data" / "county_centroids.parquet"

# Targets scored and shipped in every output (CSV, stats JSON), but excluded
# from every headline mean this script reports, with the reason documented
# inline rather than silently dropped. A target only belongs here if BOTH its
# reduced and full R2 are negative -- i.e. every model scored against it is
# worse than predicting the mean, so its "contribution" is the gap between two
# useless models, not a gain. `test_marginal_arms.py::test_excluded_targets_are_documented`
# enforces that this dict stays non-empty and every reason is a real string.
#
# `no_fuel_used_share`: within the 3,144-county panel this target has mean
# 0.0067, sd 0.0232, max 0.644, skew 23.0, kurtosis 573 -- a near-degenerate,
# almost-always-zero rate. Its reduced R2 is -1.0031 and the selected arm's
# full R2 is -0.9720 (typed_transformed's full R2 is -1.0381): both worse than
# the mean. The county-level max (0.644) is nowhere near the ~0.9+ values that
# would suggest Puerto Rico sentinel contamination, and PR is not in this
# panel (0 of 3,144 rows carry state_fips 72) -- the earlier "PR + unmasked
# sentinels" explanation for this target's broken R2 does not hold up; see
# analysis-output/source-a/source-a-findings.md #22.
# Imported from `analyze_external_target` rather than defined here: the
# drop-one sweep draws the same basket and has to apply the same exclusion,
# and two copies of this list would drift.
EXCLUDED_TARGETS: dict[str, str] = dict(_EXCLUDED_TARGETS)

# The text variant underlying the SELECTED scope. `prose_plus_history` is the
# scope Task 9 selected (mean lift 0.004530 on the 28-target selection basket,
# vs typed_sections' 0.003072) -- see docs/source_a_representation_decision.md.
# The selected *representation* is this variant's vectors with the `_ccr`
# (common-component-removed) transform applied, which is what
# `build_source_a_embedding` scores under `minilm_{VARIANT_KEY}_ccr_pca29`.
# `uniform` is still encoded separately, unconditionally, as the unselected
# reference arm (see EMBEDDING_ARMS below).
VARIANT_KEY: str = "prose_plus_history"

logger = logging.getLogger(__name__)

# Embedding arms entering the marginal comparison, and the width each is reduced
# to. `None` means the native 384 dimensions. The 29-dimension arms exist so the
# comparison against the typed block is width-matched: findings §21.2 states that
# an unknown share of the embedding's penalty is width rather than content, and
# names this arm as the missing control.
#
# `minilm_uniform*` are the UNSELECTED reference -- the width-matched twin of
# the leakage-carrying arm from the Part 3 sweep. They stay in the comparison
# as a contrast and are not eligible to be the embedding representative.
#
# `minilm_prose_plus_history_ccr_pca29` is the SELECTED embedding arm fixed by
# Task 9's pre-registered scope selection.
#
# `minilm_uniform_ccr_pca29` separates the two things `minilm_prose_plus_history_ccr_pca29`
# changes at once relative to `minilm_uniform_pca29`: text SCOPE
# (`prose_plus_history` vs `uniform`) and the CCR transform. This arm holds
# scope at `uniform` and adds only CCR, so
# `minilm_uniform_ccr_pca29 - minilm_uniform_pca29` isolates CCR's effect and
# `minilm_prose_plus_history_ccr_pca29 - minilm_uniform_ccr_pca29` isolates
# scope's effect, holding CCR fixed. See findings #22.3.
EMBEDDING_ARMS: dict[str, int | None] = {
    "minilm_uniform": None,
    "minilm_uniform_l2": None,
    "minilm_uniform_pca29": 29,
    "minilm_uniform_pca64": 64,
    "minilm_uniform_ccr_pca29": 29,
    "minilm_prose_plus_history_ccr_pca29": 29,
}


# --- Intervals on the headline means -----------------------------------------
#
# Every figure this script headlines is a mean over the decision basket, printed
# to four decimal places with nothing attached saying how far it could move. The
# basket is small (41 targets) and clustered -- five of its targets are
# heating-fuel shares off a single ACS table -- so the gap between two arms'
# means is not readable without an interval.
#
# The resampling unit is the TARGET, because the target is the unit the headline
# mean is taken over. It is not the county: counties sit inside the folds and
# the cross-validation already accounts for them.
#
# Two resamples are reported side by side, and the gap between them is the
# concrete size of the clustering caveat this script's prose otherwise states in
# words:
#
# - `naive` resamples individual targets, which assumes targets are independent.
#   They are not, so this interval is the optimistic one.
# - `table_clustered` resamples whole ACS tables, keeping every target drawn from
#   a table together, which respects the strongest dependence we can name.
#
# Within a replicate every arm is scored on the SAME draw -- a paired bootstrap
# -- so an interval on a *difference* between two arms is not inflated by
# target-level variance both arms share.
#
# Nothing here re-fits a model. It resamples the per-target contributions
# `score_representation` already computed.
N_BOOTSTRAP: int = 10_000

# Percentile bounds of the reported interval: a 95% interval.
BOOTSTRAP_PERCENTILES: tuple[float, float] = (2.5, 97.5)

# The arm every other arm's difference is reported against. Two coordinate
# columns are the cheap competitor the geography control raised, so
# `<arm> - latlong_only` is the difference the shipping decision turns on.
BOOTSTRAP_REFERENCE_ARM: str = "latlong_only"

# The arm Task 9's pre-registered scope selection fixed, named once so the
# summary log and the notebook quote the same key.
SELECTED_ARM: str = f"minilm_{VARIANT_KEY}_ccr_pca29"

# Target -> ACS table, for the clustered resample. Read off the target
# definitions rather than re-typed here, so a target added upstream lands in its
# cluster without an edit in this file.
TARGET_TABLES: dict[str, str] = {
    target.column: target.table for target in EXTERNAL_TARGETS
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

    Encodes two text variants: `uniform`, unconditionally, for the unselected
    reference arms; and `VARIANT_KEY` (the Task 9 selected scope's underlying
    text), for the selected arm. The selected arm is common-component-removed
    (`remove_common_component`) before being handed back, because the
    selection in `docs/source_a_representation_decision.md` picked the `_ccr`
    representation, not the raw pooled vectors -- scoring the raw vectors
    under the selected arm's name would silently substitute a different,
    unselected representation.

    Args:
        fips_order: The `fips_code` sequence the panel is in.

    Returns:
        Mapping of representation key to a vector array whose rows align to
        `fips_order`. Counties with no text get a zero vector, which is what the
        encoder harness already does for them. Keys with a PCA width in
        `EMBEDDING_ARMS` carry the unreduced vectors here -- the reduction
        itself happens per fold in `out_of_fold_predictions_with_reduction`,
        never here, so it never sees held-out rows.
    """
    from sentence_transformers import SentenceTransformer

    from pillar_matrix import build_matrix

    matrix, _ = build_matrix()
    matrix["tier"] = assign_tiers(matrix["content_length"])
    text = pd.read_parquet(EMBEDDINGS_PARQUET_PATH)[["fips_code", "embedding_text"]]
    matrix = matrix.merge(text, on="fips_code", how="left")
    sections = pd.read_parquet(SECTIONS_PARQUET_PATH)

    logger.info("loading %s", ENCODER_NAME)
    model = SentenceTransformer(ENCODER_NAME)

    # Reindex helper onto the panel's row order. `build_matrix` and the panel
    # are both keyed on fips_code but the panel is an inner join against the
    # targets, so it is a subset in its own order rather than the matrix's.
    by_fips = pd.DataFrame({"fips_code": matrix["fips_code"]})
    position = by_fips.reset_index().set_index("fips_code")["index"]
    rows = fips_order.map(position).to_numpy()
    missing = int(pd.isna(rows).sum())
    if missing:
        raise ValueError(f"{missing} panel counties absent from the encoded matrix")
    rows = rows.astype(int)

    # Unselected reference: `uniform` text, unchanged from the Part 3 sweep.
    uniform_variant = next(v for v in TEXT_VARIANTS if v.key == "uniform")
    uniform_texts = build_variant_texts(matrix, sections, uniform_variant)
    uniform_vectors, uniform_diagnostics = encode_variant(
        model, uniform_texts, matrix["tier"]
    )
    logger.info("uniform: %s", uniform_diagnostics)
    uniform_normed = l2_normalize(uniform_vectors)

    # `uniform`, common-component-removed. Same transform as the selected arm
    # below, applied to `uniform`'s scope instead of `VARIANT_KEY`'s -- this is
    # what makes the scope-vs-CCR decomposition possible (see EMBEDDING_ARMS).
    # Centring is computed over the full corpus (`uniform_vectors`, all 3,144
    # counties), the same transductive convention the selected arm uses below:
    # the corpus mean sees every county's vector, including states that are
    # held out for any given fold, because the mean is not fitted with target
    # information and is fixed before the panel subset or the folds exist.
    uniform_ccr = remove_common_component(uniform_vectors)

    # Selected scope: `VARIANT_KEY` text, common-component-removed. This must
    # match what Task 9 scored as `prose_plus_history_ccr` -- centring is
    # computed over the same full corpus (`matrix`, not the panel subset)
    # before the panel's rows are pulled out, exactly as the selection sweep
    # did in `analyze_source_a_tiered_embedding.main`. As with `uniform_ccr`
    # above, this mean is transductive (sees held-out states) unlike the PCA
    # reduction applied afterward, which is refit inside each fold on training
    # rows only -- see `fit_reduction`.
    selected_variant = next(v for v in TEXT_VARIANTS if v.key == VARIANT_KEY)
    selected_texts = build_variant_texts(matrix, sections, selected_variant)
    selected_vectors, selected_diagnostics = encode_variant(
        model, selected_texts, matrix["tier"]
    )
    logger.info("%s: %s", VARIANT_KEY, selected_diagnostics)
    selected_ccr = remove_common_component(selected_vectors)

    return {
        "minilm_uniform": uniform_vectors[rows],
        "minilm_uniform_l2": uniform_normed[rows],
        "minilm_uniform_pca29": uniform_vectors[rows],
        "minilm_uniform_pca64": uniform_vectors[rows],
        "minilm_uniform_ccr_pca29": uniform_ccr[rows],
        f"minilm_{VARIANT_KEY}_ccr_pca29": selected_ccr[rows],
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


def _ensure_latlong_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach county centroid `lat`/`lon` to the panel if it lacks them.

    The selected arm's largest per-target gains cluster on climate/region
    variables (electric/gas/fuel-oil heating, foreign-born share, drove-alone
    share) -- exactly what two floating-point coordinates would also predict.
    `GroupKFold(state_fips)` does not control for this: a held-out New England
    county's article shares regional vocabulary with training New England
    counties, so the encoder can place it geographically without ever seeing
    its state. This control lets every arm's contribution be re-reported net
    of geography (`contribution_geo` in `score_representation`), rather than
    net of nothing more than county size and the other five pillars.

    Args:
        panel: Joined panel from `load_panel`.

    Returns:
        `panel`, with `lat`/`lon` columns added if missing.

    Raises:
        ValueError: If any panel county is absent from the centroid table.
    """
    if {"lat", "lon"}.issubset(panel.columns):
        return panel

    centroids = pd.read_parquet(CENTROIDS_PATH)[["fips_code", "lat", "lon"]]
    merged = panel.merge(centroids, on="fips_code", how="left")
    missing = int(merged["lat"].isna().sum())
    if missing:
        raise ValueError(f"{missing} panel counties absent from the centroid table")
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
        One row per (target, representation). Every row also carries
        `r2_reduced_geo` (the `size_and_others` model with `lat`/`lon` added)
        and `contribution_geo` (`r2_full - r2_reduced_geo`), so every arm's
        contribution can be read net of geography as well as net of nothing
        more than size and the other five pillars.
    """
    panel = _ensure_tier_column(panel)
    panel = _ensure_latlong_columns(panel)
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

        # `size + others + lat/lon`: the geography-augmented reduced baseline
        # (fix (a)), and its own contribution against `size_and_others` alone
        # doubles as the `latlong_only` arm (fix (b)) -- both are the same
        # design, scored once.
        latlong = usable[["lat", "lon"]].astype(float).to_numpy()
        size_and_others_geo = np.hstack([size_and_others, latlong])

        typed = [column for column in a_columns if column not in ablate]
        transformed, _ = transform_typed(usable, typed, usable["tier"])
        designs = {
            "typed": np.hstack([size_and_others, usable[typed].astype(float).to_numpy()]),
            "typed_transformed": np.hstack([size_and_others, transformed]),
            "latlong_only": size_and_others_geo,
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

        # `latlong_only`'s design IS `size_and_others_geo`, so its own full R2
        # is exactly the geography-augmented reduced baseline every other
        # arm's `contribution_geo` is measured against below.
        reduced_r2_geo = float(r2_score(y, representations["latlong_only"][0]))
        if reduced_r2 < 0:
            logger.warning(
                "%-24s reduced R2 %.4f is negative -- every contribution for "
                "this target is the gap between two worse-than-mean models, "
                "not a real gain. Flag it in the writeup.",
                target.column,
                reduced_r2,
            )

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
                    "r2_reduced_geo": reduced_r2_geo,
                    "r2_full": full_r2,
                    "contribution": full_r2 - reduced_r2,
                    "contribution_geo": full_r2 - reduced_r2_geo,
                    "reduced_r2_negative": bool(reduced_r2 < 0),
                    "excluded": target.column in EXCLUDED_TARGETS,
                }
            )
            logger.info(
                "%-24s %-18s contribution %+.5f (full %.4f, reduced %.4f, "
                "reduced+geo %.4f, contribution_geo %+.5f)",
                target.column,
                key,
                full_r2 - reduced_r2,
                full_r2,
                reduced_r2,
                reduced_r2_geo,
                full_r2 - reduced_r2_geo,
            )

    return pd.DataFrame(rows)


def _draw_target_positions(
    rng: np.random.Generator,
    targets: list[str],
    cluster_by_table: bool,
) -> np.ndarray:
    """Draw one bootstrap resample of target positions.

    Args:
        rng: Seeded generator. Called once per replicate.
        targets: Basket targets in the column order of the score matrices.
        cluster_by_table: Resample whole ACS tables when True, individual
            targets when False.

    Returns:
        Positions into `targets`, drawn with replacement. The naive draw is
        exactly `len(targets)` long; the clustered draw's length varies by
        replicate, because ACS tables hold different numbers of targets and a
        replicate that draws `b25040` twice carries ten heating-fuel rows.
    """
    if not cluster_by_table:
        return rng.integers(0, len(targets), size=len(targets))

    members: dict[str, list[int]] = {}
    for position, target in enumerate(targets):
        members.setdefault(TARGET_TABLES[target], []).append(position)
    tables = sorted(members)
    drawn = rng.integers(0, len(tables), size=len(tables))
    return np.concatenate([members[tables[index]] for index in drawn])


def _interval(draws: np.ndarray, point: float) -> dict[str, object]:
    """Wrap one bootstrap distribution as a reportable interval.

    Args:
        draws: One resampled statistic per replicate.
        point: The statistic on the observed basket, unresampled. Reported
            alongside rather than recomputed from the draws, because the
            percentile interval is not required to be centred on it.

    Returns:
        Point estimate, both bounds, and whether the interval covers zero --
        the last because "indistinguishable from zero" is a different claim
        from "small", and the prose quoting this needs to know which it has.
    """
    low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
    return {
        "point": float(point),
        "low": float(low),
        "high": float(high),
        "covers_zero": bool(low <= 0.0 <= high),
    }


def bootstrap_representations(scores: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Interval every arm's decision-basket means, paired across arms.

    Runs the resample described at `N_BOOTSTRAP` twice, naive and clustered by
    ACS table, and reports three statistics per arm per scheme: the arm's plain
    marginal contribution, its contribution net of latitude and longitude, and
    its difference from `BOOTSTRAP_REFERENCE_ARM`.

    The difference is reported rather than a ratio. A ratio of two means whose
    denominator is this small is unstable, and an interval on one is unreadable;
    the "96% of the encoder" figure stays a point estimate in prose.

    Two of the three statistics turn out to be the same one.
    `latlong_only`'s full model IS every other arm's geo-reduced baseline
    (`test_latlong_only_design_matches_the_geo_reduced_baseline`), so

        contribution_geo(arm) = contribution(arm) - contribution(latlong_only)

    holds identically, not approximately. Both are kept in the artifact: they
    cost nothing, and `test_geo_net_interval_is_the_latlong_difference` uses
    their agreement as a standing check that the identity has not been broken
    by a change to either design.

    Args:
        scores: Output of `score_representation`. Excluded targets are dropped
            here for the same reason `summarize` drops them from the headline
            means -- an interval on a basket that includes them would be
            interval-ing a number nothing quotes.

    Returns:
        Mapping of representation key to its `naive` and `table_clustered`
        blocks plus the resample's own parameters.

    Raises:
        ValueError: If some arm did not score some basket target, which would
            silently pair different arms against different baskets.
    """
    basket = scores[~scores["target"].isin(EXCLUDED_TARGETS)]
    targets = sorted(basket["target"].unique())
    arms = sorted(basket["representation"].unique())
    position = {target: index for index, target in enumerate(targets)}

    plain = np.full((len(arms), len(targets)), np.nan)
    geo = np.full((len(arms), len(targets)), np.nan)
    for arm_index, arm in enumerate(arms):
        for row in basket[basket["representation"] == arm].itertuples():
            plain[arm_index, position[row.target]] = row.contribution
            geo[arm_index, position[row.target]] = row.contribution_geo
    if np.isnan(plain).any() or np.isnan(geo).any():
        raise ValueError(
            "every representation must score every decision-basket target "
            "before the arms can be paired"
        )

    reference = arms.index(BOOTSTRAP_REFERENCE_ARM)
    observed_plain = plain.mean(axis=1)
    observed_geo = geo.mean(axis=1)

    out: dict[str, dict[str, object]] = {
        arm: {
            "n_replicates": N_BOOTSTRAP,
            "percentiles": list(BOOTSTRAP_PERCENTILES),
            "n_targets": len(targets),
            "n_tables": len({TARGET_TABLES[target] for target in targets}),
            "reference_arm": BOOTSTRAP_REFERENCE_ARM,
        }
        for arm in arms
    }

    for scheme, cluster_by_table in (("naive", False), ("table_clustered", True)):
        # Re-seeded per scheme so each is reproducible on its own, and so
        # adding a scheme never shifts an existing one's numbers.
        rng = np.random.default_rng(RANDOM_SEED)
        drawn_plain = np.empty((N_BOOTSTRAP, len(arms)))
        drawn_geo = np.empty((N_BOOTSTRAP, len(arms)))
        drawn_difference = np.empty((N_BOOTSTRAP, len(arms)))
        for replicate in range(N_BOOTSTRAP):
            # One draw, every arm -- this is what makes the comparison paired.
            selection = _draw_target_positions(rng, targets, cluster_by_table)
            replicate_plain = plain[:, selection].mean(axis=1)
            drawn_plain[replicate] = replicate_plain
            drawn_geo[replicate] = geo[:, selection].mean(axis=1)
            drawn_difference[replicate] = replicate_plain - replicate_plain[reference]

        for arm_index, arm in enumerate(arms):
            out[arm][scheme] = {
                "contribution": _interval(
                    drawn_plain[:, arm_index], observed_plain[arm_index]
                ),
                "contribution_geo": _interval(
                    drawn_geo[:, arm_index], observed_geo[arm_index]
                ),
                f"minus_{BOOTSTRAP_REFERENCE_ARM}": _interval(
                    drawn_difference[:, arm_index],
                    observed_plain[arm_index] - observed_plain[reference],
                ),
            }
    return out


def summarize(scores: pd.DataFrame) -> dict[str, object]:
    """Collapse per-target contributions to one figure per representation.

    Reports two baskets per representation: `mean_contribution` /
    `median_contribution` over every scored target (kept for audit and
    backward compatibility -- nothing here is silently dropped), and
    `decision_basket_mean_contribution` / `..._median_contribution` over the
    targets NOT in `EXCLUDED_TARGETS` -- the figures headline prose should
    quote. `negative_baseline_targets` names every target (regardless of
    representation) whose reduced-model R2 is below zero, per the general
    guard: such a target's "contribution" is the gap between two
    worse-than-mean models and must be flagged rather than averaged in
    silently.

    Each representation also carries a `bootstrap` block from
    `bootstrap_representations` -- naive and table-clustered intervals on the
    decision-basket means, so no figure this artifact publishes is a bare point
    estimate.

    Args:
        scores: Output of `score_representation`.

    Returns:
        JSON-serializable summary.
    """
    excluded_columns = set(EXCLUDED_TARGETS)
    negative_baseline = (
        scores[scores["reduced_r2_negative"]]
        .drop_duplicates(subset="target")[["target", "r2_reduced"]]
        .set_index("target")["r2_reduced"]
        .to_dict()
    )

    bootstrap = bootstrap_representations(scores)

    by_representation: dict[str, object] = {}
    for key, group in scores.groupby("representation"):
        basket = group[~group["target"].isin(excluded_columns)]
        by_representation[str(key)] = {
            "mean_contribution": float(group["contribution"].mean()),
            "median_contribution": float(group["contribution"].median()),
            "n_positive": int((group["contribution"] > 0).sum()),
            "n_targets": int(len(group)),
            "n_columns": int(group["n_columns"].iloc[0]),
            "decision_basket_mean_contribution": float(basket["contribution"].mean()),
            "decision_basket_median_contribution": float(basket["contribution"].median()),
            "decision_basket_n_positive": int((basket["contribution"] > 0).sum()),
            "decision_basket_n_targets": int(len(basket)),
            "mean_contribution_geo": float(group["contribution_geo"].mean()),
            "median_contribution_geo": float(group["contribution_geo"].median()),
            "decision_basket_mean_contribution_geo": float(basket["contribution_geo"].mean()),
            "decision_basket_median_contribution_geo": float(basket["contribution_geo"].median()),
            "by_target": {
                str(row.target): float(row.contribution) for row in group.itertuples()
            },
            "by_target_geo": {
                str(row.target): float(row.contribution_geo) for row in group.itertuples()
            },
            "bootstrap": bootstrap[str(key)],
        }
    return {
        "question": (
            "Is Source A's near-zero marginal contribution a fact about the "
            "pillar or about its representation?"
        ),
        "encoder": ENCODER_NAME,
        "text_variant": VARIANT_KEY,
        "n_targets": int(scores["target"].nunique()),
        "excluded_targets": dict(EXCLUDED_TARGETS),
        "negative_baseline_targets": {
            str(target): float(r2) for target, r2 in negative_baseline.items()
        },
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

    if stats["negative_baseline_targets"]:
        logger.warning(
            "targets with negative reduced R2 (flag in writeup, do not average "
            "in silently): %s",
            stats["negative_baseline_targets"],
        )

    logger.info("--- Source A marginal contribution by representation ---")
    for key, summary in stats["by_representation"].items():
        logger.info(
            "%-24s (%4d cols) full-basket mean %+.5f | decision-basket mean %+.5f "
            "| geo-adjusted decision-basket mean %+.5f | positive on %d/%d",
            key,
            summary["n_columns"],
            summary["mean_contribution"],
            summary["decision_basket_mean_contribution"],
            summary["decision_basket_mean_contribution_geo"],
            summary["n_positive"],
            summary["n_targets"],
        )
    selected = stats["by_representation"][SELECTED_ARM]["bootstrap"]
    logger.info(
        "--- %s: 95%% intervals over %d targets in %d ACS tables, %d replicates ---",
        SELECTED_ARM,
        selected["n_targets"],
        selected["n_tables"],
        selected["n_replicates"],
    )
    for scheme in ("naive", "table_clustered"):
        for statistic in (
            "contribution",
            "contribution_geo",
            f"minus_{BOOTSTRAP_REFERENCE_ARM}",
        ):
            interval = selected[scheme][statistic]
            logger.info(
                "%-16s %-24s %+.5f [%+.5f, %+.5f]%s",
                scheme,
                statistic,
                interval["point"],
                interval["low"],
                interval["high"],
                "  (covers zero)" if interval["covers_zero"] else "",
            )

    logger.info("wrote %s and %s", OUTPUT_CSV_PATH, OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
