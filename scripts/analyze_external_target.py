"""Does E_macro predict anything outside itself, for a county it has never seen?

The first non-circular test in this project. Every other crossvalidation here
predicts one pillar's feature from the other five, which measures whether six
federal sources agree rather than whether any of them is useful. This one scores
the matrix against three county-level outcomes drawn from outside all six
pillars (`ingest_external_targets.py`).

## The question the design is built around

The downstream consumer joins on DMA and holds millions of impressions per
market, so it can estimate a 210-level geographic fixed effect precisely and for
free. Any static geo-keyed feature is exactly collinear with that effect by
construction, which means **no cross-sectional correlation in this repo is
evidence against it** (`docs/plans/dma_regrain.md`, problem 1).

A fixed effect has one weakness: it has no parameter for a unit it has never
seen. So the test is extrapolation to held-out geography.

    grand_mean   intercept only -- and precisely what a geo fixed effect
                 predicts for a held-out unit, which is why it is the bar
    size         log population, log AGI, log GDP
    emacro       all pillar features
    size+emacro  both

## The drop-one arm

Added 2026-08-12 for `docs/pillar_status.md`. Alongside the four models above,
one model per pillar re-scores `size+emacro` with that pillar's whole block
withheld, and the difference is what the block was worth:

    contribution(P) = R2(size + all pillars) - R2(size + all pillars except P)

This is the fair form of the question Source F failed on raw pairwise
correlation. F's r = 0.495 against Source D tonnage collapsing to -0.057 under a
size control says nothing about whether F's block explains variance the rest of
the matrix leaves on the table, and against an external target rather than
another pillar's column. Every pillar takes the same test, because a test only
the suspect sits proves nothing either way.

Two extra arms ride along. `size_emacro_drop_BE` withholds Sources B and E
together, which closes `docs/PROJECT_GOAL.md` open decision #2: a joint cost
close to the sum of the separate costs means the two are complementary, a much
smaller one means they are substantially redundant. And because Source A's
`has_metro_attachment` restates Source F's `metro_2023`, F's contribution is
also measured with that column removed from both sides -- otherwise Source A
stands in for the block under test and F is charged for A's redundancy.

Scored as pooled out-of-fold R2 under **GroupKFold on `state_fips`**, so every
evaluated county sits in a state the model never trained on. Two consequences,
both deliberate: state dummies are excluded from every design, because under
state-blocked folds a state fixed effect degenerates to the grand mean for
held-out states -- which is the whole argument, made concrete -- and counties are
spatially autocorrelated, so random k-fold would put a neighbour of every test
county into train and inflate every number here. The repo's other sweeps use
random `KFold`; that is defensible when the question is association and wrong
when the question is extrapolation.

**The headline quantity is `size+emacro` minus `size` on held-out states.** It
answers: for a market with no history, how much of the between-county variation
does E_macro recover beyond how big the place is?

## The geography control

Added 2026-08-24. `analyze_source_a_representation_marginal.py` found that two
columns of county centroid latitude and longitude reproduce 96% of the measured
gain from Source A's best encoder arm, which reframed that pillar from "encodes
economic content" to "encodes where the county is". That control was applied to
Source A's representation arms and to nothing else, so the drop-one figure every
pillar verdict rests on had never been asked the same question.

It is asked here. Alongside every model above, a parallel `geo` family carries
two centroid columns in **both** the full and the reduced design:

    contribution_geo(P) = R2(size + lat/lon + all pillars)
                        - R2(size + lat/lon + all pillars except P)

Geography sits on both sides deliberately, and that is a different construction
from the representation script's `contribution_geo`, which puts lat/lon in the
reduced model only in order to ask whether one arm beats coordinates outright.
The question here is not whether a pillar beats geography; it is what a pillar
is still worth to somebody who already knows where the county is -- which is the
position the consuming team is actually in, since a DMA fixed effect encodes
location by construction.

A pillar whose contribution survives the control carries something other than
position on the map. A pillar whose contribution collapses under it was being
paid for geography, and the five ACS proxies are all strongly spatial.

## Intervals on the drop-one figure

Also added 2026-08-24, and for the same reason: the representation section
reports bootstrap intervals on its arms while the pillar-worth figure reports
six means to four decimal places with nothing attached. `A4` already concedes
that Sources B and C cannot be separated, which is a statement about an interval
made without one.

`bootstrap_drop_one` resamples **targets**, the unit the headline mean is taken
over, paired across pillars so an interval on one pillar's lead over another is
not inflated by target-level variance both share. Two schemes are reported: a
naive resample, and one clustered on the ACS table each target is drawn from,
because the basket carries five heating-fuel shares from `b25040` alone. No
model is re-fitted; the per-target contributions `score_target` already computed
are what get resampled.

## Two baskets, reported side by side

The five targets this script originally scored are still reported as their own
basket, under `headline_basket`, because the +0.190 headline and every number in
the notebook's pillar-worth section were measured on them. Everything else is
reported on the full basket `ingest_external_targets` now supplies. Quoting a
number from one against a number from the other is the error the notebook's
evidence-basket table exists to prevent.

## Two decompositions the report carries

**By population decile.** The case for joining at county rather than DMA grain
is that fixed effects fail on thin units, and small counties are the public proxy
for thin. If E_macro's advantage concentrates in the low deciles, that argument
is measured rather than asserted. A pooled average would hide it in either
direction.

**By training-set size.** The DMA penalty has two separable halves: fewer rows,
and aggregation blurring within-market variation. The first is measurable now by
retraining on random county subsets down to n = 210, which is the DMA row count.
The second needs the re-derivation layer in `docs/plans/dma_regrain.md` Phase 1B
and is *not* measured here. Reporting the first without naming the second would
overstate what this script settles.

## What this cannot do

The targets are public proxies, not the consumer's label, which is unobtainable
(`docs/PROJECT_GOAL.md`, "Operating constraints"). **This answers the
fixed-effect objection by analogy, not directly**, and any writeup must say so.
The three are also cross-sectional and single-period, so temporal transfer --
one of the five things a fixed effect genuinely fails at -- is untested.

One target carries a definitional overlap that must travel with its number:
`median_household_income` against a size baseline containing `log_agi`, which is
built from Source E's own AGI totals. Treat that row as a sanity check -- if the
matrix cannot predict income, something is broken -- rather than as evidence.

Outputs: `outputs/external_target_scores.csv`,
`outputs/external_target_by_decile.csv`,
`analysis-output/cross-source/external_target_stats.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from county_population import fetch_county_population
from ingest_external_targets import EXTERNAL_TARGETS, fetch_external_targets
from pillar_matrix import SIZE_FEATURES, build_matrix

# Matches every other crossvalidation script in this project.
RANDOM_SEED: int = 42
N_FOLDS: int = 5
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)

# Population deciles for the thin-unit decomposition.
N_DECILES: int = 10

# Permutations behind the drop-one noise floor. Twenty is enough to place a
# floor that sits near zero against contributions an order of magnitude larger;
# it is not enough to resolve a borderline one, and the writeup should say so
# rather than quote the p-value as though it were finely resolved.
N_PLACEBO_REPS: int = 20

# Training-set sizes for the row-count sensitivity. 210 is the DMA count, which
# is the comparison the grain question turns on; the rest bracket it.
SUBSAMPLE_SIZES: tuple[int, ...] = (210, 400, 800, 1600, 3000)
N_SUBSAMPLE_REPS: int = 10

# Bootstrap replicates behind every reported interval. Matches
# `analyze_source_a_representation_marginal.py`, which resamples the same way
# over an overlapping basket -- a different replicate count between the two
# would make their intervals incomparable for no reason.
N_BOOTSTRAP: int = 10_000
BOOTSTRAP_PERCENTILES: tuple[float, float] = (2.5, 97.5)

# The five targets this script scored before `ingest_external_targets` widened
# the basket to 42. Every number in the notebook's pillar-worth section was
# measured on these, so they keep their own summary block rather than being
# silently absorbed into a wider mean that would move the headline without
# saying so.
# Targets whose baseline model is itself worse than predicting the mean. A
# "contribution" measured on one of these is the gap between two useless fits
# rather than a gain, so they are scored and reported per target but kept out of
# every headline mean.
#
# This lived in `analyze_source_a_representation_marginal.py` and now lives here
# instead, because both sweeps draw the same basket and only one of them was
# applying it: the drop-one figure was averaging `no_fuel_used_share` into six
# pillar verdicts while the representation section next to it excluded the same
# target by name. The representation script imports this rather than keeping its
# own copy, so the two cannot drift apart.
#
# The reason text is unchanged and carries the original diagnosis: the earlier
# "Puerto Rico sentinel contamination" explanation was checked and does not hold
# (0 of 3,144 panel rows carry state_fips 72), so the target is simply too
# degenerate for this model class rather than mis-ingested.
EXCLUDED_TARGETS: dict[str, str] = {
    "no_fuel_used_share": (
        "degenerate target: reduced R2 -1.0031, full R2 -0.9720 to -1.0381 "
        "depending on arm -- every model scored is worse than predicting the "
        "mean, so the reported positive contribution is the gap between two "
        "useless models, not a gain. Panel mean 0.0067, sd 0.0232, max 0.644, "
        "skew 23.0, kurtosis 573; PR is not in the panel (0/3144 rows), so this "
        "is not sentinel/PR contamination -- the target is genuinely too "
        "degenerate for this model class. Kept ingested and scored (see the "
        "per-target table) but excluded from every headline mean."
    ),
}

HEADLINE_TARGETS: tuple[str, ...] = (
    "broadband_rate",
    "median_household_income",
    "median_age",
    "median_home_value",
    "mean_commute_minutes",
)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"
CENTROIDS_PATH: Path = REPO_ROOT / "data" / "county_centroids.parquet"

# Two float columns standing in for "where the county is". Deliberately the
# cheapest possible geography: the argument this control makes is that a
# competitor gets the same predictive value without a model download, so a
# richer encoding of location would weaken rather than strengthen it.
GEO_FEATURES: tuple[str, ...] = ("lat", "lon")

# Target -> ACS table, for the clustered resample. Read off the target
# definitions rather than re-typed here, so a target added upstream lands in
# its cluster without an edit in this file.
TARGET_TABLES: dict[str, str] = {target.column: target.table for target in EXTERNAL_TARGETS}

SCORES_PATH: Path = OUTPUTS_DIR / "external_target_scores.csv"
DECILE_PATH: Path = OUTPUTS_DIR / "external_target_by_decile.csv"
PLACEBO_PATH: Path = OUTPUTS_DIR / "external_target_drop_one_placebo.csv"
# Written so the stats artifact is rebuildable from disk in full. Every other
# frame behind it already had a CSV; this one only ever reached the JSON, so a
# summary-only rebuild used to have to scavenge it back out of the old artifact.
TRAINING_SIZE_PATH: Path = OUTPUTS_DIR / "external_target_by_training_size.csv"
STATS_PATH: Path = ANALYSIS_DIR / "external_target_stats.json"

logger = logging.getLogger(__name__)


# Pillar columns that restate a target by construction rather than predicting it.
# Same principle as RESTATEMENT_COLUMNS in `analyze_pillar_matrix_signal.py`: two
# products measuring one underlying fact is bookkeeping, not evidence.
#
# `wage_per_return_thousands` is Source E's average wage income per tax return,
# which is close to a definition of median household income rather than a
# prediction of it. `retirement_destination` is USDA's code for counties with
# high net in-migration of people aged 60 and over, so it restates age structure.
#
# Every target is reported twice, with and without its restatements. The ablated
# figure is the defensible one.
# `median_home_value` and `mean_commute_minutes` have no entry because no pillar
# column measures either construct. Source F's `housing_stress` is a cost-burden
# share rather than a value, which is related but not definitional, so it stays
# in the design.
#
# Targets not listed here at all -- every "clean" verdict in
# `ingest_external_targets.TARGET_CIRCULARITY` -- fall through to the
# empty-tuple default on `.get()` below, same as the three explicit empty
# entries above.
#
# The Task 3 basket expansion added seven more ablations, all against Source E
# and Source F columns already in use here:
#
#   - `per_capita_income` and `median_family_income` restate
#     `wage_per_return_thousands`, on the same grounds as
#     `median_household_income`.
#   - `median_monthly_housing_cost` restates `housing_stress`. Unlike
#     `median_home_value` above, this target is a cost rather than a value,
#     which `housing_stress` measures directly.
#   - `poverty_rate` restates `persistent_poverty`, Source F's decade-scale
#     poverty flag.
#   - `bachelors_share` and `masters_share` both restate `low_postsecondary_ed`,
#     Source F's below-threshold postsecondary-attainment flag.
#   - `labor_force_participation` restates `low_employment`, Source F's flag
#     for counties below the labour-force-participation threshold.
#   - `household_ss_income_share` (share of households receiving Social
#     Security income) restates `retirement_destination` on the same reasoning
#     already applied to `median_age`: it is at least as strong an age-
#     structure proxy as median age itself.
#   - `household_earnings_share` (share of households with wage or salary
#     income) restates `low_employment`, Source F's county-level binary
#     threshold flag on the employment rate of 25-54 year olds --
#     `household_earnings_share` is the same construct measured continuously
#     at household level, closer to `labor_force_participation` above than to
#     the commute-mode targets that were correctly kept clean.
TARGET_RESTATEMENTS: dict[str, tuple[str, ...]] = {
    "median_household_income": ("wage_per_return_thousands",),
    "median_age": ("retirement_destination",),
    "broadband_rate": (),
    "median_home_value": (),
    "mean_commute_minutes": (),
    "per_capita_income": ("wage_per_return_thousands",),
    "median_family_income": ("wage_per_return_thousands",),
    "median_monthly_housing_cost": ("housing_stress",),
    "poverty_rate": ("persistent_poverty",),
    "bachelors_share": ("low_postsecondary_ed",),
    "masters_share": ("low_postsecondary_ed",),
    "labor_force_participation": ("low_employment",),
    "household_ss_income_share": ("retirement_destination",),
    "household_earnings_share": ("low_employment",),
}


# Source A's `has_metro_attachment` fires when a Wikipedia intro states the
# county belongs to a metropolitan statistical area, which is the OMB
# delineation Source F's `metro_2023` reports directly. It therefore sits inside
# the reduced design when Source F's block is dropped, covering for part of what
# F was carrying and understating F's measured contribution. The two models
# below remove it from both sides so F's contribution can be read without A
# standing in for it.
SOURCE_F_PROXY_IN_A: tuple[str, ...] = ("has_metro_attachment",)


@dataclass(frozen=True)
class ModelSpec:
    """One predictor set to be scored against every target.

    Attributes:
        name: Short identifier used in outputs.
        label: Human-readable description used in reports.
        uses_size: Whether the size features enter the design.
        uses_pillars: Whether the pillar feature blocks enter the design.
        uses_geo: Whether county centroid latitude and longitude enter the
            design. Set on both sides of a geo-family difference, so the
            contribution it states is net of location rather than a contest
            against it.
        drop_pillars: Pillar letters whose blocks are withheld, for the
            drop-one-pillar arm.
        drop_columns: Individual columns withheld on top of `drop_pillars`.
        reference: Model this one is differenced against to state a
            contribution. Empty for models that are not part of the drop-one
            arm.
    """

    name: str
    label: str
    uses_size: bool
    uses_pillars: bool
    uses_geo: bool = False
    drop_pillars: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    reference: str = ""


# The drop-one arm. `docs/pillar_status.md` asks whether Source F earns its slot
# once raw pairwise correlation is set aside; the honest form of that question is
# whether the block explains variance the rest of the matrix does not, measured
# against a target outside all six pillars. Every pillar takes the same test --
# a test only the suspect sits is not a fair test, and the symmetric table is
# what a go/no-go needs anyway.
#
# `size_emacro_drop_BE` closes `docs/PROJECT_GOAL.md` open decision #2 in the
# same run: if dropping B and E together costs about what dropping each
# separately costs, the two pillars are complementary; if it costs much less,
# they are substantially redundant and the effective pillar count is five.
DROP_MODELS: tuple[ModelSpec, ...] = tuple(
    ModelSpec(
        f"size_emacro_drop_{pillar}",
        f"size + E_macro pillars, Source {pillar} withheld",
        True,
        True,
        drop_pillars=(pillar,),
        reference="size_emacro",
    )
    for pillar in "ABCDEF"
) + (
    ModelSpec(
        "size_emacro_drop_BE",
        "size + E_macro pillars, Sources B and E both withheld",
        True,
        True,
        drop_pillars=("B", "E"),
        reference="size_emacro",
    ),
    ModelSpec(
        "size_emacro_no_ametro",
        "size + E_macro pillars, Source A's metro restatement withheld",
        True,
        True,
        drop_columns=SOURCE_F_PROXY_IN_A,
        reference="size_emacro",
    ),
    ModelSpec(
        "size_emacro_drop_F_no_ametro",
        "size + E_macro pillars, Source F and A's metro restatement both withheld",
        True,
        True,
        drop_pillars=("F",),
        drop_columns=SOURCE_F_PROXY_IN_A,
        # Differenced against the model that has already lost A's restatement,
        # so what remains is Source F's own contribution rather than the two
        # removals compounded.
        reference="size_emacro_no_ametro",
    ),
)

# The geography control. Same drop-one difference as `DROP_MODELS`, with two
# centroid columns present in the full and the reduced design alike, so what it
# reports is what each pillar adds to somebody who already knows where the
# county is.
#
# `size_geo` is carried so the whole matrix's lift can be restated against a
# geography-aware baseline too, not only each pillar's slice of it: a headline
# of +0.190 over size means something different if a third of it is available
# from two floats.
GEO_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("size_geo", "size + lat/lon", True, False, uses_geo=True),
    # No `reference`. `contribution` is defined as reference minus self, which
    # states a withheld block's worth because self is the *reduced* model. This
    # arm is the fuller one, so the same subtraction would report the negative
    # of its lift. `geo_control_summary` differences it in the right direction
    # instead.
    ModelSpec("size_geo_emacro", "size + lat/lon + E_macro pillars", True, True, uses_geo=True),
) + tuple(
    ModelSpec(
        f"size_geo_emacro_drop_{pillar}",
        f"size + lat/lon + E_macro pillars, Source {pillar} withheld",
        True,
        True,
        uses_geo=True,
        drop_pillars=(pillar,),
        reference="size_geo_emacro",
    )
    for pillar in "ABCDEF"
)

# Order matters: `score_by_training_size` indexes MODELS[1] and MODELS[3] for the
# size and size+E_macro designs, so the four original specs stay at the front.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("grand_mean", "intercept only (= fixed effect on an unseen unit)", False, False),
    ModelSpec("size", "county size only", True, False),
    ModelSpec("emacro", "E_macro pillars only", False, True),
    ModelSpec("size_emacro", "size + E_macro pillars", True, True),
) + DROP_MODELS + GEO_MODELS

# `size_geo_emacro_drop_{P}` against `size_emacro_drop_{P}`: the pillar letter a
# geo-family drop model is testing, for the side-by-side table. Keyed by model
# name so `drop_one_summary` does not have to re-parse names.
GEO_DROP_MODELS: dict[str, str] = {
    f"size_geo_emacro_drop_{pillar}": pillar for pillar in "ABCDEF"
}


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_panel() -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Join the pillar matrix, the external targets, and county population.

    Returns:
        Tuple of (panel, pillar_columns, blocks). `panel` carries `fips_code`,
        `state_fips`, `population`, `lat`/`lon`, every size and pillar feature,
        and one column per external target. `blocks` maps pillar letter to its
        column list, which the drop-one models need.

    Raises:
        FileNotFoundError: If any pillar parquet or the target cache is absent.
        ValueError: If a panel county has no centroid. The geography control is
            not optional and a silently null `lat` would be imputed to the
            national median, which reads as a real location rather than a
            missing one.
    """
    matrix, blocks = build_matrix()
    pillar_columns = [column for columns in blocks.values() for column in columns]

    targets = fetch_external_targets()
    # Each target ships a `_se` companion carrying its ACS standard error, which
    # `score_by_decile` needs to compute the per-stratum noise floor.
    target_columns = [target.column for target in EXTERNAL_TARGETS]
    target_columns += [f"{column}_se" for column in target_columns]
    population = fetch_county_population()[["fips_code", "population"]]
    centroids = pd.read_parquet(CENTROIDS_PATH)[["fips_code", *GEO_FEATURES]]

    panel = (
        matrix.merge(targets[["fips_code", *target_columns]], on="fips_code", how="inner")
        .merge(population, on="fips_code", how="inner")
        .merge(centroids, on="fips_code", how="left")
        .reset_index(drop=True)
    )
    missing_centroid = int(panel[list(GEO_FEATURES)].isna().any(axis=1).sum())
    if missing_centroid:
        raise ValueError(
            f"{missing_centroid} panel counties absent from {CENTROIDS_PATH.name}"
        )
    logger.info(
        "panel: %d counties x %d pillar features, %d targets",
        len(panel),
        len(pillar_columns),
        len(target_columns),
    )
    return panel, pillar_columns, blocks


def withheld_columns(model: ModelSpec, blocks: dict[str, list[str]]) -> tuple[str, ...]:
    """List the pillar columns a drop-one model holds out of its design.

    Args:
        model: The model being assembled.
        blocks: Pillar-to-columns mapping from `build_matrix`.

    Returns:
        Column names to withhold, empty for the four original models.
    """
    dropped = [column for pillar in model.drop_pillars for column in blocks[pillar]]
    dropped.extend(column for column in model.drop_columns if column not in dropped)
    return tuple(dropped)


def build_design(
    panel: pd.DataFrame,
    model: ModelSpec,
    pillar_columns: list[str],
    ablate: tuple[str, ...] = (),
) -> np.ndarray:
    """Assemble one model's predictor array.

    The intercept-only design is a single constant column rather than an empty
    array, so the same pipeline scores every model without branching.

    Args:
        panel: Joined panel from `load_panel`.
        model: The predictor set to assemble.
        pillar_columns: Every pillar feature column name.
        ablate: Pillar columns to drop, for the restatement-free variant.

    Returns:
        Float array of shape (n_counties, n_predictors).
    """
    columns: list[str] = []
    if model.uses_size:
        columns.extend(SIZE_FEATURES)
    if model.uses_geo:
        columns.extend(GEO_FEATURES)
    if model.uses_pillars:
        columns.extend(column for column in pillar_columns if column not in ablate)
    if not columns:
        return np.ones((len(panel), 1), dtype=float)
    return panel[columns].astype(float).to_numpy()


def _pipeline() -> Pipeline:
    """Build the impute-scale-ridge pipeline used for every fit.

    Median imputation with a missingness indicator, matching
    `analyze_pillar_matrix_signal.py`: BLS suppresses a large share of the LQ
    cells and "suppressed" is itself informative, so the indicator has to
    survive into the model rather than being imputed away.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=RIDGE_ALPHAS)),
        ]
    )


def out_of_fold_predictions(
    design: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Predict every observation from folds that exclude its whole state.

    Args:
        design: Predictor array.
        y: Target vector.
        groups: State FIPS per row, used as the grouping variable.

    Returns:
        Out-of-fold prediction per row.
    """
    folds = GroupKFold(n_splits=N_FOLDS)
    return cross_val_predict(_pipeline(), design, y, cv=folds, groups=groups)


def score_target(
    panel: pd.DataFrame,
    pillar_columns: list[str],
    blocks: dict[str, list[str]],
    column: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score every model against one external target.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        blocks: Pillar-to-columns mapping, for the drop-one models.
        column: Target column name.
        label: Human-readable target description.

    Returns:
        Tuple of (scores, predictions). `scores` has one row per model;
        `predictions` maps model name to its out-of-fold prediction vector,
        aligned to the target's non-null subset.
    """
    usable = panel[panel[column].notna()].reset_index(drop=True)
    y = usable[column].astype(float).to_numpy()
    groups = usable["state_fips"].to_numpy()

    ablate = TARGET_RESTATEMENTS.get(column, ())
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for model in MODELS:
        withheld = withheld_columns(model, blocks)
        design = build_design(usable, model, pillar_columns, ablate=withheld)
        predicted = out_of_fold_predictions(design, y, groups)
        predictions[model.name] = predicted

        if ablate and model.uses_pillars:
            ablated_design = build_design(
                usable, model, pillar_columns, ablate=tuple(ablate) + withheld
            )
            ablated_r2 = float(
                r2_score(y, out_of_fold_predictions(ablated_design, y, groups))
            )
        else:
            ablated_r2 = float(r2_score(y, predicted))

        rows.append(
            {
                "target": column,
                "target_label": label,
                "model": model.name,
                "model_label": model.label,
                "n": len(y),
                "n_predictors": design.shape[1],
                "r2_out_of_state": float(r2_score(y, predicted)),
                "r2_ablated": ablated_r2,
                "ablated_columns": ";".join(ablate) if model.uses_pillars else "",
                "withheld_pillars": ";".join(model.drop_pillars),
                "reference_model": model.reference,
                "rmse": float(np.sqrt(np.mean((y - predicted) ** 2))),
            }
        )

    scores = pd.DataFrame(rows)
    by_model = scores.set_index("model")
    scores["lift_over_size"] = scores["r2_out_of_state"] - by_model.loc["size", "r2_out_of_state"]
    scores["lift_over_size_ablated"] = scores["r2_ablated"] - by_model.loc["size", "r2_ablated"]
    scores["lift_over_grand_mean"] = (
        scores["r2_out_of_state"] - by_model.loc["grand_mean", "r2_out_of_state"]
    )
    # What the withheld block was worth: how much R2 the full model loses
    # without it. Positive means the block carried something the rest of the
    # matrix did not supply. The ablated column is the defensible one, on the
    # same argument §3 of `external-target-findings.md` already makes.
    scores["contribution"] = [
        by_model.loc[row.reference_model, "r2_out_of_state"] - row.r2_out_of_state
        if row.reference_model
        else np.nan
        for row in scores.itertuples()
    ]
    scores["contribution_ablated"] = [
        by_model.loc[row.reference_model, "r2_ablated"] - row.r2_ablated
        if row.reference_model
        else np.nan
        for row in scores.itertuples()
    ]
    return scores, predictions


def score_placebo(
    panel: pd.DataFrame,
    pillar_columns: list[str],
    blocks: dict[str, list[str]],
    column: str,
    contributions: dict[str, float],
) -> pd.DataFrame:
    """Measure what a block of the same shape carrying no county alignment appears to add.

    The noise floor the drop-one verdict is judged against. For each pillar, the
    reduced design (that pillar withheld) gets the block added back with its rows
    permuted: county alignment is destroyed while every column keeps its marginal
    distribution and the design keeps its width. Whatever apparent contribution
    survives that is what the measurement produces from nothing.

    The permutation is applied inside the design rather than to the target, so
    the size features and the other pillars stay aligned to `y` and only the
    block under test is scrambled.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        blocks: Pillar-to-columns mapping from `build_matrix`.
        column: Target column name.
        contributions: Measured ablated contribution per pillar letter, used to
            report how often a shuffled block matches or beats the real one.

    Returns:
        DataFrame with one row per pillar.
    """
    usable = panel[panel[column].notna()].reset_index(drop=True)
    y = usable[column].astype(float).to_numpy()
    groups = usable["state_fips"].to_numpy()
    ablate = TARGET_RESTATEMENTS.get(column, ())

    rows: list[dict[str, object]] = []
    for pillar, block_columns in blocks.items():
        kept = [
            col
            for col in pillar_columns
            if col not in block_columns and col not in ablate
        ]
        shuffled_columns = [col for col in block_columns if col not in ablate]
        reduced = np.hstack(
            [
                usable[list(SIZE_FEATURES)].astype(float).to_numpy(),
                usable[kept].astype(float).to_numpy(),
            ]
        )
        r2_reduced = float(r2_score(y, out_of_fold_predictions(reduced, y, groups)))

        rng = np.random.default_rng(RANDOM_SEED)
        block_values = usable[shuffled_columns].astype(float).to_numpy()
        placebo = np.empty(N_PLACEBO_REPS)
        for rep in range(N_PLACEBO_REPS):
            permuted = block_values[rng.permutation(len(usable))]
            design = np.hstack([reduced, permuted])
            placebo[rep] = (
                float(r2_score(y, out_of_fold_predictions(design, y, groups))) - r2_reduced
            )

        measured = contributions.get(pillar, float("nan"))
        rows.append(
            {
                "target": column,
                "pillar": pillar,
                "n_reps": N_PLACEBO_REPS,
                "contribution_ablated": measured,
                "placebo_mean": float(placebo.mean()),
                "placebo_p95": float(np.percentile(placebo, 95)),
                "placebo_max": float(placebo.max()),
                "p": float((np.sum(placebo >= measured) + 1) / (N_PLACEBO_REPS + 1)),
            }
        )
    return pd.DataFrame(rows)


def score_by_decile(
    panel: pd.DataFrame,
    column: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Break one target's out-of-fold error down by county population decile.

    Carries the **sampling-noise floor** alongside the fit. ACS estimates in
    small counties are noisy, so part of the within-decile variance is sampling
    error that no model can explain. Without that quantified, a model scoring
    badly on small counties is indistinguishable from a target that is mostly
    noise there -- the ambiguity that left the first round of this analysis
    unable to settle whether E_macro helps more or less on thin units.

    `r2_ceiling` is `1 - mean(SE^2) / var(y)` within the decile: the largest R2
    any model could reach against a target measured this precisely.
    `share_of_explainable` divides the observed R2 by that ceiling, and is the
    figure that compares fairly across deciles.

    Args:
        panel: Joined panel from `load_panel`.
        column: Target column name.
        predictions: Out-of-fold predictions per model, from `score_target`.

    Returns:
        DataFrame with one row per decile.
    """
    usable = panel[panel[column].notna()].reset_index(drop=True)
    y = usable[column].astype(float).to_numpy()
    standard_error = usable[f"{column}_se"].astype(float).to_numpy()
    decile = pd.qcut(usable["population"], N_DECILES, labels=False, duplicates="drop")

    rows: list[dict[str, object]] = []
    for index in sorted(pd.unique(decile.dropna())):
        mask = (decile == index).to_numpy()
        observed = y[mask]
        total_variance = float(np.var(observed, ddof=1))
        noise_variance = float(np.nanmean(standard_error[mask] ** 2))

        row: dict[str, object] = {
            "target": column,
            "population_decile": int(index) + 1,
            "n": int(mask.sum()),
            "median_population": float(usable.loc[mask, "population"].median()),
            "variance_total": total_variance,
            "variance_sampling_noise": noise_variance,
            "noise_share": noise_variance / total_variance if total_variance > 0 else np.nan,
        }
        row["r2_ceiling"] = 1.0 - float(row["noise_share"])

        for name, predicted in predictions.items():
            residual = observed - predicted[mask]
            row[f"rmse_{name}"] = float(np.sqrt(np.mean(residual**2)))
            # R2 against the decile's own mean, so deciles are comparable.
            row[f"r2_{name}"] = (
                1.0 - float(np.mean(residual**2)) / total_variance
                if total_variance > 0
                else np.nan
            )

        size_rmse = float(row["rmse_size"])
        combined_rmse = float(row["rmse_size_emacro"])
        row["rmse_reduction"] = (
            (size_rmse - combined_rmse) / size_rmse if size_rmse > 0 else np.nan
        )
        ceiling = float(row["r2_ceiling"])
        row["share_of_explainable"] = (
            float(row["r2_size_emacro"]) / ceiling if ceiling > 0 else np.nan
        )
        # Lift over the baseline a consumer holding only population would have.
        # This and `rmse_reduction` diverge by construction and answer different
        # questions: proportional error reduction is scale-free, while variance
        # explained is measured against a decile's own spread, which is much
        # wider among small counties.
        row["r2_lift_over_size"] = float(row["r2_size_emacro"]) - float(row["r2_size"])
        rows.append(row)
    return pd.DataFrame(rows)


def score_by_training_size(
    panel: pd.DataFrame, pillar_columns: list[str], column: str
) -> pd.DataFrame:
    """Measure how E_macro's lift over size degrades as training rows are removed.

    Isolates the row-count half of the DMA grain penalty. The aggregation half --
    within-market variation being averaged away -- is not measured here and needs
    `docs/plans/dma_regrain.md` Phase 1B.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        column: Target column name.

    Returns:
        DataFrame with one row per training size, carrying mean lift over the
        size baseline across N_SUBSAMPLE_REPS random subsets.
    """
    usable = panel[panel[column].notna()].reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for size in SUBSAMPLE_SIZES:
        if size >= len(usable):
            continue
        lifts: list[float] = []
        for rep in range(N_SUBSAMPLE_REPS):
            sample = usable.sample(n=size, random_state=RANDOM_SEED + rep).reset_index(drop=True)
            if sample["state_fips"].nunique() < N_FOLDS:
                continue
            y = sample[column].astype(float).to_numpy()
            groups = sample["state_fips"].to_numpy()
            baseline = out_of_fold_predictions(
                build_design(sample, MODELS[1], pillar_columns), y, groups
            )
            combined = out_of_fold_predictions(
                build_design(sample, MODELS[3], pillar_columns), y, groups
            )
            lifts.append(float(r2_score(y, combined) - r2_score(y, baseline)))
        if lifts:
            rows.append(
                {
                    "target": column,
                    "n_train_units": size,
                    "n_reps": len(lifts),
                    "mean_lift_over_size": float(np.mean(lifts)),
                    "sd_lift_over_size": float(np.std(lifts, ddof=1)) if len(lifts) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def drop_one_summary(scores: pd.DataFrame) -> dict[str, object]:
    """Collapse the drop-one arm to one verdict row per withheld model.

    Args:
        scores: Per-target, per-model scores from `score_target`.

    Returns:
        Mapping of model name to its mean contribution across targets, the
        count of targets where the contribution is positive, and the per-target
        contributions themselves.
    """
    # `.astype(bool)` would be wrong here. In memory a model with no reference
    # carries "", which is falsy -- but round-tripped through CSV that cell
    # reads back as NaN, and NaN is *truthy*, so every referenceless model
    # (`size`, `emacro`, `grand_mean`, `size_geo`) would be summarized as though
    # it stated a contribution. Compare against the empty string instead, so a
    # rebuild from the CSVs produces exactly what the in-memory run produced.
    has_reference = scores["reference_model"].fillna("").astype(str).str.len() > 0
    dropped = scores[has_reference]
    summary: dict[str, object] = {}
    for name, frame in dropped.groupby("model"):
        summary[str(name)] = {
            "withheld_pillars": frame["withheld_pillars"].iloc[0],
            "reference_model": frame["reference_model"].iloc[0],
            "n_targets": int(len(frame)),
            "mean_contribution": float(frame["contribution"].mean()),
            "mean_contribution_ablated": float(frame["contribution_ablated"].mean()),
            "n_positive_ablated": int((frame["contribution_ablated"] > 0).sum()),
            "by_target": {
                row.target: float(row.contribution_ablated) for row in frame.itertuples()
            },
        }
    return summary


def _draw_target_positions(
    rng: np.random.Generator, targets: list[str], cluster_by_table: bool
) -> np.ndarray:
    """Draw one bootstrap resample of target positions.

    Args:
        rng: Seeded generator. Called once per replicate.
        targets: Basket targets, in the column order of the score matrices.
        cluster_by_table: Resample whole ACS tables when True, individual
            targets when False.

    Returns:
        Positions into `targets`, drawn with replacement. The naive draw is
        exactly `len(targets)` long; the clustered draw's length varies by
        replicate, because ACS tables hold different numbers of targets and a
        replicate drawing `b25040` twice carries ten heating-fuel rows.
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
            alongside rather than read off the draws, because a percentile
            interval is not required to be centred on it.

    Returns:
        Point estimate, both bounds, and whether the interval covers zero --
        the last because "indistinguishable from zero" is a different claim
        from "small", and prose quoting this needs to know which it has.
    """
    low, high = np.percentile(draws, BOOTSTRAP_PERCENTILES)
    return {
        "point": float(point),
        "low": float(low),
        "high": float(high),
        "covers_zero": bool(low <= 0.0 <= high),
    }


def _contribution_matrix(
    scores: pd.DataFrame, models: dict[str, str], targets: list[str]
) -> np.ndarray:
    """Lay one model family's ablated contributions out as pillar x target.

    Args:
        scores: Per-target, per-model scores from `score_target`.
        models: Model name to the pillar letter it withholds.
        targets: Basket targets, defining column order.

    Returns:
        Array of shape (len(models), len(targets)), pillar-sorted by letter.

    Raises:
        ValueError: If some model did not score some basket target, which would
            silently pair different pillars against different baskets.
    """
    position = {target: index for index, target in enumerate(targets)}
    pillars = sorted(models.values())
    by_pillar = {pillar: name for name, pillar in models.items()}

    matrix = np.full((len(pillars), len(targets)), np.nan)
    for index, pillar in enumerate(pillars):
        frame = scores[scores["model"] == by_pillar[pillar]]
        for row in frame.itertuples():
            if row.target in position:
                matrix[index, position[row.target]] = row.contribution_ablated
    if np.isnan(matrix).any():
        raise ValueError("every drop model must score every basket target before pairing")
    return matrix


def bootstrap_drop_one(
    scores: pd.DataFrame, targets: list[str]
) -> dict[str, object]:
    """Interval every pillar's drop-one contribution, paired across pillars.

    The resampling unit is the **target**, because the target is the unit the
    reported mean is taken over. It is not the county: counties sit inside the
    folds, and `GroupKFold` on state already accounts for them.

    Within a replicate every pillar is scored on the same draw. That pairing is
    what makes the two derived statistics readable:

    - `pairwise` gives an interval on each pillar's lead over each other
      pillar, which is the quantity `A4` currently declines to quote for B
      against C. Target-level variance both pillars share cancels.
    - `geo_minus_plain` gives an interval on how much of a pillar's
      contribution was geography, by differencing the two families on one draw
      rather than differencing two independently-resampled means.

    Nothing is re-fitted. This resamples the per-target contributions
    `score_target` already computed.

    Args:
        scores: Per-target, per-model scores from `score_target`.
        targets: Basket targets to resample over.

    Returns:
        Mapping with the resample's parameters and, per scheme, a `by_pillar`
        block and a `pairwise` block.
    """
    plain_models = {f"size_emacro_drop_{pillar}": pillar for pillar in "ABCDEF"}
    plain = _contribution_matrix(scores, plain_models, targets)
    geo = _contribution_matrix(scores, GEO_DROP_MODELS, targets)
    pillars = sorted(plain_models.values())

    observed_plain = plain.mean(axis=1)
    observed_geo = geo.mean(axis=1)

    out: dict[str, object] = {
        "n_replicates": N_BOOTSTRAP,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "n_targets": len(targets),
        "n_tables": len({TARGET_TABLES[target] for target in targets}),
        "pillars": pillars,
    }

    for scheme, cluster_by_table in (("naive", False), ("table_clustered", True)):
        # Re-seeded per scheme so each reproduces on its own, and so adding a
        # scheme never shifts an existing one's numbers.
        rng = np.random.default_rng(RANDOM_SEED)
        drawn_plain = np.empty((N_BOOTSTRAP, len(pillars)))
        drawn_geo = np.empty((N_BOOTSTRAP, len(pillars)))
        for replicate in range(N_BOOTSTRAP):
            # One draw, every pillar and both families -- this is the pairing.
            selection = _draw_target_positions(rng, targets, cluster_by_table)
            drawn_plain[replicate] = plain[:, selection].mean(axis=1)
            drawn_geo[replicate] = geo[:, selection].mean(axis=1)

        by_pillar = {
            pillar: {
                "contribution": _interval(drawn_plain[:, index], observed_plain[index]),
                "contribution_geo": _interval(drawn_geo[:, index], observed_geo[index]),
                "geo_minus_plain": _interval(
                    drawn_geo[:, index] - drawn_plain[:, index],
                    observed_geo[index] - observed_plain[index],
                ),
            }
            for index, pillar in enumerate(pillars)
        }
        pairwise = {
            f"{first}_minus_{second}": _interval(
                drawn_plain[:, i] - drawn_plain[:, j],
                observed_plain[i] - observed_plain[j],
            )
            for i, first in enumerate(pillars)
            for j, second in enumerate(pillars)
            if i < j
        }
        out[scheme] = {"by_pillar": by_pillar, "pairwise": pairwise}
    return out


def geo_control_summary(scores: pd.DataFrame, targets: list[str]) -> dict[str, object]:
    """State what survives once two centroid columns are in the baseline.

    Args:
        scores: Per-target, per-model scores from `score_target`.
        targets: Basket targets to average over.

    Returns:
        Per-pillar plain and geo-controlled contributions with the share
        retained, plus the whole-matrix lift measured both ways and what
        latitude and longitude add on their own over the size baseline.
    """
    basket = scores[scores["target"].isin(targets)]
    by_model = basket.groupby("model")

    def mean_contribution(name: str) -> float:
        return float(by_model.get_group(name)["contribution_ablated"].mean())

    per_pillar: dict[str, object] = {}
    for pillar in "ABCDEF":
        plain = mean_contribution(f"size_emacro_drop_{pillar}")
        controlled = mean_contribution(f"size_geo_emacro_drop_{pillar}")
        per_pillar[pillar] = {
            "contribution": plain,
            "contribution_geo": controlled,
            # Undefined rather than infinite when the plain figure is itself at
            # zero, which is Source A's case: "kept 3000% of nothing" is not a
            # readable sentence and would be quoted as though it were one.
            "share_retained": controlled / plain if abs(plain) > 1e-6 else None,
            "n_positive_geo": int(
                (by_model.get_group(f"size_geo_emacro_drop_{pillar}")[
                    "contribution_ablated"
                ] > 0).sum()
            ),
        }

    combined = basket[basket["model"] == "size_emacro"]
    size_geo = basket[basket["model"] == "size_geo"]

    # Differenced per target and then averaged, rather than averaging the two
    # R2 columns and subtracting: a target absent from one arm would silently
    # shift the other's mean instead of raising.
    geo_r2 = basket[basket["model"] == "size_geo"].set_index("target")["r2_ablated"]
    geo_emacro_r2 = (
        basket[basket["model"] == "size_geo_emacro"].set_index("target")["r2_ablated"]
    )
    matrix_lift_over_size_geo = float((geo_emacro_r2 - geo_r2).mean())

    return {
        "by_pillar": per_pillar,
        "geo_features": list(GEO_FEATURES),
        "matrix_lift_over_size": float(combined["lift_over_size_ablated"].mean()),
        "matrix_lift_over_size_geo": matrix_lift_over_size_geo,
        # What the two coordinate columns are worth on their own, against the
        # same size baseline the headline uses. The comparison the
        # representation section made for Source A, made for the whole matrix.
        "latlong_lift_over_size": float(size_geo["lift_over_size_ablated"].mean()),
    }


def noise_floor_summary(
    placebos: pd.DataFrame, targets: list[str]
) -> dict[str, dict[str, object]]:
    """Collapse the placebo runs to one noise floor per pillar, on one basket.

    Basket-aware because the drop-one figure is drawn on the headline basket
    and the geography control on the wide one: a floor averaged over 41 targets
    is not the bar a 5-target contribution has to clear.

    Args:
        placebos: Per-target, per-pillar placebo scores from `score_placebo`.
        targets: Basket targets to summarize over.

    Returns:
        Mapping of pillar letter to its measured contribution, mean and maximum
        placebo, and the count of targets clearing the per-target 95th
        percentile.
    """
    basket = placebos[placebos["target"].isin(targets)]
    return {
        str(pillar): {
            "n_targets": int(len(frame)),
            "mean_contribution_ablated": float(frame["contribution_ablated"].mean()),
            "mean_placebo": float(frame["placebo_mean"].mean()),
            "max_placebo": float(frame["placebo_max"].max()),
            "n_targets_above_floor": int(
                (frame["contribution_ablated"] > frame["placebo_p95"]).sum()
            ),
        }
        for pillar, frame in basket.groupby("pillar")
    }


def basket_summary(
    scores: pd.DataFrame, placebos: pd.DataFrame, targets: list[str]
) -> dict[str, object]:
    """Summarize one basket end to end: lift, drop-one ordering, noise floor.

    Every block the pillar-worth figure needs, restricted to one basket, so the
    figure can be drawn on the five original targets while the geography
    control and the intervals are reported on the wide one. Reading a number
    from one basket against a number from the other is the error the notebook's
    evidence-basket table exists to prevent, and keeping them in separate
    blocks is what makes that error visible rather than available.

    Args:
        scores: Per-target, per-model scores from `score_target`.
        placebos: Per-target, per-pillar placebo scores from `score_placebo`.
        targets: Basket targets.

    Returns:
        Mean lift over size, count of targets helped, the full drop-one block,
        and the noise floor, all on this basket alone.
    """
    basket = scores[scores["target"].isin(targets)]
    combined = basket[basket["model"] == "size_emacro"]
    return {
        "n_targets": len(targets),
        "targets": list(targets),
        "mean_lift_over_size": float(combined["lift_over_size"].mean()),
        "mean_lift_over_size_ablated": float(combined["lift_over_size_ablated"].mean()),
        "targets_with_positive_lift": int((combined["lift_over_size_ablated"] > 0).sum()),
        "drop_one": drop_one_summary(basket),
        "noise_floor": noise_floor_summary(placebos, targets),
        "geo_control": geo_control_summary(scores, targets),
        "bootstrap": bootstrap_drop_one(scores, targets),
    }


def summarize(
    scores: pd.DataFrame, deciles: pd.DataFrame, placebos: pd.DataFrame
) -> dict[str, object]:
    """Assemble the sweep-level summary written alongside the CSVs.

    Args:
        scores: Per-target, per-model scores.
        deciles: Per-target, per-decile error breakdown.

    Returns:
        JSON-serializable summary dictionary.
    """
    scored_targets = sorted(scores["target"].unique())
    # Degenerate targets are scored and reported per target, but every mean
    # below is taken over the basket without them. Averaging a gap between two
    # worse-than-mean fits into six pillar verdicts is the error the
    # representation section already refuses to make on this same basket.
    full_basket = [target for target in scored_targets if target not in EXCLUDED_TARGETS]
    headline = [
        target
        for target in HEADLINE_TARGETS
        if target in set(scored_targets) and target not in EXCLUDED_TARGETS
    ]
    basket_scores = scores[scores["target"].isin(full_basket)]
    combined = basket_scores[basket_scores["model"] == "size_emacro"]
    emacro_only = basket_scores[basket_scores["model"] == "emacro"]
    return {
        "drop_one": drop_one_summary(basket_scores),
        "drop_one_noise_floor": noise_floor_summary(placebos, full_basket),
        "geo_control": geo_control_summary(scores, full_basket),
        "bootstrap": bootstrap_drop_one(scores, full_basket),
        # The five original targets, kept as their own block so the notebook's
        # pillar-worth figure stays on the basket its prose describes after the
        # sweep widened to 42. `headline_basket` and the full-basket figures
        # above it are different baskets and are never to be read across.
        "headline_basket": basket_summary(scores, placebos, headline),
        "n_targets": len(full_basket),
        "n_targets_scored": len(scored_targets),
        "excluded_targets": dict(EXCLUDED_TARGETS),
        "n_folds": N_FOLDS,
        "fold_strategy": "GroupKFold on state_fips (spatially blocked)",
        "random_seed": RANDOM_SEED,
        "mean_r2_size_emacro": float(combined["r2_out_of_state"].mean()),
        "mean_lift_over_size": float(combined["lift_over_size"].mean()),
        "mean_lift_over_size_ablated": float(combined["lift_over_size_ablated"].mean()),
        "mean_r2_emacro_alone": float(emacro_only["r2_out_of_state"].mean()),
        "targets_with_positive_lift": int((combined["lift_over_size_ablated"] > 0).sum()),
        "by_target": {
            row["target"]: {
                "r2_size": float(
                    scores[(scores["target"] == row["target"]) & (scores["model"] == "size")][
                        "r2_out_of_state"
                    ].iloc[0]
                ),
                "r2_size_emacro": float(row["r2_out_of_state"]),
                "lift_over_size": float(row["lift_over_size"]),
                "lift_over_size_ablated": float(row["lift_over_size_ablated"]),
                "ablated_columns": row["ablated_columns"],
                "excluded_from_headline": row["target"] in EXCLUDED_TARGETS,
            }
            for _, row in scores[scores["model"] == "size_emacro"].iterrows()
        },
        "rmse_reduction_smallest_decile": float(
            deciles[deciles["population_decile"] == 1]["rmse_reduction"].mean()
        ),
        "rmse_reduction_largest_decile": float(
            deciles[deciles["population_decile"] == N_DECILES]["rmse_reduction"].mean()
        ),
        "noise_share_smallest_decile": float(
            deciles[deciles["population_decile"] == 1]["noise_share"].mean()
        ),
        "noise_share_largest_decile": float(
            deciles[deciles["population_decile"] == N_DECILES]["noise_share"].mean()
        ),
        "share_of_explainable_smallest_decile": float(
            deciles[deciles["population_decile"] == 1]["share_of_explainable"].mean()
        ),
        "share_of_explainable_largest_decile": float(
            deciles[deciles["population_decile"] == N_DECILES]["share_of_explainable"].mean()
        ),
        "r2_lift_smallest_decile": float(
            deciles[deciles["population_decile"] == 1]["r2_lift_over_size"].mean()
        ),
        "r2_lift_largest_decile": float(
            deciles[deciles["population_decile"] == N_DECILES]["r2_lift_over_size"].mean()
        ),
        "r2_size_smallest_decile": float(
            deciles[deciles["population_decile"] == 1]["r2_size"].mean()
        ),
    }


def assemble_stats(
    scores: pd.DataFrame,
    deciles: pd.DataFrame,
    placebos: pd.DataFrame,
    sizes: pd.DataFrame,
) -> dict[str, object]:
    """Build the stats artifact from the four score frames.

    Split out of `main` so the artifact can be rebuilt from the committed CSVs
    without re-fitting anything. Every statistic in it is a pure function of
    those frames, so a summary-only change does not need a 25-minute sweep to
    land -- and, more to the point, the rebuild goes through this exact function
    rather than a parallel copy of it that could drift.

    Args:
        scores: Per-target, per-model scores.
        deciles: Per-target, per-decile error breakdown.
        placebos: Per-target, per-pillar noise floor.
        sizes: Per-target, per-training-size lift.

    Returns:
        JSON-serializable stats dictionary.
    """
    stats = summarize(scores, deciles, placebos)
    stats["by_training_size"] = sizes.to_dict(orient="records")
    return stats


def main() -> None:
    """Run the external-target sweep and write its three artifacts."""
    configure_logging()
    panel, pillar_columns, blocks = load_panel()

    all_scores: list[pd.DataFrame] = []
    all_deciles: list[pd.DataFrame] = []
    all_sizes: list[pd.DataFrame] = []
    all_placebos: list[pd.DataFrame] = []

    for target in EXTERNAL_TARGETS:
        logger.info("scoring %s (%s)", target.column, target.label)
        scores, predictions = score_target(
            panel, pillar_columns, blocks, target.column, target.label
        )
        for _, row in scores.iterrows():
            logger.info(
                "  %-12s R2=%+.4f  rmse=%.4f  lift over size=%+.4f  ablated=%+.4f",
                row["model"],
                row["r2_out_of_state"],
                row["rmse"],
                row["lift_over_size"],
                row["lift_over_size_ablated"],
            )
        all_scores.append(scores)
        all_deciles.append(score_by_decile(panel, target.column, predictions))
        all_sizes.append(score_by_training_size(panel, pillar_columns, target.column))

        # Keyed off the plain drop models only. The geo family withholds the
        # same pillar letters, so matching on `withheld_pillars` alone would
        # overwrite each pillar's measured contribution with its
        # geography-controlled twin and score the placebo against the wrong bar.
        contributions = {
            row.withheld_pillars: float(row.contribution_ablated)
            for row in scores.itertuples()
            if row.model in (f"size_emacro_drop_{pillar}" for pillar in "ABCDEF")
        }
        all_placebos.append(
            score_placebo(panel, pillar_columns, blocks, target.column, contributions)
        )

    scores = pd.concat(all_scores, ignore_index=True)
    deciles = pd.concat(all_deciles, ignore_index=True)
    sizes = pd.concat(all_sizes, ignore_index=True)
    placebos = pd.concat(all_placebos, ignore_index=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    deciles.to_csv(DECILE_PATH, index=False)
    placebos.to_csv(PLACEBO_PATH, index=False)
    sizes.to_csv(TRAINING_SIZE_PATH, index=False)

    stats = assemble_stats(scores, deciles, placebos, sizes)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("wrote %s", SCORES_PATH)
    logger.info("wrote %s", DECILE_PATH)
    logger.info("wrote %s", PLACEBO_PATH)
    logger.info("wrote %s", TRAINING_SIZE_PATH)
    logger.info("wrote %s", STATS_PATH)
    logger.info(
        "mean lift over size across %d targets: %+.4f | RMSE reduction smallest decile %+.3f, "
        "largest %+.3f",
        stats["n_targets"],
        stats["mean_lift_over_size"],
        stats["rmse_reduction_smallest_decile"],
        stats["rmse_reduction_largest_decile"],
    )
    for name, row in sorted(stats["drop_one"].items()):
        logger.info(
            "  %-30s contribution %+.4f raw, %+.4f ablated, positive on %d/%d targets",
            name,
            row["mean_contribution"],
            row["mean_contribution_ablated"],
            row["n_positive_ablated"],
            row["n_targets"],
        )

    geo = stats["geo_control"]
    boot = stats["bootstrap"]["table_clustered"]["by_pillar"]
    logger.info(
        "geography control | matrix lift %+.4f over size, %+.4f over size+lat/lon; "
        "lat/lon alone %+.4f over size",
        geo["matrix_lift_over_size"],
        geo["matrix_lift_over_size_geo"],
        geo["latlong_lift_over_size"],
    )
    for pillar, row in sorted(geo["by_pillar"].items()):
        interval = boot[pillar]["contribution"]
        share = row["share_retained"]
        logger.info(
            "  Source %s  %+.4f [%+.4f, %+.4f]  ->  %+.4f net of lat/lon  (%s retained)",
            pillar,
            interval["point"],
            interval["low"],
            interval["high"],
            row["contribution_geo"],
            "n/a" if share is None else f"{share:.0%}",
        )


if __name__ == "__main__":
    main()
