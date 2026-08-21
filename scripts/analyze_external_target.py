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

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"

SCORES_PATH: Path = OUTPUTS_DIR / "external_target_scores.csv"
DECILE_PATH: Path = OUTPUTS_DIR / "external_target_by_decile.csv"
PLACEBO_PATH: Path = OUTPUTS_DIR / "external_target_drop_one_placebo.csv"
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
#   - `bachelors_share` and `graduate_share` both restate `low_postsecondary_ed`,
#     Source F's below-threshold postsecondary-attainment flag.
#   - `labor_force_participation` restates `low_employment`, Source F's flag
#     for counties below the labour-force-participation threshold.
#   - `household_ss_income_share` (share of households receiving Social
#     Security income) restates `retirement_destination` on the same reasoning
#     already applied to `median_age`: it is at least as strong an age-
#     structure proxy as median age itself.
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
    "graduate_share": ("low_postsecondary_ed",),
    "labor_force_participation": ("low_employment",),
    "household_ss_income_share": ("retirement_destination",),
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

# Order matters: `score_by_training_size` indexes MODELS[1] and MODELS[3] for the
# size and size+E_macro designs, so the four original specs stay at the front.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("grand_mean", "intercept only (= fixed effect on an unseen unit)", False, False),
    ModelSpec("size", "county size only", True, False),
    ModelSpec("emacro", "E_macro pillars only", False, True),
    ModelSpec("size_emacro", "size + E_macro pillars", True, True),
) + DROP_MODELS


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
        `state_fips`, `population`, every size and pillar feature, and one
        column per external target. `blocks` maps pillar letter to its column
        list, which the drop-one models need.

    Raises:
        FileNotFoundError: If any pillar parquet or the target cache is absent.
    """
    matrix, blocks = build_matrix()
    pillar_columns = [column for columns in blocks.values() for column in columns]

    targets = fetch_external_targets()
    # Each target ships a `_se` companion carrying its ACS standard error, which
    # `score_by_decile` needs to compute the per-stratum noise floor.
    target_columns = [target.column for target in EXTERNAL_TARGETS]
    target_columns += [f"{column}_se" for column in target_columns]
    population = fetch_county_population()[["fips_code", "population"]]

    panel = (
        matrix.merge(targets[["fips_code", *target_columns]], on="fips_code", how="inner")
        .merge(population, on="fips_code", how="inner")
        .reset_index(drop=True)
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
    dropped = scores[scores["reference_model"].astype(bool)]
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


def summarize(scores: pd.DataFrame, deciles: pd.DataFrame) -> dict[str, object]:
    """Assemble the sweep-level summary written alongside the CSVs.

    Args:
        scores: Per-target, per-model scores.
        deciles: Per-target, per-decile error breakdown.

    Returns:
        JSON-serializable summary dictionary.
    """
    combined = scores[scores["model"] == "size_emacro"]
    emacro_only = scores[scores["model"] == "emacro"]
    return {
        "drop_one": drop_one_summary(scores),
        "n_targets": int(scores["target"].nunique()),
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
            }
            for _, row in combined.iterrows()
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

        contributions = {
            row.withheld_pillars: float(row.contribution_ablated)
            for row in scores.itertuples()
            if len(str(row.withheld_pillars)) == 1
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

    stats = summarize(scores, deciles)
    stats["by_training_size"] = sizes.to_dict(orient="records")
    stats["drop_one_noise_floor"] = {
        str(pillar): {
            "n_targets": int(len(frame)),
            "mean_contribution_ablated": float(frame["contribution_ablated"].mean()),
            "mean_placebo": float(frame["placebo_mean"].mean()),
            "max_placebo": float(frame["placebo_max"].max()),
            "n_targets_above_floor": int(
                (frame["contribution_ablated"] > frame["placebo_p95"]).sum()
            ),
        }
        for pillar, frame in placebos.groupby("pillar")
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("wrote %s", SCORES_PATH)
    logger.info("wrote %s", DECILE_PATH)
    logger.info("wrote %s", PLACEBO_PATH)
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


if __name__ == "__main__":
    main()
