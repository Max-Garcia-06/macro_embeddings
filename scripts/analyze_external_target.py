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

# Training-set sizes for the row-count sensitivity. 210 is the DMA count, which
# is the comparison the grain question turns on; the rest bracket it.
SUBSAMPLE_SIZES: tuple[int, ...] = (210, 400, 800, 1600, 3000)
N_SUBSAMPLE_REPS: int = 10

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"

SCORES_PATH: Path = OUTPUTS_DIR / "external_target_scores.csv"
DECILE_PATH: Path = OUTPUTS_DIR / "external_target_by_decile.csv"
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
TARGET_RESTATEMENTS: dict[str, tuple[str, ...]] = {
    "median_household_income": ("wage_per_return_thousands",),
    "median_age": ("retirement_destination",),
    "broadband_rate": (),
}


@dataclass(frozen=True)
class ModelSpec:
    """One predictor set to be scored against every target.

    Attributes:
        name: Short identifier used in outputs.
        label: Human-readable description used in reports.
        uses_size: Whether the size features enter the design.
        uses_pillars: Whether the pillar feature blocks enter the design.
    """

    name: str
    label: str
    uses_size: bool
    uses_pillars: bool


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("grand_mean", "intercept only (= fixed effect on an unseen unit)", False, False),
    ModelSpec("size", "county size only", True, False),
    ModelSpec("emacro", "E_macro pillars only", False, True),
    ModelSpec("size_emacro", "size + E_macro pillars", True, True),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    """Join the pillar matrix, the external targets, and county population.

    Returns:
        Tuple of (panel, pillar_columns). `panel` carries `fips_code`,
        `state_fips`, `population`, every size and pillar feature, and one
        column per external target.

    Raises:
        FileNotFoundError: If any pillar parquet or the target cache is absent.
    """
    matrix, blocks = build_matrix()
    pillar_columns = [column for columns in blocks.values() for column in columns]

    targets = fetch_external_targets()
    target_columns = [target.column for target in EXTERNAL_TARGETS]
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
    return panel, pillar_columns


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
    panel: pd.DataFrame, pillar_columns: list[str], column: str, label: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score every model against one external target.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
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
        design = build_design(usable, model, pillar_columns)
        predicted = out_of_fold_predictions(design, y, groups)
        predictions[model.name] = predicted

        if ablate and model.uses_pillars:
            ablated_design = build_design(usable, model, pillar_columns, ablate=ablate)
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
    return scores, predictions


def score_by_decile(
    panel: pd.DataFrame,
    column: str,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Break one target's out-of-fold error down by county population decile.

    Args:
        panel: Joined panel from `load_panel`.
        column: Target column name.
        predictions: Out-of-fold predictions per model, from `score_target`.

    Returns:
        DataFrame with one row per decile carrying each model's RMSE and the
        proportional error reduction from `size` to `size_emacro`.
    """
    usable = panel[panel[column].notna()].reset_index(drop=True)
    y = usable[column].astype(float).to_numpy()
    decile = pd.qcut(usable["population"], N_DECILES, labels=False, duplicates="drop")

    rows: list[dict[str, object]] = []
    for index in sorted(pd.unique(decile.dropna())):
        mask = (decile == index).to_numpy()
        row: dict[str, object] = {
            "target": column,
            "population_decile": int(index) + 1,
            "n": int(mask.sum()),
            "median_population": float(usable.loc[mask, "population"].median()),
        }
        for name, predicted in predictions.items():
            row[f"rmse_{name}"] = float(np.sqrt(np.mean((y[mask] - predicted[mask]) ** 2)))
        size_rmse = float(row["rmse_size"])
        combined_rmse = float(row["rmse_size_emacro"])
        row["rmse_reduction"] = (
            (size_rmse - combined_rmse) / size_rmse if size_rmse > 0 else float("nan")
        )
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
    }


def main() -> None:
    """Run the external-target sweep and write its three artifacts."""
    configure_logging()
    panel, pillar_columns = load_panel()

    all_scores: list[pd.DataFrame] = []
    all_deciles: list[pd.DataFrame] = []
    all_sizes: list[pd.DataFrame] = []

    for target in EXTERNAL_TARGETS:
        logger.info("scoring %s (%s)", target.column, target.label)
        scores, predictions = score_target(panel, pillar_columns, target.column, target.label)
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

    scores = pd.concat(all_scores, ignore_index=True)
    deciles = pd.concat(all_deciles, ignore_index=True)
    sizes = pd.concat(all_sizes, ignore_index=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    deciles.to_csv(DECILE_PATH, index=False)

    stats = summarize(scores, deciles)
    stats["by_training_size"] = sizes.to_dict(orient="records")
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("wrote %s", SCORES_PATH)
    logger.info("wrote %s", DECILE_PATH)
    logger.info("wrote %s", STATS_PATH)
    logger.info(
        "mean lift over size across %d targets: %+.4f | RMSE reduction smallest decile %+.3f, "
        "largest %+.3f",
        stats["n_targets"],
        stats["mean_lift_over_size"],
        stats["rmse_reduction_smallest_decile"],
        stats["rmse_reduction_largest_decile"],
    )


if __name__ == "__main__":
    main()
