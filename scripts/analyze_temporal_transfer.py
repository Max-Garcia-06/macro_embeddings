"""Does E_macro predict how a county *changes*, which is what a fixed effect cannot?

The limits section of the status notebook lists temporal transfer as "not
started" and calls it "the one argument a fixed effect cannot answer". That is
the right description of its importance and the reason it is worth running
first among the untested items.

## Why this is the test the cross-sectional arm cannot be

`analyze_external_target.py` answers the fixed-effect objection by holding out
whole states: a fixed effect has no parameter for a place it has never seen, so
extrapolation to unseen geography is a seam where a static feature layer can
win. That is a real seam and it is also the *weaker* of the two, because the
consuming team sees almost every market almost all the time.

The second seam does not depend on the unit being unseen. **A geographic fixed
effect estimated on history predicts, for a unit it knows well, that unit's own
past level.** It has no parameter for movement. So:

    target      y(late) - y(early), the county's change between two ACS vintages
    lagged      y(early) -- the fixed effect's own prediction, and the control
                that absorbs mean reversion
    size        log population, log AGI, log GDP
    geo         county centroid latitude and longitude
    emacro      all six pillar blocks

**The headline is `lagged + size + emacro` minus `lagged + size`.** It answers:
for a market the consumer already has years of history on, does knowing what
kind of place it is tell you anything about where it is going that its own
history does not?

`lagged` is in every design, including the baselines, and that is not a
formality. Change is mechanically anti-correlated with the level whenever the
level is measured with error, and ACS county estimates in small counties are
measured with a lot of it. Without `y(early)` in the baseline, any feature
correlated with the level would score as a predictor of the change through
regression to the mean alone.

## What this is, and what it is not

**It is a change-prediction test.** The pillar features are a single current
vintage and sit between the two target vintages, so a pillar is not being asked
to forecast from information available before the outcome window opened.

**It is not an out-of-time forecast.** That needs feature vintages predating the
early target, and only Source E has a panel deep enough to supply one
(`source_e_irs_soi_panel.parquet`, TY2018-TY2022). Building the other five at an
earlier vintage is an ingest question, not an analysis one, and it is the
obvious next step rather than something this script quietly approximates.

Reporting a change-prediction result as though it were a forecast would be the
same error the geography control caught in the Source A representation work:
a number that is real, measured correctly, and answering a different question
from the one its label implies.

## The overlap, which biases against the result

ACS 5-year estimates are labelled by their final year and cover the five years
ending there. The two vintages available in the table-based summary file era are
therefore ACS 2021 (2017-2021) and ACS 2024 (2020-2024), which **share two years
of sample**. Change measured across them is attenuated: part of what genuinely
moved is present in both windows and cancels.

That runs against the hypothesis rather than for it. A null result here is
therefore weaker evidence of "no temporal signal" than a positive result is of
"there is one", and the writeup has to say so in that direction and not the
other.

## The noise floor on a difference

Differencing two noisy estimates adds their variances: the sampling noise in
`y(late) - y(early)` is `se_late^2 + se_early^2`, against a change variance that
is much smaller than either level's. Several targets are consequently mostly
noise once differenced, and `noise_share_of_change` is reported per target so
that a low R2 can be read as "unpredictable by anyone" rather than "E_macro
failed". Targets above `MAX_NOISE_SHARE` are flagged and excluded from the
headline mean, with the exclusion listed rather than applied silently.

Outputs: `outputs/temporal_transfer_scores.csv`,
`analysis-output/cross-source/temporal_transfer_stats.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from analyze_external_target import (
    BOOTSTRAP_PERCENTILES,
    GEO_FEATURES,
    N_BOOTSTRAP,
    N_FOLDS,
    N_PLACEBO_REPS,
    RANDOM_SEED,
    TARGET_RESTATEMENTS,
    TARGET_TABLES,
    _draw_target_positions,
    _interval,
    out_of_fold_predictions,
)
from county_population import fetch_county_population
from ingest_external_targets import EXTERNAL_TARGETS, fetch_external_targets
from pillar_matrix import SIZE_FEATURES, build_matrix

# The two ACS vintages the table-based summary file supports at the widest
# separation it allows. Three years apart by label, two years of shared sample.
EARLY_YEAR: int = 2021
LATE_YEAR: int = 2024

# A target whose differenced sampling noise exceeds this share of its change
# variance carries almost no signal to find, and averaging it into the headline
# would understate the pillars against a quantity nothing could predict. Set
# before any target was scored.
MAX_NOISE_SHARE: float = 0.75

# A county needs both vintages and a change that is not identically zero.
MIN_COUNTIES: int = 500

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"
CENTROIDS_PATH: Path = REPO_ROOT / "data" / "county_centroids.parquet"

SCORES_PATH: Path = OUTPUTS_DIR / "temporal_transfer_scores.csv"
STATS_PATH: Path = ANALYSIS_DIR / "temporal_transfer_stats.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalModel:
    """One predictor set scored against every target's change.

    Attributes:
        name: Short identifier used in outputs.
        label: Human-readable description used in reports.
        uses_lagged: Whether the early-vintage level enters the design. True for
            every model here; the field exists so the one exception -- the
            fixed effect's own prediction, a constant -- is explicit rather
            than implied.
        uses_size: Whether the size features enter the design.
        uses_geo: Whether the centroid columns enter the design.
        uses_pillars: Whether the pillar blocks enter the design.
        drop_pillars: Pillar letters whose blocks are withheld.
        reference: Model this one is differenced against to state a
            contribution. Empty for models that state no contribution.
    """

    name: str
    label: str
    uses_lagged: bool
    uses_size: bool
    uses_geo: bool
    uses_pillars: bool
    drop_pillars: tuple[str, ...] = ()
    reference: str = ""


MODELS: tuple[TemporalModel, ...] = (
    TemporalModel(
        "fixed_effect",
        "intercept only (= a geographic fixed effect's prediction of change)",
        False,
        False,
        False,
        False,
    ),
    TemporalModel("lagged", "the county's own early-vintage level", True, False, False, False),
    TemporalModel("lagged_size", "lagged + county size", True, True, False, False),
    TemporalModel("lagged_size_geo", "lagged + size + lat/lon", True, True, True, False),
    TemporalModel(
        "lagged_size_emacro",
        "lagged + size + E_macro pillars",
        True,
        True,
        False,
        True,
    ),
    # No `reference`. `contribution` is defined as reference minus self, which
    # states a withheld block's worth because self is the *reduced* model. Both
    # pillar arms are the fuller one, so their lift is carried in its own
    # column and differenced in the right direction.
    TemporalModel(
        "lagged_size_geo_emacro",
        "lagged + size + lat/lon + E_macro pillars",
        True,
        True,
        True,
        True,
    ),
) + tuple(
    TemporalModel(
        f"lagged_size_emacro_drop_{pillar}",
        f"lagged + size + E_macro pillars, Source {pillar} withheld",
        True,
        True,
        False,
        True,
        drop_pillars=(pillar,),
        reference="lagged_size_emacro",
    )
    for pillar in "ABCDEF"
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def load_panel() -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    """Join the pillar matrix to both ACS vintages.

    Returns:
        Tuple of (panel, pillar_columns, blocks, targets). `panel` carries one
        `{column}_early`, `{column}_late` and matching `_se` pair per target
        that survives both vintages. `targets` lists those columns.

    Raises:
        ValueError: If a panel county has no centroid, or if the two vintages
            do not overlap on enough counties to score.
    """
    matrix, blocks = build_matrix()
    pillar_columns = [column for columns in blocks.values() for column in columns]

    early = fetch_external_targets(year=EARLY_YEAR)
    late = fetch_external_targets(year=LATE_YEAR)
    population = fetch_county_population()[["fips_code", "population"]]
    centroids = pd.read_parquet(CENTROIDS_PATH)[["fips_code", *GEO_FEATURES]]

    # Only targets both vintages actually carry. ACS line numbering moves
    # between vintages, so a target present in one file can be absent from the
    # other; scoring it would compare a column against nothing.
    targets = [
        target.column
        for target in EXTERNAL_TARGETS
        if target.column in early.columns and target.column in late.columns
    ]
    missing = [target.column for target in EXTERNAL_TARGETS if target.column not in targets]
    if missing:
        logger.warning("absent from one vintage, not scored: %s", ", ".join(missing))

    early_columns = [c for t in targets for c in (t, f"{t}_se")]
    panel = (
        matrix.merge(
            early[["fips_code", *early_columns]].rename(
                columns={c: f"{c}_early" for c in early_columns}
            ),
            on="fips_code",
            how="inner",
        )
        .merge(
            late[["fips_code", *early_columns]].rename(
                columns={c: f"{c}_late" for c in early_columns}
            ),
            on="fips_code",
            how="inner",
        )
        .merge(population, on="fips_code", how="inner")
        .merge(centroids, on="fips_code", how="left")
        .reset_index(drop=True)
    )

    missing_centroid = int(panel[list(GEO_FEATURES)].isna().any(axis=1).sum())
    if missing_centroid:
        raise ValueError(f"{missing_centroid} panel counties absent from {CENTROIDS_PATH.name}")
    if len(panel) < MIN_COUNTIES:
        raise ValueError(
            f"only {len(panel)} counties carry both ACS {EARLY_YEAR} and {LATE_YEAR}"
        )

    logger.info(
        "panel: %d counties x %d pillar features, %d targets in both vintages",
        len(panel),
        len(pillar_columns),
        len(targets),
    )
    return panel, pillar_columns, blocks, targets


def build_design(
    usable: pd.DataFrame,
    model: TemporalModel,
    pillar_columns: list[str],
    lagged_column: str,
    ablate: tuple[str, ...] = (),
) -> np.ndarray:
    """Assemble one model's predictor array.

    Args:
        usable: Rows with both vintages present for this target.
        model: The predictor set to assemble.
        pillar_columns: Every pillar feature column name.
        lagged_column: The target's early-vintage column.
        ablate: Pillar columns to drop, for the restatement-free variant and
            for the withheld block.

    Returns:
        Float array of shape (n_counties, n_predictors). The intercept-only
        design is a single constant column, so one pipeline scores every model.
    """
    columns: list[str] = []
    if model.uses_lagged:
        columns.append(lagged_column)
    if model.uses_size:
        columns.extend(SIZE_FEATURES)
    if model.uses_geo:
        columns.extend(GEO_FEATURES)
    if model.uses_pillars:
        columns.extend(column for column in pillar_columns if column not in ablate)
    if not columns:
        return np.ones((len(usable), 1), dtype=float)
    return usable[columns].astype(float).to_numpy()


def noise_share_of_change(usable: pd.DataFrame, column: str, change: np.ndarray) -> float:
    """Share of a target's change variance that is ACS sampling error.

    Differencing two independent estimates adds their variances, against a
    change variance much smaller than either level's -- which is why several
    targets are mostly noise once differenced even though neither vintage is.

    Args:
        usable: Rows with both vintages present.
        column: Target column name, without a vintage suffix.
        change: The differenced target.

    Returns:
        `mean(se_early^2 + se_late^2) / var(change)`, or NaN if the change has
        no variance.
    """
    early_se = usable[f"{column}_se_early"].astype(float).to_numpy()
    late_se = usable[f"{column}_se_late"].astype(float).to_numpy()
    noise = float(np.nanmean(early_se**2 + late_se**2))
    total = float(np.var(change, ddof=1))
    return noise / total if total > 0 else float("nan")


def score_target(
    panel: pd.DataFrame,
    pillar_columns: list[str],
    blocks: dict[str, list[str]],
    column: str,
) -> pd.DataFrame:
    """Score every model against one target's change between the two vintages.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        blocks: Pillar-to-columns mapping, for the drop-one models.
        column: Target column name, without a vintage suffix.

    Returns:
        One row per model, carrying its out-of-fold R2 on the change and, for
        the drop-one and pillar arms, the contribution against its reference.
    """
    early_column, late_column = f"{column}_early", f"{column}_late"
    usable = panel[panel[early_column].notna() & panel[late_column].notna()].reset_index(
        drop=True
    )
    if len(usable) < MIN_COUNTIES or usable["state_fips"].nunique() < N_FOLDS:
        logger.warning("%-30s skipped: %d usable counties", column, len(usable))
        return pd.DataFrame()

    change = (
        usable[late_column].astype(float).to_numpy()
        - usable[early_column].astype(float).to_numpy()
    )
    groups = usable["state_fips"].to_numpy()
    ablate = TARGET_RESTATEMENTS.get(column, ())
    noise_share = noise_share_of_change(usable, column, change)

    rows: list[dict[str, object]] = []
    for model in MODELS:
        withheld = tuple(c for p in model.drop_pillars for c in blocks[p])
        design = build_design(
            usable, model, pillar_columns, early_column, ablate=withheld + tuple(ablate)
        )
        predicted = out_of_fold_predictions(design, change, groups)
        rows.append(
            {
                "target": column,
                "model": model.name,
                "model_label": model.label,
                "n": len(change),
                "n_predictors": design.shape[1],
                "r2_change": float(r2_score(change, predicted)),
                "withheld_pillars": ";".join(model.drop_pillars),
                "reference_model": model.reference,
                "noise_share_of_change": noise_share,
                "mostly_noise": bool(noise_share > MAX_NOISE_SHARE),
                "change_sd": float(np.std(change, ddof=1)),
            }
        )

    scores = pd.DataFrame(rows)
    by_model = scores.set_index("model")["r2_change"]
    scores["contribution"] = [
        by_model[row.reference_model] - row.r2_change if row.reference_model else np.nan
        for row in scores.itertuples()
    ]
    # The quantity the headline turns on, carried on every row so the CSV can
    # be read without re-deriving it: what the pillars add over lagged + size.
    scores["lift_over_lagged_size"] = scores["r2_change"] - by_model["lagged_size"]
    scores["lift_over_lagged"] = scores["r2_change"] - by_model["lagged"]
    # The geography-controlled twin of the headline, differenced against the
    # baseline that already holds lat/lon so the two arms differ by the pillars
    # and nothing else.
    scores["lift_over_lagged_size_geo"] = scores["r2_change"] - by_model["lagged_size_geo"]
    return scores


def score_placebo(
    panel: pd.DataFrame,
    pillar_columns: list[str],
    blocks: dict[str, list[str]],
    column: str,
) -> pd.DataFrame:
    """Measure what a block carrying no county alignment appears to add to the change.

    Same construction as `analyze_external_target.score_placebo`: the block
    under test is permuted inside the design, so width and marginal
    distributions survive and only the county alignment is destroyed.

    Args:
        panel: Joined panel from `load_panel`.
        pillar_columns: Every pillar feature column name.
        blocks: Pillar-to-columns mapping.
        column: Target column name, without a vintage suffix.

    Returns:
        DataFrame with one row per pillar.
    """
    early_column, late_column = f"{column}_early", f"{column}_late"
    usable = panel[panel[early_column].notna() & panel[late_column].notna()].reset_index(
        drop=True
    )
    if len(usable) < MIN_COUNTIES or usable["state_fips"].nunique() < N_FOLDS:
        return pd.DataFrame()

    change = (
        usable[late_column].astype(float).to_numpy()
        - usable[early_column].astype(float).to_numpy()
    )
    groups = usable["state_fips"].to_numpy()
    ablate = set(TARGET_RESTATEMENTS.get(column, ()))

    rows: list[dict[str, object]] = []
    for pillar, block_columns in blocks.items():
        kept = [c for c in pillar_columns if c not in block_columns and c not in ablate]
        shuffled = [c for c in block_columns if c not in ablate]
        reduced = np.hstack(
            [
                usable[[early_column, *SIZE_FEATURES]].astype(float).to_numpy(),
                usable[kept].astype(float).to_numpy(),
            ]
        )
        r2_reduced = float(r2_score(change, out_of_fold_predictions(reduced, change, groups)))

        rng = np.random.default_rng(RANDOM_SEED)
        block_values = usable[shuffled].astype(float).to_numpy()
        placebo = np.empty(N_PLACEBO_REPS)
        for rep in range(N_PLACEBO_REPS):
            design = np.hstack([reduced, block_values[rng.permutation(len(usable))]])
            placebo[rep] = (
                float(r2_score(change, out_of_fold_predictions(design, change, groups)))
                - r2_reduced
            )
        rows.append(
            {
                "target": column,
                "pillar": pillar,
                "n_reps": N_PLACEBO_REPS,
                "placebo_mean": float(placebo.mean()),
                "placebo_p95": float(np.percentile(placebo, 95)),
                "placebo_max": float(placebo.max()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_headline(scores: pd.DataFrame, targets: list[str]) -> dict[str, object]:
    """Interval the headline lift and every pillar's contribution, paired across targets.

    Same resample as `analyze_external_target.bootstrap_drop_one`: the unit is
    the target, two schemes are reported, and every statistic is drawn on one
    shared selection per replicate so differences are paired.

    Args:
        scores: Per-target, per-model scores.
        targets: Basket targets to resample over.

    Returns:
        Mapping with the resample's parameters and, per scheme, an interval on
        the headline lift, on the geography-controlled lift, and on each
        pillar's drop-one contribution.
    """
    position = {target: index for index, target in enumerate(targets)}
    statistics = {
        "lift_over_lagged_size": ("lagged_size_emacro", "lift_over_lagged_size"),
        "lift_over_lagged_size_geo": ("lagged_size_geo_emacro", "lift_over_lagged_size_geo"),
        **{
            f"drop_{pillar}": (f"lagged_size_emacro_drop_{pillar}", "contribution")
            for pillar in "ABCDEF"
        },
    }

    matrix = np.full((len(statistics), len(targets)), np.nan)
    for index, (model, field) in enumerate(statistics.values()):
        frame = scores[scores["model"] == model]
        for row in frame.itertuples():
            if row.target in position:
                matrix[index, position[row.target]] = getattr(row, field)
    if np.isnan(matrix).any():
        raise ValueError("every model must score every basket target before pairing")

    observed = matrix.mean(axis=1)
    out: dict[str, object] = {
        "n_replicates": N_BOOTSTRAP,
        "percentiles": list(BOOTSTRAP_PERCENTILES),
        "n_targets": len(targets),
        "n_tables": len({TARGET_TABLES[target] for target in targets}),
    }
    for scheme, cluster in (("naive", False), ("table_clustered", True)):
        rng = np.random.default_rng(RANDOM_SEED)
        drawn = np.empty((N_BOOTSTRAP, len(statistics)))
        for replicate in range(N_BOOTSTRAP):
            selection = _draw_target_positions(rng, targets, cluster)
            drawn[replicate] = matrix[:, selection].mean(axis=1)
        out[scheme] = {
            name: _interval(drawn[:, index], observed[index])
            for index, name in enumerate(statistics)
        }
    return out


def summarize(
    scores: pd.DataFrame, placebos: pd.DataFrame, targets: list[str]
) -> dict[str, object]:
    """Assemble the sweep-level summary written alongside the CSV.

    Args:
        scores: Per-target, per-model scores.
        placebos: Per-target, per-pillar noise floor.
        targets: Every target scored.

    Returns:
        JSON-serializable summary dictionary.
    """
    flagged = sorted(
        scores[scores["mostly_noise"]]["target"].unique().tolist()
    )
    basket = [target for target in targets if target not in set(flagged)]
    scored = scores[scores["target"].isin(basket)]

    def mean_of(model: str, field: str) -> float:
        return float(scored[scored["model"] == model][field].mean())

    floor = (
        placebos[placebos["target"].isin(basket)]
        .groupby("pillar")["placebo_max"]
        .max()
        .to_dict()
        if len(placebos)
        else {}
    )

    return {
        "early_vintage": f"ACS {EARLY_YEAR} 5-year",
        "late_vintage": f"ACS {LATE_YEAR} 5-year",
        "overlap_years": 5 - (LATE_YEAR - EARLY_YEAR),
        "n_folds": N_FOLDS,
        "fold_strategy": "GroupKFold on state_fips (spatially blocked)",
        "random_seed": RANDOM_SEED,
        "n_targets_scored": len(targets),
        "n_targets_in_basket": len(basket),
        "max_noise_share": MAX_NOISE_SHARE,
        "excluded_mostly_noise": flagged,
        "basket": basket,
        # The bar: a fixed effect predicts no change, so its R2 on the change is
        # zero by construction and `lagged` is what it becomes once the level is
        # allowed a coefficient. Both are reported so neither has to be assumed.
        "mean_r2_lagged": mean_of("lagged", "r2_change"),
        "mean_r2_lagged_size": mean_of("lagged_size", "r2_change"),
        "mean_r2_lagged_size_emacro": mean_of("lagged_size_emacro", "r2_change"),
        "mean_lift_over_lagged_size": mean_of("lagged_size_emacro", "lift_over_lagged_size"),
        "mean_lift_over_lagged_size_geo": mean_of(
            "lagged_size_geo_emacro", "lift_over_lagged_size_geo"
        ),
        "n_targets_positive": int(
            (
                scored[scored["model"] == "lagged_size_emacro"]["lift_over_lagged_size"] > 0
            ).sum()
        ),
        "drop_one": {
            pillar: {
                "mean_contribution": mean_of(f"lagged_size_emacro_drop_{pillar}", "contribution"),
                "n_positive": int(
                    (
                        scored[scored["model"] == f"lagged_size_emacro_drop_{pillar}"][
                            "contribution"
                        ]
                        > 0
                    ).sum()
                ),
                "max_placebo": float(floor.get(pillar, float("nan"))),
            }
            for pillar in "ABCDEF"
        },
        "bootstrap": bootstrap_headline(scores, basket),
        "by_target": {
            row.target: {
                "r2_lagged_size": float(
                    scored[
                        (scored["target"] == row.target) & (scored["model"] == "lagged_size")
                    ]["r2_change"].iloc[0]
                ),
                "r2_lagged_size_emacro": float(row.r2_change),
                "lift": float(row.lift_over_lagged_size),
                "noise_share_of_change": float(row.noise_share_of_change),
            }
            for row in scored[scored["model"] == "lagged_size_emacro"].itertuples()
        },
    }


def main() -> None:
    """Run the temporal-transfer sweep and write its two artifacts."""
    configure_logging()
    panel, pillar_columns, blocks, targets = load_panel()

    all_scores: list[pd.DataFrame] = []
    all_placebos: list[pd.DataFrame] = []
    for column in targets:
        logger.info("scoring change in %s", column)
        scores = score_target(panel, pillar_columns, blocks, column)
        if scores.empty:
            continue
        all_scores.append(scores)
        all_placebos.append(score_placebo(panel, pillar_columns, blocks, column))
        headline = scores[scores["model"] == "lagged_size_emacro"].iloc[0]
        logger.info(
            "  R2 on change %+.4f, lift over lagged+size %+.4f, %.0f%% of change variance is noise",
            headline["r2_change"],
            headline["lift_over_lagged_size"],
            100 * headline["noise_share_of_change"],
        )

    scores = pd.concat(all_scores, ignore_index=True)
    placebos = (
        pd.concat(all_placebos, ignore_index=True) if all_placebos else pd.DataFrame()
    )
    scored_targets = sorted(scores["target"].unique())

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False)
    stats = summarize(scores, placebos, scored_targets)
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("wrote %s", SCORES_PATH)
    logger.info("wrote %s", STATS_PATH)
    logger.info(
        "%d targets scored, %d in the basket after dropping %d as mostly noise",
        stats["n_targets_scored"],
        stats["n_targets_in_basket"],
        len(stats["excluded_mostly_noise"]),
    )
    interval = stats["bootstrap"]["table_clustered"]["lift_over_lagged_size"]
    logger.info(
        "E_macro on the change: R2 %+.4f against lagged+size %+.4f | lift %+.4f "
        "[%+.4f, %+.4f], positive on %d/%d targets",
        stats["mean_r2_lagged_size_emacro"],
        stats["mean_r2_lagged_size"],
        interval["point"],
        interval["low"],
        interval["high"],
        stats["n_targets_positive"],
        stats["n_targets_in_basket"],
    )
    logger.info(
        "net of lat/lon: %+.4f", stats["mean_lift_over_lagged_size_geo"]
    )
    for pillar, row in sorted(stats["drop_one"].items()):
        logger.info(
            "  Source %s  %+.4f  positive on %d targets, shuffled max %+.4f",
            pillar,
            row["mean_contribution"],
            row["n_positive"],
            row["max_placebo"],
        )


if __name__ == "__main__":
    main()
