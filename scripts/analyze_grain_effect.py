"""Separate the two halves of the DMA grain penalty: row count and aggregation.

`external-target-findings.md` §4 measured the first half. Retraining on random
county subsets showed that at n = 210 -- the Nielsen DMA count -- E_macro's lift
over a size baseline collapses, and is negative on the target closest to the
consumer's domain. What that could not show is whether **aggregation itself**
costs anything beyond having fewer rows, because averaging ~15 counties into a
market removes noise as well as signal and could plausibly help.

This script runs three arms against the same five external targets:

| arm | n | what it isolates |
|---|---|---|
| `county_full` | ~3,143 | the shipping configuration |
| `county_subsample` | 208 | row count alone -- same aggregation (none), fewer rows |
| `market_aggregate` | 208 | row count **and** aggregation |

The gap between the last two is the aggregation effect, with row count held
fixed. That is the quantity `docs/plans/dma_regrain.md` Phase 1B was scoped to
produce and this script produces it without the proprietary crosswalk.

## Two things that bias this comparison, both toward aggregation

State them before the numbers, because both make the aggregate arm look better
than a real DMA delivery would:

1. **The aggregated target is cleaner.** Population-weighted averaging of five
   ACS estimates across ~16 counties reduces their sampling error substantially.
   The market arm is therefore predicting a less noisy outcome than the county
   arms are. If it still loses, it loses despite an advantage.
2. **Markets are spatially compact by construction.** `geo_aggregate` clusters
   county centroids, so within-market economic homogeneity is about as high as a
   county grouping of that size can be. A real DMA, drawn on media-market
   boundaries rather than economic ones, should be no more homogeneous.

## What it cannot claim

Groups are k-means clusters of county centroids at Nielsen cardinality, **not
DMAs** -- that delineation is proprietary (`geo_aggregate` module docstring).
They match the real thing in count and character, which is enough to measure the
mechanism, and not enough to quote a DMA number.

62 of the 118 pillar columns are approximated rather than re-derived at group
grain, Source B's 40 location quotients chief among them, because the shipped
QCEW parquet carries no employment counts. The aggregate arm is therefore a
*lower bound* on what a properly re-ingested DMA delivery could reach.

Output: `outputs/grain_effect.csv`,
`analysis-output/cross-source/grain_effect_stats.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from analyze_external_target import (
    MODELS,
    N_FOLDS,
    RANDOM_SEED,
    build_design,
    load_panel,
    out_of_fold_predictions,
)
from geo_aggregate import DEFAULT_N_MARKETS, aggregate_matrix, build_pseudo_markets
from ingest_external_targets import EXTERNAL_TARGETS
from pillar_matrix import build_matrix

N_SUBSAMPLE_REPS: int = 10
CENTROIDS_PATH: Path = Path("data/county_centroids.parquet")

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"
GRAIN_CSV_PATH: Path = OUTPUTS_DIR / "grain_effect.csv"
GRAIN_STATS_PATH: Path = ANALYSIS_DIR / "grain_effect_stats.json"

logger = logging.getLogger(__name__)


def _lift_over_size(
    frame: pd.DataFrame, pillar_columns: list[str], column: str, groups: np.ndarray
) -> float | None:
    """Out-of-fold R2 lift of size+E_macro over size alone.

    Args:
        frame: Rows carrying the target, the size features and the pillars.
        pillar_columns: Pillar feature column names present in `frame`.
        column: Target column name.
        groups: Grouping vector for the spatially blocked folds.

    Returns:
        Lift in R2, or None when the frame has too few groups to split.
    """
    usable = frame[frame[column].notna()].reset_index(drop=True)
    if len(usable) < N_FOLDS or pd.Series(groups).nunique() < N_FOLDS:
        return None
    mask = frame[column].notna().to_numpy()
    y = usable[column].astype(float).to_numpy()
    fold_groups = groups[mask]
    if pd.Series(fold_groups).nunique() < N_FOLDS:
        return None

    baseline = out_of_fold_predictions(
        build_design(usable, MODELS[1], pillar_columns), y, fold_groups
    )
    combined = out_of_fold_predictions(
        build_design(usable, MODELS[3], pillar_columns), y, fold_groups
    )
    return float(r2_score(y, combined) - r2_score(y, baseline))


def build_market_panel(
    panel: pd.DataFrame, pillar_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aggregate the county panel and its targets to pseudo-markets.

    Targets are population-weighted, which is the right aggregate for a rate or
    a per-household median and an approximation for a population median. Each
    market inherits its modal county state so both arms hold out whole states.

    Args:
        panel: County panel from `analyze_external_target.load_panel`.
        pillar_columns: Every pillar feature column name.

    Returns:
        Tuple of (market_panel, provenance).
    """
    matrix, blocks = build_matrix()
    centroids = pd.read_parquet(CENTROIDS_PATH)
    markets = build_pseudo_markets(centroids, n_markets=DEFAULT_N_MARKETS)
    aggregated, provenance = aggregate_matrix(matrix, blocks, markets)

    joined = panel.merge(markets, on="fips_code", how="inner")
    target_columns = [target.column for target in EXTERNAL_TARGETS]

    rows: list[dict[str, object]] = []
    for market_id, group in joined.groupby("market_id"):
        weights = group["population"]
        row: dict[str, object] = {
            "market_id": market_id,
            "state_fips": group["state_fips"].mode().iloc[0],
        }
        for column in target_columns:
            values = group[column]
            mask = values.notna() & weights.notna()
            row[column] = (
                float(np.average(values[mask], weights=weights[mask]))
                if mask.any() and weights[mask].sum() > 0
                else np.nan
            )
        rows.append(row)

    targets = pd.DataFrame(rows)
    market_panel = aggregated.merge(targets, on="market_id", how="inner")
    logger.info("market panel: %d markets x %d pillar features", len(market_panel), len(pillar_columns))
    return market_panel, provenance


def main() -> None:
    """Run the three-arm grain comparison and write its artifacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    panel, pillar_columns = load_panel()
    market_panel, provenance = build_market_panel(panel, pillar_columns)
    n_markets = len(market_panel)

    rows: list[dict[str, object]] = []
    for target in EXTERNAL_TARGETS:
        column = target.column
        logger.info("grain arms for %s", column)

        full = _lift_over_size(panel, pillar_columns, column, panel["state_fips"].to_numpy())
        rows.append(
            {
                "target": column,
                "arm": "county_full",
                "n_units": int(panel[column].notna().sum()),
                "mean_lift_over_size": full,
                "sd_lift_over_size": 0.0,
            }
        )

        aggregate = _lift_over_size(
            market_panel, pillar_columns, column, market_panel["state_fips"].to_numpy()
        )
        rows.append(
            {
                "target": column,
                "arm": "market_aggregate",
                "n_units": int(market_panel[column].notna().sum()),
                "mean_lift_over_size": aggregate,
                "sd_lift_over_size": 0.0,
            }
        )

        lifts: list[float] = []
        for rep in range(N_SUBSAMPLE_REPS):
            sample = panel.sample(n=n_markets, random_state=RANDOM_SEED + rep).reset_index(
                drop=True
            )
            lift = _lift_over_size(
                sample, pillar_columns, column, sample["state_fips"].to_numpy()
            )
            if lift is not None:
                lifts.append(lift)
        rows.append(
            {
                "target": column,
                "arm": "county_subsample",
                "n_units": n_markets,
                "mean_lift_over_size": float(np.mean(lifts)) if lifts else None,
                "sd_lift_over_size": float(np.std(lifts, ddof=1)) if len(lifts) > 1 else 0.0,
            }
        )

        by_arm = {row["arm"]: row["mean_lift_over_size"] for row in rows if row["target"] == column}
        logger.info(
            "  county_full=%+.4f  county_subsample=%+.4f  market_aggregate=%+.4f",
            by_arm["county_full"],
            by_arm["county_subsample"],
            by_arm["market_aggregate"],
        )

    results = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(GRAIN_CSV_PATH, index=False)

    pivot = results.pivot(index="target", columns="arm", values="mean_lift_over_size")
    stats = {
        "n_markets": n_markets,
        "n_subsample_reps": N_SUBSAMPLE_REPS,
        "grouping": "k-means on county centroids at Nielsen DMA cardinality (NOT DMAs)",
        "mean_lift_county_full": float(pivot["county_full"].mean()),
        "mean_lift_county_subsample": float(pivot["county_subsample"].mean()),
        "mean_lift_market_aggregate": float(pivot["market_aggregate"].mean()),
        "aggregation_effect": float(
            (pivot["market_aggregate"] - pivot["county_subsample"]).mean()
        ),
        "row_count_effect": float(
            (pivot["county_subsample"] - pivot["county_full"]).mean()
        ),
        "columns_rederived": sum(1 for v in provenance.values() if v == "re-derived"),
        "columns_approximated": sum(1 for v in provenance.values() if v == "approximated"),
        "columns_weighted": sum(1 for v in provenance.values() if v == "population-weighted"),
        "by_target": pivot.to_dict(orient="index"),
    }
    GRAIN_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info("wrote %s", GRAIN_CSV_PATH)
    logger.info("wrote %s", GRAIN_STATS_PATH)
    logger.info(
        "row-count effect %+.4f | aggregation effect %+.4f (positive = aggregation helps)",
        stats["row_count_effect"],
        stats["aggregation_effect"],
    )


if __name__ == "__main__":
    main()
