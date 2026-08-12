"""Measure how much of each pillar feature is explained by county size alone.

The pillar-pair sweep (`analyze_pillar_pair_crossvalidation.py`) asks
whether two pillars agree with each other. This script asks a different and,
under a rate-shaped downstream target, more decisive question: **is this feature
anything other than county size?**

Motivation is in `docs/downstream_target.md`. Under a count-shaped
target (total revenue, subscriber counts) size is a legitimate feature and a
high correlation here is harmless. Under a rate-shaped target (revenue per
request) the denominator already normalizes for volume, so a feature that tracks
size closely contributes little the downstream model does not already hold.

Size proxy is `log10(population)` from the Census Population Estimates Program
(`county_population.py`), matching the partial-correlation control used in the
sweep. It replaced `log10(num_returns)` on 2026-08-04: the tax-return count is a
Source E column, so Source E's own independence score used to be partly
self-referential (`docs/PROJECT_GOAL.md`, next-work item 2). The two proxies
correlate at r = 0.998 in logs, so the swap removes the self-reference without
moving the tiering; `r_with_log_returns` is retained per feature as evidence of
that rather than as a claim to be taken on trust.

Output: `outputs/feature_size_dependence.csv`, one row per feature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pillar_matrix import SOURCE_A_DIAGNOSTIC_COLUMNS, build_matrix

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "feature_size_dependence.csv"

# Minimum non-null pairs before a correlation is reported. BLS suppression
# leaves some sector LQ columns under 1,100 counties; below this floor the
# surviving counties are a size-biased subsample and the correlation with size
# is circular. Matches MIN_SECTOR_DISCLOSED_COUNT in the B-C correlation script.
MIN_PAIRED_COUNT: int = 100

# |r| with county size at or above this is treated as "size in disguise" for a
# rate-shaped target. Not a statistical threshold -- a reporting cutoff chosen
# so the flagged set matches the features the sweep already showed collapsing
# under the partial-correlation control.
SIZE_PROXY_THRESHOLD: float = 0.30

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_feature_panel() -> tuple[pd.DataFrame, dict[str, str]]:
    """Join every pillar's candidate features onto one per-county frame.

    The feature list comes from `pillar_matrix.build_matrix` rather than from a
    per-pillar list maintained here. That matters because this scan's whole
    purpose is to tier the columns a downstream model receives, and a hand-kept
    list drifts: it missed Source D's ten commodity shares and half of Source
    C's block for as long as those columns had been shipping. Sourcing from the
    matrix makes the scan cover every shipping column by construction, and makes
    `SIZE_COLUMNS` -- already held out of every block -- the single definition of
    what counts as a scale measure rather than a feature.

    Source A's two diagnostics are merged back in on top. `pillar_matrix` keeps
    them out of the matrix entirely, but `docs/source_a_feature_schema.md`
    publishes a size tier for both, and a diagnostic with no measured size
    loading is not much of a diagnostic.

    Returns:
        Tuple of (panel indexed by row with a `log_size` column, mapping of
        feature name to its pillar letter).
    """
    matrix, blocks = build_matrix()

    source_a = pd.read_parquet(DATA_DIR / "source_a_text_features.parquet")
    diagnostic_columns = [
        col for col in SOURCE_A_DIAGNOSTIC_COLUMNS if col in source_a.columns
    ]
    source_e = pd.read_parquet(DATA_DIR / "source_e_irs_soi.parquet")

    panel = matrix.merge(
        source_a[["fips_code", *diagnostic_columns]], on="fips_code", how="left"
    ).merge(source_e[["fips_code", "num_returns"]], on="fips_code", how="left")

    # `log_population` arrives inside the matrix, from county_population via
    # `SIZE_FEATURES`. The superseded tax-return proxy is rebuilt alongside it so
    # every row can report both -- `num_returns` itself is a `SIZE_COLUMNS`
    # member and therefore absent from the matrix by design.
    panel["log_size"] = panel["log_population"]
    panel["log_returns"] = np.log10(panel["num_returns"].clip(lower=1))

    pillar_of: dict[str, str] = {
        column: pillar for pillar, columns in blocks.items() for column in columns
    }
    pillar_of.update({column: "A" for column in diagnostic_columns})
    return panel, pillar_of


def compute_size_dependence(
    panel: pd.DataFrame, pillar_of: dict[str, str]
) -> pd.DataFrame:
    """Correlate each feature against the county-size proxy.

    Args:
        panel: Per-county frame carrying every feature, `log_size` (Census
            population) and `log_returns` (the superseded Source E proxy).
        pillar_of: Mapping of feature name to pillar letter.

    Returns:
        One row per feature with columns `feature`, `pillar`, `r_with_log_size`,
        `abs_r`, `r_squared`, `n`, `size_proxy`, and `r_with_log_returns`,
        sorted by `abs_r` descending.
    """
    rows: list[dict[str, object]] = []
    for feature, pillar in pillar_of.items():
        paired = panel[[feature, "log_size", "log_returns"]].dropna()
        if len(paired) < MIN_PAIRED_COUNT:
            logger.warning(
                "Skipping %s: only %d paired counties (floor %d)",
                feature,
                len(paired),
                MIN_PAIRED_COUNT,
            )
            continue
        values = paired[feature].astype(float)
        if values.std() == 0:
            logger.warning("Skipping %s: constant across %d counties", feature, len(paired))
            continue
        r = float(np.corrcoef(values, paired["log_size"])[0, 1])
        r_returns = float(np.corrcoef(values, paired["log_returns"])[0, 1])
        rows.append(
            {
                "feature": feature,
                "pillar": pillar,
                "r_with_log_size": r,
                "abs_r": abs(r),
                "r_squared": r**2,
                "n": len(paired),
                "size_proxy": abs(r) >= SIZE_PROXY_THRESHOLD,
                "r_with_log_returns": r_returns,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the feature-vs-size dependence scan and write the CSV."""
    configure_logging()
    panel, pillar_of = build_feature_panel()
    logger.info("Panel: %d counties x %d candidate features", len(panel), len(pillar_of))

    dependence = compute_size_dependence(panel, pillar_of)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    dependence.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Wrote %d features to %s", len(dependence), OUTPUT_CSV_PATH)

    flagged = dependence[dependence["size_proxy"]]
    logger.info(
        "%d of %d features exceed |r| = %.2f with county size",
        len(flagged),
        len(dependence),
        SIZE_PROXY_THRESHOLD,
    )

    # What the 2026-08-04 proxy swap cost. Reported every run rather than once,
    # so the claim that Census population and the retired tax-return proxy give
    # the same tiering stays checkable instead of becoming folklore.
    would_flag = dependence["r_with_log_returns"].abs() >= SIZE_PROXY_THRESHOLD
    reclassified = dependence[would_flag != dependence["size_proxy"]]
    logger.info(
        "Proxy swap: max |dr| = %.4f, %d feature(s) change tier vs the retired "
        "tax-return proxy",
        (dependence["r_with_log_size"] - dependence["r_with_log_returns"]).abs().max(),
        len(reclassified),
    )
    for row in reclassified.itertuples():
        logger.info(
            "  %s (%s): population r=%+.3f, returns r=%+.3f",
            row.feature,
            row.pillar,
            row.r_with_log_size,
            row.r_with_log_returns,
        )
    for row in flagged.itertuples():
        logger.info(
            "  %s (%s): r=%+.3f, %.0f%% shared variance, n=%d",
            row.feature,
            row.pillar,
            row.r_with_log_size,
            row.r_squared * 100,
            row.n,
        )


if __name__ == "__main__":
    main()
