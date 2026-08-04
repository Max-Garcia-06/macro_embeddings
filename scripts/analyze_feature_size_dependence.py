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

Size proxy is `log10(num_returns)` from Source E, matching the partial-correlation
control used in the sweep. That proxy is itself a Source E column, so Source E's
own independence score is partly self-referential -- swapping in Census
population is a documented prerequisite before treating this table as final
(`docs/PROJECT_GOAL.md`, next-work item 2).

Output: `outputs/feature_size_dependence.csv`, one row per feature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from extract_source_a_features import VARIANT_COLUMNS
from extract_source_a_section_features import section_feature_columns

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

    Returns:
        Tuple of (panel indexed by row with a `log_size` column, mapping of
        feature name to its pillar letter).
    """
    source_a = pd.read_parquet(DATA_DIR / "source_a_text_features.parquet")
    source_b = pd.read_parquet(DATA_DIR / "source_b_qcew.parquet")
    source_c = pd.read_parquet(DATA_DIR / "source_c_fred.parquet")
    source_d = pd.read_parquet(DATA_DIR / "source_d_faf.parquet")
    source_e = pd.read_parquet(DATA_DIR / "source_e_irs_soi.parquet")
    source_f = pd.read_parquet(DATA_DIR / "source_f_usda_typology.parquet")

    # Source A ships the 29 typed columns, not the `content_length` scalar
    # (source-a-findings.md 14.5). The scalar stays in the scan as the incumbent
    # this table was originally built to judge, but the columns that actually
    # reach a downstream model are the typed ones, and a rate-shaped target
    # cares about their size loading rather than the retired scalar's.
    a_columns = ["content_length", *VARIANT_COLUMNS["extracted_full"], *section_feature_columns()]
    a_columns = [col for col in dict.fromkeys(a_columns) if col in source_a.columns]

    lq_columns = [col for col in source_b.columns if col.startswith("lq_emp_")]
    c_columns = ["gdp_velocity_pct", "unemployment_velocity"]
    d_columns = ["out_partner_hhi", "in_partner_hhi"]
    f_columns = [
        "metro_2023",
        "low_postsecondary_ed",
        "low_employment",
        "population_loss",
        "housing_stress",
        "retirement_destination",
        "persistent_poverty",
    ]

    panel = (
        source_e[["fips_code", "num_returns", "capital_to_wage_ratio"]]
        .merge(
            source_d[["fips_code", "total_outbound_tons", "total_inbound_tons", *d_columns]],
            on="fips_code",
            how="outer",
        )
        .merge(source_c[["fips_code", *c_columns]], on="fips_code", how="outer")
        .merge(source_f[["fips_code", *f_columns]], on="fips_code", how="outer")
        .merge(source_a[["fips_code", *a_columns]], on="fips_code", how="outer")
        .merge(source_b[["fips_code", *lq_columns]], on="fips_code", how="outer")
    )

    # log10 on both size and tonnage: both are heavy-tailed county counts, and
    # the sweep's partial-correlation control uses the same transform.
    panel["log_size"] = np.log10(panel["num_returns"].clip(lower=1))
    panel["log_tons"] = np.log10(
        (panel["total_outbound_tons"] + panel["total_inbound_tons"]).clip(lower=1)
    )

    pillar_of: dict[str, str] = {col: "A" for col in a_columns}
    pillar_of.update({col: "B" for col in lq_columns})
    pillar_of.update({col: "C" for col in c_columns})
    pillar_of.update({col: "D" for col in ["log_tons", *d_columns]})
    pillar_of["capital_to_wage_ratio"] = "E"
    pillar_of.update({col: "F" for col in f_columns})
    return panel, pillar_of


def compute_size_dependence(
    panel: pd.DataFrame, pillar_of: dict[str, str]
) -> pd.DataFrame:
    """Correlate each feature against the county-size proxy.

    Args:
        panel: Per-county frame carrying every feature plus `log_size`.
        pillar_of: Mapping of feature name to pillar letter.

    Returns:
        One row per feature with columns `feature`, `pillar`, `r_with_log_size`,
        `abs_r`, `r_squared`, `n`, `size_proxy`, sorted by `abs_r` descending.
    """
    rows: list[dict[str, object]] = []
    for feature, pillar in pillar_of.items():
        paired = panel[[feature, "log_size"]].dropna()
        if len(paired) < MIN_PAIRED_COUNT:
            logger.warning(
                "Skipping %s: only %d paired counties (floor %d)",
                feature,
                len(paired),
                MIN_PAIRED_COUNT,
            )
            continue
        r = float(
            np.corrcoef(paired[feature].astype(float), paired["log_size"])[0, 1]
        )
        rows.append(
            {
                "feature": feature,
                "pillar": pillar,
                "r_with_log_size": r,
                "abs_r": abs(r),
                "r_squared": r**2,
                "n": len(paired),
                "size_proxy": abs(r) >= SIZE_PROXY_THRESHOLD,
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
