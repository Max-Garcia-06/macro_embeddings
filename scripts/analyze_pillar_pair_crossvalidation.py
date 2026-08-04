"""Run the full pillar-to-pillar crossvalidation sweep across all six E_macro sources.

Every crossvalidation before this one routed through Source C: each pillar was
tested against Source C's velocity metrics and nothing else. Those links are
uniformly weak (|r| <= 0.08). The two direct pillar-to-pillar tests that were
run ad hoc -- Source D x F and Source A x F -- produced the two largest effect
sizes in the project (|r| = 0.217 and 0.247), which is the empirical basis for
running the remaining thirteen pairs here.

The sweep tests representative scalar features from each pillar against every
feature of every other pillar, permutation-tests each correlation, and applies
one Benjamini-Hochberg correction across the whole sweep rather than per pair.
Results are reported both per feature pair and collapsed to a best-per-pillar-
pair summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from county_population import load_size_proxy
from stats_utils import benjamini_hochberg, permutation_test_corr

# Matches every other crossvalidation script in this project so effect sizes and
# p-values stay directly comparable across rounds.
RANDOM_SEED: int = 42
N_PERMUTATIONS: int = 499
MIN_PAIRED_OBSERVATIONS: int = 100
FDR_ALPHA: float = 0.05

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
ANALYSIS_DIR: Path = REPO_ROOT / "analysis-output" / "cross-source"

OUTPUT_CSV_PATH: Path = OUTPUTS_DIR / "pillar_pair_crossvalidation.csv"
OUTPUT_STATS_PATH: Path = ANALYSIS_DIR / "pillar_pair_stats.json"

DISTRESS_FLAGS: tuple[str, ...] = (
    "low_postsecondary_ed",
    "low_employment",
    "population_loss",
    "housing_stress",
    "retirement_destination",
    "persistent_poverty",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PillarFeature:
    """One scalar feature drawn from one pillar.

    Attributes:
        pillar: Single-letter pillar identifier, "A" through "F".
        column: Column name in the assembled panel.
        label: Human-readable description used in reports.
    """

    pillar: str
    column: str
    label: str


# One to two representative scalars per pillar. Source B carries both the
# summary scalar the pipeline currently exposes (`dominant_lq`) and the single
# sector LQ that source-b-findings.md identified as its strongest signal, so the
# sweep can distinguish "the pillar is weak" from "the feature is weak".
#
# Source A carries a second feature for the same reason. `content_length` is what
# the pipeline shipped; `sec_n_industry_mentions` counts the industries named in
# a county's Wikipedia economy section, and is the feature family that carried
# Source A's multivariate result (§14 of source-a-findings.md). Including it
# tests whether Source A's weak showing in this sweep is a property of the pillar
# or of the scalar chosen to stand for it.
#
# Worth knowing before reading the result: bivariately, `content_length` is still
# Source A's strongest scalar (mean |r| = 0.132 across this sweep's targets,
# against 0.116 for `n_distinct_proper_nouns` and 0.038 for this column). The
# typed features win in the multivariate harness, where they combine and where
# specific features match specific targets -- one at a time they are weak. That
# is not a contradiction, it is the difference between the two questions, and
# this entry is what makes it visible rather than assumed.
PILLAR_FEATURES: tuple[PillarFeature, ...] = (
    PillarFeature("A", "content_length", "intro-text length"),
    PillarFeature("A", "sec_n_industry_mentions", "industries named in economy section"),
    PillarFeature("B", "dominant_lq", "dominant sector LQ"),
    PillarFeature("B", "lq_emp_53", "Real Estate & Rental & Leasing LQ"),
    PillarFeature("C", "unemployment_velocity", "unemployment velocity"),
    PillarFeature("C", "gdp_velocity_pct", "GDP velocity (normalized)"),
    PillarFeature("D", "log_total_tons", "freight tonnage (log10)"),
    PillarFeature("D", "mean_partner_hhi", "trade partner concentration"),
    PillarFeature("E", "capital_to_wage_ratio", "capital-to-wage ratio"),
    PillarFeature("F", "distress_count", "demographic distress count"),
    PillarFeature("F", "metro_2023", "metro status"),
)


def configure_logging() -> None:
    """Configure root logging format/level for script execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def build_panel() -> pd.DataFrame:
    """Join all six pillars into one county-level panel of sweep features.

    Source A is read from `source_a_text_features.parquet` (the embedding step
    was removed from that pipeline); Source B's `dominant_lq` is derived here
    as the maximum disclosed LQ across the 20 sector columns.

    Returns:
        DataFrame indexed by row with `fips_code` plus every column named in
        PILLAR_FEATURES. Missing values are preserved rather than filled.

    Raises:
        FileNotFoundError: If any source parquet is absent.
    """
    a = pd.read_parquet(DATA_DIR / "source_a_text_features.parquet")
    b = pd.read_parquet(DATA_DIR / "source_b_qcew.parquet")
    c = pd.read_parquet(DATA_DIR / "source_c_fred.parquet")
    d = pd.read_parquet(DATA_DIR / "source_d_faf.parquet")
    e = pd.read_parquet(DATA_DIR / "source_e_irs_soi.parquet")
    f = pd.read_parquet(DATA_DIR / "source_f_usda_typology.parquet")

    lq_columns = [col for col in b.columns if col.startswith("lq_emp_")]
    b = b.assign(dominant_lq=b[lq_columns].max(axis=1))

    d = d.assign(
        log_total_tons=np.log10((d["total_outbound_tons"] + d["total_inbound_tons"]).clip(lower=1)),
        mean_partner_hhi=d[["out_partner_hhi", "in_partner_hhi"]].mean(axis=1),
    )

    f = f.assign(distress_count=f[list(DISTRESS_FLAGS)].astype("float").sum(axis=1))

    panel = (
        f[["fips_code", "county_name", "distress_count", "metro_2023"]]
        .merge(
            a[["fips_code", "content_length", "sec_n_industry_mentions"]],
            on="fips_code",
            how="left",
        )
        .merge(b[["fips_code", "dominant_lq", "lq_emp_53"]], on="fips_code", how="left")
        .merge(
            c[["fips_code", "unemployment_velocity", "gdp_velocity_pct"]],
            on="fips_code",
            how="left",
        )
        .merge(d[["fips_code", "log_total_tons", "mean_partner_hhi"]], on="fips_code", how="left")
        .merge(
            e[["fips_code", "capital_to_wage_ratio"]],
            on="fips_code",
            how="left",
        )
        .merge(load_size_proxy(), on="fips_code", how="left")
    )
    panel["metro_2023"] = panel["metro_2023"].astype("float")
    return panel


def run_sweep(panel: pd.DataFrame) -> pd.DataFrame:
    """Correlate every cross-pillar feature pair, with one FDR correction over the sweep.

    Within-pillar pairs are skipped: correlating two Source C velocities against
    each other says nothing about whether pillars explain one another.

    Args:
        panel: Output of build_panel.

    Returns:
        DataFrame with one row per feature pair, sorted by descending |r|, with
        columns: pillar_pair, feature_x, feature_y, label_x, label_y, r, p, q,
        n, significant.
    """
    records: list[dict[str, object]] = []

    for left, right in combinations(PILLAR_FEATURES, 2):
        if left.pillar == right.pillar:
            continue

        paired = panel[[left.column, right.column]].dropna()
        if len(paired) < MIN_PAIRED_OBSERVATIONS:
            logger.warning(
                "Skipping %s x %s: only %d paired observations",
                left.column,
                right.column,
                len(paired),
            )
            continue

        r, p = permutation_test_corr(
            paired[left.column], paired[right.column], N_PERMUTATIONS, RANDOM_SEED
        )
        records.append(
            {
                "pillar_pair": f"{left.pillar}x{right.pillar}",
                "feature_x": left.column,
                "feature_y": right.column,
                "label_x": f"{left.pillar} {left.label}",
                "label_y": f"{right.pillar} {right.label}",
                "r": r,
                "p": p,
                "n": len(paired),
            }
        )

    results = pd.DataFrame(records)
    results["q"] = benjamini_hochberg(results["p"].tolist())
    results["significant"] = results["q"] < FDR_ALPHA
    results["abs_r"] = results["r"].abs()
    return results.sort_values("abs_r", ascending=False).reset_index(drop=True)


def partial_correlation(x: pd.Series, y: pd.Series, control: pd.Series) -> float:
    """Correlate two variables after linearly removing a shared control variable.

    Both inputs are regressed on the control and the residuals are correlated,
    which answers whether a pillar-pair link survives once county size is
    accounted for -- the obvious confound, since large counties simultaneously
    move more freight, carry longer Wikipedia articles, and are classified metro.

    The control is removed **linearly** (OLS on `[1, control]`). Any size
    confounding that is nonlinear in the control survives into the residuals,
    so a link that looks size-robust here may still be size-driven. Callers
    pass `log_population` rather than raw counts partly for this reason.

    Args:
        x: First variable.
        y: Second variable.
        control: Variable to partial out of both.

    Returns:
        Partial correlation coefficient. **Returns 0.0 if either residual
        series is constant** -- note that 0.0 is also a legitimate result, so
        a caller cannot distinguish a degenerate input from a genuine null
        from the return value alone.
    """
    design = np.column_stack([np.ones(len(control)), control.to_numpy(dtype=float)])
    residuals = []
    for series in (x, y):
        values = series.to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residuals.append(values - design @ coefficients)

    if np.std(residuals[0]) == 0 or np.std(residuals[1]) == 0:
        return 0.0
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def add_size_controlled_correlations(results: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute every sweep correlation controlling for county size.

    Census PEP county population is used as the size proxy (`county_population`),
    covering all 3,144 counties and belonging to no pillar. It replaced Source
    E's `num_returns` on 2026-08-04, because using one pillar's column to control
    correlations involving that same pillar made Source E's independence partly
    self-referential. The two agree at r = 0.998 in logs, so the swap changes the
    provenance of the control rather than its verdicts.

    **`r_size_controlled` carries no significance test.** `run_sweep`
    permutation-tests and BH-corrects the *raw* correlation only; the
    size-controlled value is a point estimate with no `p`, no `q`, and no
    entry in the FDR family. Anything reported off it -- including the
    headline claim that a link "survives" the size control -- is an effect-size
    statement, not a significance claim. Testing it properly means
    permutation-testing each partial correlation and extending the BH family
    to cover them, which would change which links come out significant.

    `size_confounded` is likewise a **heuristic, not a test**: the 0.5 cutoff
    is a convention chosen here, so a pair just either side of it is not
    meaningfully different from its neighbour.

    Reading the output: 14 of 41 pairs currently *flip sign* under the
    control, the largest being Source D freight tonnage against Source F metro
    status (r = 0.495 raw, -0.057 controlled). A sign reversal is a
    suppression / Simpson's-paradox signature, not simply "the effect went
    away" -- the within-size-stratum relationship genuinely runs opposite to
    the pooled one, and is worth inspecting stratified rather than dismissing.

    Args:
        results: Output of run_sweep.
        panel: Output of build_panel, plus a `log_population` column.

    Returns:
        `results` with `r_size_controlled` and `size_confounded` columns added,
        where `size_confounded` marks pairs that lose more than half their
        effect size once size is partialled out.
    """
    partials: list[float] = []
    for row in results.itertuples():
        paired = panel[[row.feature_x, row.feature_y, "log_population"]].dropna()
        partials.append(
            partial_correlation(
                paired[row.feature_x], paired[row.feature_y], paired["log_population"]
            )
        )

    annotated = results.copy()
    annotated["r_size_controlled"] = partials
    annotated["size_confounded"] = annotated["r_size_controlled"].abs() < (
        annotated["abs_r"] * 0.5
    )
    return annotated


def summarize_by_pillar_pair(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature-pair results to the strongest link per pillar pair.

    Args:
        results: Output of run_sweep.

    Returns:
        DataFrame with one row per pillar pair, sorted by descending |r|.
    """
    best = results.loc[results.groupby("pillar_pair")["abs_r"].idxmax()]
    return best.sort_values("abs_r", ascending=False).reset_index(drop=True)


def build_stats(results: pd.DataFrame, by_pair: pd.DataFrame) -> dict[str, object]:
    """Assemble the machine-readable statistics bundle for this sweep.

    Args:
        results: Output of run_sweep.
        by_pair: Output of summarize_by_pillar_pair.

    Returns:
        JSON-serializable dict of sweep-level counts and per-pillar-pair results.
    """
    return {
        "n_permutations": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "fdr_alpha": FDR_ALPHA,
        "n_feature_pair_tests": int(len(results)),
        "n_pillar_pairs_tested": int(by_pair["pillar_pair"].nunique()),
        "n_significant_feature_pairs": int(results["significant"].sum()),
        "n_size_confounded_feature_pairs": int(results["size_confounded"].sum()),
        "best_per_pillar_pair": {
            row.pillar_pair: {
                "r": float(row.r),
                "p": float(row.p),
                "q": float(row.q),
                "n": int(row.n),
                "significant": bool(row.significant),
                "r_size_controlled": float(row.r_size_controlled),
                "size_confounded": bool(row.size_confounded),
                "feature_x": row.label_x,
                "feature_y": row.label_y,
            }
            for row in by_pair.itertuples()
        },
    }


def main() -> None:
    """Run the full 15-pair pillar crossvalidation sweep and write its artifacts."""
    configure_logging()

    panel = build_panel()
    logger.info("Panel assembled: %d counties x %d columns", len(panel), panel.shape[1])

    results = add_size_controlled_correlations(run_sweep(panel), panel)
    by_pair = summarize_by_pillar_pair(results)
    logger.info(
        "%d feature-pair tests across %d pillar pairs; %d significant at FDR q < %.2f",
        len(results),
        by_pair["pillar_pair"].nunique(),
        int(results["significant"].sum()),
        FDR_ALPHA,
    )

    for row in by_pair.itertuples():
        logger.info(
            "  %s: r=%+.4f -> %+.4f size-controlled (q=%.4f, n=%d)  %s <-> %s",
            row.pillar_pair,
            row.r,
            row.r_size_controlled,
            row.q,
            row.n,
            row.label_x,
            row.label_y,
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV_PATH, index=False)
    OUTPUT_STATS_PATH.write_text(json.dumps(build_stats(results, by_pair), indent=2))
    logger.info("Wrote %s and %s", OUTPUT_CSV_PATH, OUTPUT_STATS_PATH)


if __name__ == "__main__":
    main()
