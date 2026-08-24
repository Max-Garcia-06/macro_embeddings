"""Published numbers that later tasks must not move.

Every value here is copied from a committed artifact, not recomputed. If a
change to the encoder or the harness moves one of these, that is a finding to
investigate, not a number to update.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# From analysis-output/source-a/source_a_representation_marginal_stats.json,
# the typed arm's per-target contribution on the original 5-target external
# basket -- the arm that reproduces the published -0.0000 from a separate
# harness. The basket legitimately grew to 42 targets in a later task (Task 3),
# which moves the MEAN over the whole basket for a reason that has nothing to
# do with drift, so a mean-based lock stopped being able to tell "the basket
# grew" apart from "the typed arm's scoring changed". Each of these five
# targets is still scored independently of the basket -- the reduced model
# (size + other five pillars) does not depend on what else is in the basket --
# so their individual contributions are the quantity that actually detects
# drift and survives the expansion. Confirmed unchanged against
# TARGET_RESTATEMENTS for all five (broadband_rate, median_home_value and
# mean_commute_minutes ablate nothing; median_household_income still ablates
# wage_per_return_thousands; median_age still ablates retirement_destination).
TYPED_MARGINAL_BY_TARGET = {
    "broadband_rate": -0.0025629520333230182,
    "median_household_income": -0.0031565484115253506,
    "median_age": 0.0069603527656034725,
    "median_home_value": -0.0025107274757893983,
    "mean_commute_minutes": 0.0010412017577416943,
}

# From outputs/source_a_tiered_embedding.csv, mean pooled lift across 28 targets.
# Full precision deliberately: at 6 decimal places the rounding error is within
# 1.2x of the tolerance below, which makes the assertion flaky rather than tight.
POOLED_LIFT_EXPECTED = {
    "typed_sections": 0.003071681526,
    "uniform": 0.003217592569,
    "uniform_l2": 0.003513738119,
    "lead_only": 0.001685562922,
}


@pytest.mark.parametrize(("target", "expected"), TYPED_MARGINAL_BY_TARGET.items())
def test_typed_marginal_by_target_unchanged(target: str, expected: float) -> None:
    path = REPO_ROOT / "analysis-output" / "source-a" / "source_a_representation_marginal_stats.json"
    stats = json.loads(path.read_text())
    actual = stats["by_representation"]["typed"]["by_target"][target]
    assert actual == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("representation", "expected"), POOLED_LIFT_EXPECTED.items())
def test_pooled_lift_unchanged(
    tiered_embedding_results, representation: str, expected: float
) -> None:
    subset = tiered_embedding_results[
        tiered_embedding_results["representation"] == representation
    ]
    assert len(subset) == 28, f"{representation} should score 28 targets"
    assert subset["lift"].mean() == pytest.approx(expected, abs=1e-9)
