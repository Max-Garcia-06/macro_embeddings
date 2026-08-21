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
# the arm that reproduces the published -0.0000 from a separate harness.
TYPED_MARGINAL_MEAN = -4.573467945852005e-05

# From outputs/source_a_tiered_embedding.csv, mean pooled lift across 28 targets.
# Full precision deliberately: at 6 decimal places the rounding error is within
# 1.2x of the tolerance below, which makes the assertion flaky rather than tight.
POOLED_LIFT_EXPECTED = {
    "typed_sections": 0.003071681526,
    "uniform": 0.003217592569,
    "uniform_l2": 0.003513738119,
    "lead_only": 0.001685562922,
}


def test_typed_marginal_mean_unchanged() -> None:
    path = REPO_ROOT / "analysis-output" / "source-a" / "source_a_representation_marginal_stats.json"
    stats = json.loads(path.read_text())
    actual = stats["by_representation"]["typed"]["mean_contribution"]
    assert actual == pytest.approx(TYPED_MARGINAL_MEAN, abs=1e-12)


@pytest.mark.parametrize(("representation", "expected"), POOLED_LIFT_EXPECTED.items())
def test_pooled_lift_unchanged(
    tiered_embedding_results, representation: str, expected: float
) -> None:
    subset = tiered_embedding_results[
        tiered_embedding_results["representation"] == representation
    ]
    assert len(subset) == 28, f"{representation} should score 28 targets"
    assert subset["lift"].mean() == pytest.approx(expected, abs=1e-9)
