"""Shared statistical-testing helpers used across multiple Source EDA scripts.

`analyze_source_a_clusters.mantel_test` already covers the pairwise-matrix
case (embedding similarity vs. geographic/economic distance). This module
covers the simpler flat-vector case: does a per-county scalar signal (e.g.
freight tonnage, demographic distress count) correlate with another per-county
scalar (e.g. GDP velocity) more than chance would produce?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_test_corr(
    x: pd.Series,
    y: pd.Series,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Permutation test for Pearson correlation between two per-county vectors.

    Repeatedly shuffles `y` and recomputes the correlation to build a null
    distribution, then reports the fraction of permuted correlations at least
    as extreme as the observed one. Rows where either value is null are
    dropped before testing.

    Args:
        x: First per-county scalar series.
        y: Second per-county scalar series, same index as `x`.
        n_permutations: Number of shuffles for the null distribution.
        seed: Random seed for the shuffles.

    Returns:
        Tuple of (observed Pearson correlation, two-sided permutation p-value).
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    x_arr = paired["x"].to_numpy()
    y_arr = paired["y"].to_numpy()
    observed_r = float(np.corrcoef(x_arr, y_arr)[0, 1])

    rng = np.random.default_rng(seed)
    permuted_rs = np.empty(n_permutations)
    for i in range(n_permutations):
        permuted_rs[i] = np.corrcoef(x_arr, rng.permutation(y_arr))[0, 1]

    p_value = (np.sum(np.abs(permuted_rs) >= abs(observed_r)) + 1) / (n_permutations + 1)
    return observed_r, float(p_value)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values for a family of p-values.

    Needed whenever a script scans many per-category correlations (e.g. one
    per NAICS sector) and ranks/reports the largest few: each individual
    p-value is valid on its own, but selecting the top few by magnitude out
    of a larger family inflates apparent significance without a correction.

    Args:
        p_values: Raw p-values from independent tests, order corresponds to
            the caller's own list of results.

    Returns:
        Adjusted q-values in the same order as `p_values`.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    q_values = [0.0] * m
    running_min = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running_min = min(running_min, p_values[idx] * m / rank)
        q_values[idx] = running_min
    return q_values
