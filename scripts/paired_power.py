"""Power and dependence diagnostics for the paired target-basket tests.

Source A's headline comparisons are paired tests across the cross-pillar target
basket: score every variant on every target, difference them target by target,
and ask whether the differences sit above zero. The Wilcoxon p-value that falls
out of that hides two properties of the basket which change how the number
should be read.

**How much power the test had.** `p = 0.082` on the typed-versus-scalar
comparison is the ordinary output of a real effect measured at roughly half the
sample its effect size requires (Cohen dz = 0.335, power 0.53 at n = 28). It is
not evidence against the effect, and it reads very differently from a test that
was well powered and still failed. `paired_effect` reports dz, achieved power,
and the target counts that would reach 80% and 90%, so every published p-value
carries the sample size it needed alongside the one it had.

**How many independent draws the basket really contains.** Twenty of the 28
targets are QCEW location quotients. Those are compositional — each is a share
measured against a national base — so they are mechanically coupled, and both
the Wilcoxon and the paired t treat them as independent. `cluster_dependence`
estimates the intraclass correlation of the paired differences within target
pillar, converts it to a Kish design effect, and reports the effective n. It
also runs the test again on pillar means, which is the blocked version of the
same comparison and is immune to one large pillar dominating the count.

Neither function changes any point estimate. They exist so that a reader cannot
take a nominal n of 28 at face value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import nct, t, ttest_1samp

# One-sided alpha. The paired comparisons all ask a directional question -- does
# this variant beat the incumbent -- so the power figures are computed against
# the test that question deserves.
ALPHA: float = 0.05

# Upper bound on the target-count search in `_n_for_power`. Any effect needing
# more targets than this is not going to be settled by adding targets to this
# repo, and the caller should read the ceiling as "not reachable here".
MAX_SEARCH_N: int = 2000


@dataclass(frozen=True)
class PairedEffect:
    """Effect size and achieved power for one vector of paired differences.

    Attributes:
        n: Nominal number of paired observations.
        mean: Mean paired difference.
        median: Median paired difference. Far below the mean means the effect is
            carried by a handful of targets rather than spread across them.
        sd: Standard deviation of the differences.
        dz: Cohen dz, the paired standardized effect size (mean / sd).
        power: Probability this test detects `dz` at `n`, one-sided, alpha 0.05.
        n_for_80: Paired observations needed for 80% power at `dz`.
        n_for_90: Paired observations needed for 90% power at `dz`.
    """

    n: int
    mean: float
    median: float
    sd: float
    dz: float
    power: float
    n_for_80: int
    n_for_90: int


@dataclass(frozen=True)
class ClusterDependence:
    """How far the effective sample size falls below the nominal one.

    Attributes:
        n_nominal: Paired observations as counted by the significance test.
        n_clusters: Distinct clusters, here target pillars.
        mean_cluster_size: Kish's average cluster size, which accounts for the
            unequal pillar counts (B contributes 20 targets, E and F one each).
        icc: Intraclass correlation of the differences within cluster, floored
            at zero. Negative estimates mean no detectable dependence.
        design_effect: `1 + (mean_cluster_size - 1) * icc`.
        n_effective: `n_nominal / design_effect`.
        power_at_effective_n: Power recomputed at the effective n, which is the
            honest figure when the clusters are as lopsided as this basket's.
        cluster_mean_p: Two-sided one-sample t-test on the per-cluster mean
            differences. This is the blocked version of the headline test: every
            pillar gets one vote regardless of how many targets it contributed.
    """

    n_nominal: int
    n_clusters: int
    mean_cluster_size: float
    icc: float
    design_effect: float
    n_effective: float
    power_at_effective_n: float
    cluster_mean_p: float


def _power_at(dz: float, n: float, alpha: float = ALPHA) -> float:
    """Power of a one-sided paired t-test.

    Args:
        dz: Cohen dz, the standardized paired effect size.
        n: Number of paired observations. May be fractional, for effective n.
        alpha: One-sided significance level.

    Returns:
        Probability of rejecting the null at `alpha` when `dz` is the true
        effect. Zero when `n` is too small to have any degrees of freedom.
    """
    if n <= 1 or not np.isfinite(dz):
        return 0.0
    df = n - 1
    critical = t.ppf(1 - alpha, df)
    return float(1 - nct.cdf(critical, df, dz * np.sqrt(n)))


def _n_for_power(dz: float, target_power: float, alpha: float = ALPHA) -> int:
    """Smallest paired sample reaching `target_power` at effect size `dz`.

    Args:
        dz: Cohen dz.
        target_power: Power to reach, e.g. 0.8.
        alpha: One-sided significance level.

    Returns:
        Required number of paired observations, or `MAX_SEARCH_N` when the
        effect is too small to reach that power within the search bound.
    """
    if dz <= 0 or not np.isfinite(dz):
        return MAX_SEARCH_N
    for n in range(3, MAX_SEARCH_N + 1):
        if _power_at(dz, n, alpha) >= target_power:
            return n
    return MAX_SEARCH_N


def paired_effect(differences: pd.Series | np.ndarray) -> PairedEffect:
    """Effect size and achieved power for a vector of paired differences.

    Args:
        differences: One paired difference per target. Variant minus incumbent
            for a head-to-head, or the variant's own lift for a versus-zero
            test.

    Returns:
        Populated `PairedEffect`. A zero-variance input (the incumbent compared
        against itself) yields dz of zero and power of zero rather than raising.
    """
    values = np.asarray(differences, dtype="float64")
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    dz = float(values.mean() / sd) if sd > 0 else 0.0
    return PairedEffect(
        n=int(len(values)),
        mean=float(values.mean()),
        median=float(np.median(values)),
        sd=sd,
        dz=dz,
        power=_power_at(dz, len(values)),
        n_for_80=_n_for_power(dz, 0.80),
        n_for_90=_n_for_power(dz, 0.90),
    )


def _intraclass_correlation(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """One-way random-effects ICC and Kish's average cluster size.

    Args:
        values: Paired differences.
        clusters: Cluster label per observation, same length as `values`.

    Returns:
        Tuple of (ICC floored at zero, average cluster size). The average size
        is Kish's `m0`, which corrects for unequal cluster sizes; using the
        plain mean would overstate dependence when one cluster dominates.
    """
    labels, sizes = np.unique(clusters, return_counts=True)
    n_total, n_clusters = len(values), len(labels)
    if n_clusters < 2 or n_total <= n_clusters:
        return 0.0, float(n_total)

    grand_mean = values.mean()
    cluster_means = np.array([values[clusters == label].mean() for label in labels])
    between = float((sizes * (cluster_means - grand_mean) ** 2).sum() / (n_clusters - 1))
    within = float(
        sum(((values[clusters == label] - mean) ** 2).sum() for label, mean in zip(labels, cluster_means))
        / (n_total - n_clusters)
    )

    average_size = float((n_total - (sizes**2).sum() / n_total) / (n_clusters - 1))
    if within <= 0 or average_size <= 1:
        return 0.0, average_size
    icc = (between - within) / (between + (average_size - 1) * within)
    return max(0.0, float(icc)), average_size


def cluster_dependence(
    differences: pd.Series | np.ndarray, clusters: pd.Series | np.ndarray
) -> ClusterDependence:
    """Effective sample size once within-cluster dependence is accounted for.

    Args:
        differences: One paired difference per target.
        clusters: Cluster label per target. Target pillar, in this repo.

    Returns:
        Populated `ClusterDependence`. When the ICC estimate is non-positive the
        design effect is 1 and effective n equals nominal n.
    """
    values = np.asarray(differences, dtype="float64")
    labels = np.asarray(clusters)
    icc, average_size = _intraclass_correlation(values, labels)
    design_effect = 1.0 + (average_size - 1.0) * icc
    n_effective = len(values) / design_effect if design_effect > 0 else float(len(values))

    unique = np.unique(labels)
    cluster_means = np.array([values[labels == label].mean() for label in unique])
    cluster_p = (
        float(ttest_1samp(cluster_means, 0.0).pvalue) if len(unique) > 1 else float("nan")
    )

    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    dz = float(values.mean() / sd) if sd > 0 else 0.0
    return ClusterDependence(
        n_nominal=int(len(values)),
        n_clusters=int(len(unique)),
        mean_cluster_size=average_size,
        icc=icc,
        design_effect=float(design_effect),
        n_effective=float(n_effective),
        power_at_effective_n=_power_at(dz, n_effective),
        cluster_mean_p=cluster_p,
    )


def diagnostics(
    differences: pd.Series | np.ndarray, clusters: pd.Series | np.ndarray
) -> dict[str, object]:
    """Both diagnostics as one JSON-serializable block.

    Args:
        differences: One paired difference per target.
        clusters: Cluster label per target.

    Returns:
        Dict with `effect` and `clustering` sub-blocks, for embedding in a
        stats JSON alongside the p-value it qualifies.
    """
    return {
        "effect": asdict(paired_effect(differences)),
        "clustering": asdict(cluster_dependence(differences, clusters)),
    }
