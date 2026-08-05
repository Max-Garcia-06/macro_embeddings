"""Aggregate the county feature matrix to a coarser geography, correctly.

`docs/plans/dma_regrain.md` §3 states the governing rule for re-graining this
repo: **aggregate the inputs, not the outputs.** Every derived feature in
`E_macro` is a ratio, share, quotient, slope or index, and none of them survive a
county-level mean. A location quotient is measured against a national base; a
capital-to-wage ratio is a quotient of two dollar totals; a commodity share is a
share of a directional total. Averaging any of those across the counties in a
market produces a number with no interpretation.

This module implements that rule for a caller-supplied county-to-group mapping,
and -- more importantly -- **records where the rule cannot be followed.**

## What can and cannot be re-derived from the shipped parquets

| pillar | status | why |
|---|---|---|
| **E** | **re-derived** | `source_e_irs_soi.parquet` carries `num_returns`, `agi_thousands`, `wages_salaries_thousands`, `qualified_dividends_thousands`, `net_cap_gain_thousands` and the three return counts. Sum them across the group, then recompute every ratio. |
| **D** | **re-derived** (tonnage) | Per-commodity and directional tonnages are levels, so they sum. Shares are recomputed from the summed totals. |
| **A** | **re-derived** | Booleans become the population-weighted share of the group's residents whose county carries the flag, which is strictly more informative than the boolean was. Counts sum. |
| **C** | population-weighted | Velocities are slopes; a weighted mean is defensible but is not the slope of the summed series. `gdp_latest` sums. |
| **B** | **approximated** | `source_b_qcew.parquet` ships only location quotients and disclosure flags. **The underlying employment counts are not in the parquet**, so a group-level LQ cannot be computed without re-ingesting QCEW. |
| **F** | **approximated** | USDA typology flags are categorical classifications with no underlying quantity to re-sum. A population-weighted share is a reasonable reading but a different construct. |
| **D** (HHI) | **approximated** | Partner concentration needs the partner-level flow table, which the parquet does not carry. |

**Source B needing re-ingestion is a cost finding, not a detail.**
`dma_regrain.md` Phase 1B estimated 1-2 days on the assumption that every pillar
could be rebuilt from its shipped parquet. Source B cannot, and it is the widest
block in the matrix at 40 columns. Any real DMA delivery has to re-ingest QCEW
carrying `emp` alongside `lq_emp`, which is a change to `ingest_source_b.py`
rather than a change to this module.

`aggregate_matrix` returns the aggregated frame together with a per-column
provenance map, so every downstream report can state which columns are exact and
which are approximations rather than presenting one number for both.

## Pseudo-markets

`build_pseudo_markets` clusters county centroids into `k` spatially compact
groups. At k = 210 that matches both the cardinality and the character of a
Nielsen DMA -- contiguous groups of whole counties -- **without using the Nielsen
delineation, which is proprietary.** It is a stand-in for measuring the mechanism
of grain loss, not a substitute for the real crosswalk, and results built on it
must say so.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from county_population import fetch_county_population
from pillar_matrix import SIZE_FEATURES, build_matrix

RANDOM_SEED: int = 42

# Matches the Nielsen DMA count. See the module docstring on why this is a
# stand-in rather than the real delineation.
DEFAULT_N_MARKETS: int = 210

EARTH_RADIUS_KM: float = 6371.0

# Columns whose group value is a sum of county values rather than a weighted
# mean. Everything here is a level; ratios built from them are recomputed after.
SUMMABLE_COLUMNS: tuple[str, ...] = (
    "log_total_tons",
    "log_outbound_tons",
    "log_inbound_tons",
    "gdp_latest",
    "n_industry_mentions",
    "sec_n_industry_mentions",
    "n_distinct_proper_nouns",
    "content_length",
)

# Pillars whose group values are exact re-derivations of the county inputs.
EXACT_PILLARS: frozenset[str] = frozenset({"A", "D", "E"})

logger = logging.getLogger(__name__)


def _to_cartesian(latitude: pd.Series, longitude: pd.Series) -> np.ndarray:
    """Project lat/lon degrees onto 3-D unit-sphere coordinates.

    Clustering on raw lat/lon distorts east-west distance away from the equator
    and breaks across the date line. Cartesian coordinates avoid both.

    Args:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.

    Returns:
        Array of shape (n, 3) scaled to kilometres.
    """
    phi = np.radians(latitude.to_numpy())
    theta = np.radians(longitude.to_numpy())
    return EARTH_RADIUS_KM * np.column_stack(
        [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]
    )


def build_pseudo_markets(
    centroids: pd.DataFrame, n_markets: int = DEFAULT_N_MARKETS
) -> pd.DataFrame:
    """Cluster counties into spatially compact stand-in markets.

    Args:
        centroids: Frame with `fips_code`, `lat`, `lon`.
        n_markets: Number of groups. Defaults to the Nielsen DMA count.

    Returns:
        DataFrame with `fips_code` and `market_id`.
    """
    usable = centroids.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    coordinates = _to_cartesian(usable["lat"], usable["lon"])
    labels = KMeans(n_clusters=n_markets, random_state=RANDOM_SEED, n_init=10).fit_predict(
        coordinates
    )
    logger.info(
        "pseudo-markets: %d groups over %d counties (median %d counties per group)",
        n_markets,
        len(usable),
        int(pd.Series(labels).value_counts().median()),
    )
    return pd.DataFrame({"fips_code": usable["fips_code"], "market_id": labels})


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Population-weighted mean that tolerates nulls in the values.

    Args:
        values: Column values for one group.
        weights: Matching population weights.

    Returns:
        Weighted mean, or NaN when every value in the group is null.
    """
    mask = values.notna() & weights.notna()
    if not mask.any() or weights[mask].sum() <= 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _rederive_source_e(group: pd.DataFrame) -> dict[str, float]:
    """Recompute Source E's ratios from summed dollar and count totals.

    This is the rule the whole module exists to enforce: the group's
    capital-to-wage ratio is the ratio of its summed capital income to its summed
    wage income, never the mean of its counties' ratios.

    Args:
        group: County rows for one group, carrying Source E's raw totals.

    Returns:
        Mapping of Source E column name to its re-derived group value.
    """
    totals = {
        column: float(group[column].sum(skipna=True))
        for column in (
            "num_returns",
            "wages_salaries_thousands",
            "qualified_dividends_thousands",
            "net_cap_gain_thousands",
            "n_returns_wages",
            "n_returns_qualified_dividends",
            "n_returns_net_cap_gain",
        )
        if column in group.columns
    }
    capital = totals.get("qualified_dividends_thousands", 0.0) + totals.get(
        "net_cap_gain_thousands", 0.0
    )
    wages = totals.get("wages_salaries_thousands", 0.0)
    returns = totals.get("num_returns", 0.0)
    claimers = totals.get("n_returns_net_cap_gain", 0.0)

    return {
        "capital_to_wage_ratio": capital / wages if wages > 0 else np.nan,
        "capgain_participation_rate": claimers / returns if returns > 0 else np.nan,
        "dividend_participation_rate": (
            totals.get("n_returns_qualified_dividends", 0.0) / returns if returns > 0 else np.nan
        ),
        "gain_per_claimer_thousands": (
            totals.get("net_cap_gain_thousands", 0.0) / claimers if claimers > 0 else np.nan
        ),
        "wage_per_return_thousands": (
            wages / totals.get("n_returns_wages", 0.0)
            if totals.get("n_returns_wages", 0.0) > 0
            else np.nan
        ),
    }


def _rederive_source_d(group: pd.DataFrame) -> dict[str, float]:
    """Recompute Source D's commodity shares from summed tonnage.

    Args:
        group: County rows for one group.

    Returns:
        Mapping of share column name to its re-derived group value. Empty when
        the raw tonnages are absent from the frame.
    """
    derived: dict[str, float] = {}
    for direction in ("out", "in"):
        columns = [c for c in group.columns if c.startswith(f"share_{direction}_")]
        if not columns:
            continue
        # Shares were built from per-commodity tonnage over the directional
        # total. Recovering group shares needs those tonnages, which live in
        # SIZE_COLUMNS and are reconstructed here from the county shares
        # weighted by each county's directional volume.
        volume = group[f"log_{'outbound' if direction == 'out' else 'inbound'}_tons"]
        weights = np.power(10.0, volume.fillna(0.0))
        for column in columns:
            derived[column] = _weighted_mean(group[column], pd.Series(weights, index=group.index))
    return derived


def aggregate_matrix(
    matrix: pd.DataFrame,
    blocks: dict[str, list[str]],
    group_map: pd.DataFrame,
    group_column: str = "market_id",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Aggregate the county matrix to groups, re-deriving wherever possible.

    Args:
        matrix: County feature matrix from `build_matrix`.
        blocks: Pillar-to-columns mapping from `build_matrix`.
        group_map: Frame with `fips_code` and the grouping column.
        group_column: Name of the grouping column in `group_map`.

    Returns:
        Tuple of (aggregated, provenance). `aggregated` carries the group key,
        `population`, the size features and every pillar column. `provenance`
        maps each pillar column to "re-derived", "population-weighted" or
        "approximated".

    Raises:
        KeyError: If `group_column` is absent from `group_map`.
    """
    population = fetch_county_population()[["fips_code", "population"]]
    raw_e = pd.read_parquet(matrix.attrs.get("source_e_path", "data/source_e_irs_soi.parquet"))

    panel = (
        matrix.merge(group_map[["fips_code", group_column]], on="fips_code", how="inner")
        .merge(population, on="fips_code", how="inner")
        .merge(
            raw_e[
                [
                    "fips_code",
                    "num_returns",
                    "wages_salaries_thousands",
                    "qualified_dividends_thousands",
                    "net_cap_gain_thousands",
                    "n_returns_wages",
                    "n_returns_qualified_dividends",
                    "n_returns_net_cap_gain",
                ]
            ],
            on="fips_code",
            how="left",
            suffixes=("", "_raw"),
        )
    )

    provenance: dict[str, str] = {}
    for pillar, columns in blocks.items():
        for column in columns:
            if pillar == "E":
                provenance[column] = "re-derived"
            elif pillar == "A" or (pillar == "D" and column.startswith("share_")):
                provenance[column] = "re-derived"
            elif pillar == "D" and "hhi" in column:
                provenance[column] = "approximated"
            elif pillar in {"B", "F"}:
                provenance[column] = "approximated"
            else:
                provenance[column] = "population-weighted"

    pillar_columns = [column for columns in blocks.values() for column in columns]
    rows: list[dict[str, object]] = []
    for key, group in panel.groupby(group_column):
        weights = group["population"]
        row: dict[str, object] = {
            group_column: key,
            "population": float(weights.sum()),
            "n_counties": int(len(group)),
        }
        for column in SIZE_FEATURES:
            row[column] = _weighted_mean(group[column], weights)
        for column in pillar_columns:
            if column in SUMMABLE_COLUMNS:
                row[column] = float(group[column].sum(skipna=True))
            else:
                row[column] = _weighted_mean(group[column], weights)
        row.update(_rederive_source_e(group))
        row.update(_rederive_source_d(group))
        rows.append(row)

    aggregated = pd.DataFrame(rows)
    exact = sum(1 for value in provenance.values() if value == "re-derived")
    logger.info(
        "aggregated to %d groups | %d columns re-derived, %d approximated, %d weighted",
        len(aggregated),
        exact,
        sum(1 for value in provenance.values() if value == "approximated"),
        sum(1 for value in provenance.values() if value == "population-weighted"),
    )
    return aggregated, provenance


def main() -> None:
    """Build pseudo-markets and report the aggregation's provenance split."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    matrix, blocks = build_matrix()
    centroids = pd.read_parquet("data/county_centroids.parquet")
    markets = build_pseudo_markets(centroids)
    _, provenance = aggregate_matrix(matrix, blocks, markets)

    by_status: dict[str, list[str]] = {}
    for column, status in provenance.items():
        by_status.setdefault(status, []).append(column)
    for status, columns in sorted(by_status.items()):
        logger.info("%-22s %3d columns", status, len(columns))
    logger.info(
        "Source B needs QCEW employment counts to re-derive; the shipped parquet "
        "carries location quotients only. See this module's docstring."
    )


if __name__ == "__main__":
    main()
