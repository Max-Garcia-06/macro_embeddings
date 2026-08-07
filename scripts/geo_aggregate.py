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
| **B** | **re-derived** (since 2026-08-05) | `ingest_source_b.py` now carries `emp_{naics2}` alongside `lq_emp_{naics2}`, so a group LQ is summed sector employment over summed total employment against the national base -- the actual definition, rather than a mean of quotients. |
| **F** | **approximated** | USDA typology flags are categorical classifications with no underlying quantity to re-sum. A population-weighted share is a reasonable reading but a different construct. |
| **D** (HHI) | **re-derived** (since 2026-08-07) | `ingest_source_d.py` now writes `source_d_partners.parquet`, the partner-tons distribution the index is built from, so a group HHI is re-summed over partner *groups* rather than averaged from county HHIs. |

**Source B's re-derivation carries one known bias.** BLS suppresses a cell when
small employer counts could expose an individual operation, and a suppressed cell
has no employment level to sum. Group employment therefore undercounts by
whatever the suppressed counties held. Suppression concentrates in small-employer
counties, so the undercount is small relative to a market total -- but it is a
downward bias on the sectors BLS suppresses most (`lq_emp_21` at 67.4%,
`lq_emp_99` at 65.8%), and any market-level LQ for those sectors should be read
with that in mind.

The national base is computed by summing the county rows in this same file rather
than taken from BLS's published national totals, so it inherits the same
suppression and excludes non-county areas. That makes it internally consistent
across groups, which is what the comparison needs, and slightly different from
BLS's own denominator.

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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from county_population import fetch_county_population
from ingest_source_b import NAICS2_CODES
from pillar_matrix import SIZE_FEATURES, build_matrix

RANDOM_SEED: int = 42

DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
SOURCE_B_PATH: Path = DATA_DIR / "source_b_qcew.parquet"
SOURCE_E_PATH: Path = DATA_DIR / "source_e_irs_soi.parquet"
SOURCE_D_PARTNERS_PATH: Path = DATA_DIR / "source_d_partners.parquet"

# Raw Source E totals the ratio re-derivation sums before dividing.
SOURCE_E_TOTALS: tuple[str, ...] = (
    "num_returns",
    "wages_salaries_thousands",
    "qualified_dividends_thousands",
    "net_cap_gain_thousands",
    "n_returns_wages",
    "n_returns_qualified_dividends",
    "n_returns_net_cap_gain",
)

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


def national_sector_shares(employment: pd.DataFrame) -> dict[str, float]:
    """Compute each sector's share of national private employment.

    The denominator of a location quotient. Summed from the same county rows the
    groups are built from, so the base is internally consistent across groups --
    see the module docstring on how that differs from BLS's own denominator.

    Args:
        employment: Frame carrying `emp_{naics2}` columns.

    Returns:
        Mapping of NAICS-2 code to its national employment share.
    """
    totals = {
        code: float(employment[f"emp_{code}"].sum(skipna=True))
        for code in NAICS2_CODES
        if f"emp_{code}" in employment.columns
    }
    national = float(employment["emp_total_private"].sum(skipna=True))
    return {code: value / national for code, value in totals.items()} if national > 0 else {}


def _rederive_source_b(group: pd.DataFrame, shares: dict[str, float]) -> dict[str, float]:
    """Recompute Source B's location quotients from summed employment.

    A group's LQ for a sector is its share of the group's private employment
    divided by that sector's national share. This is the definition; the mean of
    the member counties' quotients is a different and meaningless quantity.

    Args:
        group: County rows for one group, carrying `emp_{naics2}` columns.
        shares: National sector shares from `national_sector_shares`.

    Returns:
        Mapping of `lq_emp_{naics2}` to its re-derived group value.
    """
    sector_totals = {
        code: float(group[f"emp_{code}"].sum(skipna=True))
        for code in NAICS2_CODES
        if f"emp_{code}" in group.columns
    }
    group_total = float(group["emp_total_private"].sum(skipna=True))
    if group_total <= 0:
        return {f"lq_emp_{code}": np.nan for code in sector_totals}

    derived: dict[str, float] = {}
    for code, total in sector_totals.items():
        national_share = shares.get(code, 0.0)
        derived[f"lq_emp_{code}"] = (
            (total / group_total) / national_share if national_share > 0 else np.nan
        )
    return derived


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


def rederive_partner_hhi(
    group_map: pd.DataFrame, group_column: str, partners: pd.DataFrame
) -> pd.DataFrame:
    """Recompute both partner-concentration indices at group grain.

    An HHI is a ratio of sums, so it cannot be averaged from county HHIs; it has
    to be rebuilt from the partner-tons distribution itself
    (`data/source_d_partners.parquet`, written by `ingest_source_d.py`). Three
    rules, each a judgement worth stating:

    - **Partner counties are remapped to their own group.** Concentration at
      market grain means concentration across *markets*, so two partner counties
      inside the same market are one partner, not two. Not remapping would leave
      the index measuring county spread while every other column measures the
      market.
    - **FAF-zone partners keep their own key.** A zone is not a county and cannot
      be assigned to a market; it stays a distinct partner, exactly as it is at
      county grain.
    - **Flows inside the group are dropped.** A shipment from one county in a
      market to another county in the same market is internal to that unit at
      this grain, not trade with a partner. Keeping it would inflate
      concentration for large multi-county markets specifically -- a size-
      correlated artifact in a column whose whole purpose is to not be one.

    Args:
        group_map: Frame with `fips_code` and the group column.
        group_column: Name of the group key column.
        partners: Long-format partner rows: `fips_code`, `direction`,
            `partner_key`, `tons`.

    Returns:
        Frame indexed by group with `out_partner_hhi` and `in_partner_hhi`.
    """
    county_group = group_map.set_index("fips_code")[group_column]
    frame = partners.copy()
    frame["home_group"] = frame["fips_code"].map(county_group)
    frame = frame[frame["home_group"].notna()]

    # Zone keys carry a "zone:" prefix and never match a FIPS code, so the map
    # leaves them untouched and `fillna` keeps them as their own partner.
    frame["partner_group"] = frame["partner_key"].map(county_group).fillna(frame["partner_key"])
    frame = frame[frame["partner_group"] != frame["home_group"]]

    pooled = frame.groupby(["home_group", "direction", "partner_group"])["tons"].sum()
    totals = pooled.groupby(level=["home_group", "direction"]).transform("sum")
    hhi = (
        (pooled / totals.where(totals > 0)).pow(2)
        .groupby(level=["home_group", "direction"])
        .sum()
        .unstack("direction")
    )
    hhi.columns = [f"{direction}_partner_hhi" for direction in hhi.columns]
    hhi.index.name = group_column
    return hhi


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
    raw_e = pd.read_parquet(SOURCE_E_PATH)[["fips_code", *SOURCE_E_TOTALS]]

    raw_b = pd.read_parquet(SOURCE_B_PATH)
    employment_columns = [f"emp_{code}" for code in NAICS2_CODES if f"emp_{code}" in raw_b.columns]
    # The total-private row is the LQ denominator; without it the sector sums
    # undercount by every suppressed cell and inflate the quotient.
    has_employment = bool(employment_columns) and "emp_total_private" in raw_b.columns
    if has_employment:
        employment_columns.append("emp_total_private")
    if not has_employment:
        logger.warning(
            "source_b_qcew.parquet carries no emp_* columns; Source B will be "
            "population-weighted rather than re-derived. Re-run ingest_source_b.py."
        )

    panel = (
        matrix.merge(group_map[["fips_code", group_column]], on="fips_code", how="inner")
        .merge(population, on="fips_code", how="inner")
        .merge(raw_e, on="fips_code", how="left", suffixes=("", "_raw"))
    )
    if has_employment:
        panel = panel.merge(
            raw_b[["fips_code", *employment_columns]], on="fips_code", how="left"
        )
    shares = national_sector_shares(panel) if has_employment else {}

    has_partners = SOURCE_D_PARTNERS_PATH.exists()
    partner_hhi = (
        rederive_partner_hhi(group_map, group_column, pd.read_parquet(SOURCE_D_PARTNERS_PATH))
        if has_partners
        else pd.DataFrame()
    )

    provenance: dict[str, str] = {}
    for pillar, columns in blocks.items():
        for column in columns:
            if pillar in {"A", "E"}:
                provenance[column] = "re-derived"
            elif pillar == "D":
                provenance[column] = (
                    "re-derived"
                    if "hhi" not in column or has_partners
                    else "approximated"
                )
            elif pillar == "B":
                provenance[column] = (
                    "re-derived"
                    if has_employment and column.startswith("lq_emp_")
                    else "approximated"
                )
            elif pillar == "F":
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
        if has_partners and key in partner_hhi.index:
            row.update(partner_hhi.loc[key].to_dict())
        if has_employment:
            row.update(_rederive_source_b(group, shares))
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
    if not SOURCE_D_PARTNERS_PATH.exists():
        logger.info(
            "Source D's two HHIs are approximated: %s is absent. Re-run "
            "ingest_source_d.py to write it.",
            SOURCE_D_PARTNERS_PATH.name,
        )


if __name__ == "__main__":
    main()
