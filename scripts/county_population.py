"""County population from the Census Population Estimates Program.

Every size control in this repo -- the partial-correlation control in
`analyze_pillar_pair_crossvalidation.py` and the dependence scan in
`analyze_feature_size_dependence.py` -- originally used `num_returns`, the
count of filed tax returns, as its proxy for how big a county is. That column
belongs to Source E, so Source E's own independence score was partly
self-referential, and `docs/PROJECT_GOAL.md` next-work item 2 records replacing
it as a prerequisite for trusting the size tables in full.

This module supplies the replacement: Census PEP county population totals,
which belong to no pillar. The two proxies agree closely (log-log r = 0.998,
logged on first use by `load_size_proxy`), so the swap removes the
self-reference without moving the conclusions -- which is the outcome that makes
the size tables trustworthy rather than merely different.

`population` is the vintage-2024 estimate; the file also carries 2020-2023, and
`POPULATION_ESTIMATE_COLUMN` is the single place to change the vintage.

Output: `data/county_population.parquet`, cached after the first download.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
POPULATION_CACHE_PATH: Path = REPO_ROOT / "data" / "county_population.parquet"

POPULATION_URL: str = (
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/"
    "counties/totals/co-est2024-alldata.csv"
)
REQUEST_TIMEOUT_SECONDS: int = 60

# Vintage of the estimate used as the size proxy. The published file carries
# POPESTIMATE2020..2024; 2024 is the most recent and the closest match to the
# 2025-vintage QCEW and FRED pillars.
POPULATION_ESTIMATE_COLUMN: str = "POPESTIMATE2024"
POPULATION_VINTAGE: str = "Census PEP vintage 2024"

# SUMLEV 50 rows are counties; 40 rows are state totals and are dropped.
COUNTY_SUMMARY_LEVEL: int = 50

logger = logging.getLogger(__name__)


def fetch_county_population(cache_path: Path = POPULATION_CACHE_PATH) -> pd.DataFrame:
    """Load county population totals, downloading and caching on first use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with columns `fips_code`, `county_name`, `population`.

    Raises:
        requests.HTTPError: If the Census download fails.
    """
    if cache_path.exists():
        logger.info("Loading cached county population from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading Census population estimates...")
    response = requests.get(POPULATION_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    raw = pd.read_csv(
        io.BytesIO(response.content),
        encoding="latin-1",
        dtype={"STATE": str, "COUNTY": str},
    )
    counties = raw[raw["SUMLEV"] == COUNTY_SUMMARY_LEVEL].copy()
    counties["fips_code"] = (
        counties["STATE"].str.zfill(2) + counties["COUNTY"].str.zfill(3)
    )
    population = counties[["fips_code", "CTYNAME", POPULATION_ESTIMATE_COLUMN]].rename(
        columns={"CTYNAME": "county_name", POPULATION_ESTIMATE_COLUMN: "population"}
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    population.to_parquet(cache_path, engine="pyarrow", index=False)
    logger.info("Cached %d counties to %s", len(population), cache_path)
    return population


def load_size_proxy(cache_path: Path = POPULATION_CACHE_PATH) -> pd.DataFrame:
    """Return the county size proxy every size control in this repo should use.

    Args:
        cache_path: Local Parquet cache path.

    Returns:
        DataFrame with columns `fips_code` and `log_population`, the base-10 log
        of Census population. Logged because county population spans four orders
        of magnitude, matching the transform the sweep already applied to the
        tax-return proxy.
    """
    population = fetch_county_population(cache_path)
    return pd.DataFrame(
        {
            "fips_code": population["fips_code"],
            "log_population": np.log10(population["population"].clip(lower=1)),
        }
    )


def main() -> None:
    """Refresh the population cache and report agreement with the old proxy."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    size = load_size_proxy()
    logger.info("%s: %d counties", POPULATION_VINTAGE, len(size))

    returns_path = REPO_ROOT / "data" / "source_e_irs_soi.parquet"
    if not returns_path.exists():
        return
    returns = pd.read_parquet(returns_path)[["fips_code", "num_returns"]]
    paired = size.merge(returns, on="fips_code", how="inner").dropna()
    log_returns = np.log10(paired["num_returns"].clip(lower=1))
    r = float(np.corrcoef(paired["log_population"], log_returns)[0, 1])
    logger.info(
        "Agreement with the superseded tax-return proxy: r = %.4f on %d counties",
        r,
        len(paired),
    )


if __name__ == "__main__":
    main()
