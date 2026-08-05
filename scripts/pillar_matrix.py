"""Assemble the six E_macro pillars into one county-keyed feature matrix.

This is the candidate `E_macro` feature set as a downstream model would receive
it -- every pillar's columns side by side on `fips_code`, no fusion, no
dimensionality reduction, nulls preserved. `analyze_pillar_matrix_signal.py`
consumes it; the fusion step (which does not exist yet) eventually will too.

Two conventions matter for anything built on top of this module:

- **Nulls are real.** BLS suppresses ~35% of the county x sector LQ cells and
  those stay NaN rather than being zero-filled, because "no disclosed
  employment" and "no employment" are different facts. Imputation is the
  consumer's decision, made per model, not baked in here.
- **Size is not a pillar.** The pillar-pair sweep established that county size
  drives most of the raw cross-pillar correlation, so the pure scale measures
  are held out of every pillar block in `SIZE_COLUMNS` and the control handed to
  each baseline is the three-column `SIZE_FEATURES`. None of the held-out
  columns is any pillar's headline signal, so removing them costs no pillar its
  case. Freight tonnage is deliberately *not* treated as size: it correlates
  with scale, but it is the whole of Source D's signal, and folding it into the
  control would decide Source D's verdict by construction rather than by
  evidence. Source D instead gains the commodity *shares* its raw per-commodity
  tonnages were standing in for.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from county_population import load_size_proxy

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"

# Source F's six demographic risk flags, summed into `distress_count`. Matches
# the definition used by analyze_pillar_pair_crossvalidation.py and the key
# findings notebook so effect sizes stay comparable across rounds.
DISTRESS_FLAGS: tuple[str, ...] = (
    "low_postsecondary_ed",
    "low_employment",
    "population_loss",
    "housing_stress",
    "retirement_destination",
    "persistent_poverty",
)

# Source D ships raw tonnage by commodity group; these are the two directional
# totals that get log-scaled, since county tonnage spans six orders of magnitude.
D_TONNAGE_COLUMNS: tuple[str, ...] = ("total_outbound_tons", "total_inbound_tons")

# FAF5 commodity supergroups, as shipped by `ingest_source_d.py`.
D_COMMODITY_GROUPS: tuple[str, ...] = (
    "sctg0109", "sctg1014", "sctg1519", "sctg2033", "sctg3499",
)

# The ten raw per-commodity tonnages, held out of D's scored block as size
# columns. They are levels wearing a commodity label -- 0.52-0.97 Spearman
# against population (`source-d-findings.md` §9) -- and §11 kept them in the
# block only because a count-shaped downstream target would have made them
# legitimately predictive. That target is a rate (see `docs/downstream_target.md`
# Part 1), so the exclusion rule already applied to Source E's dollar totals
# applies here too. Their composition counterparts, the ten `share_*` columns
# derived in `_derive_pillar_columns`, are the size-free version and stay.
D_COMMODITY_TONNAGE_COLUMNS: tuple[str, ...] = tuple(
    f"{direction}_{group}" for direction in ("out", "in") for group in D_COMMODITY_GROUPS
)

# The 20 primary 2-digit NAICS sectors Source B ships. Mirrored here rather than
# imported from `ingest_source_b` so the matrix does not depend on an ingestion
# module; the two are checked against each other by the column-collision guard
# in `build_matrix`.
B_NAICS2_CODES: tuple[str, ...] = (
    "11", "21", "22", "23", "31-33", "42", "44-45", "48-49", "51", "52",
    "53", "54", "55", "56", "61", "62", "71", "72", "81", "99",
)

# No `tons_per_capita` column, deliberately. `docs/downstream_target.md` proposes
# it as the highest-value change a rate target would imply, and it is not one:
#
#   log10(tons / population) == log_total_tons - log_population
#
# to floating-point precision, and both of those already sit in the matrix --
# `log_total_tons` in D's block, `log_population` in SIZE_FEATURES, held in every
# baseline. The per-capita column is an exact linear combination of two columns
# any model here already has, so it adds nothing a linear model cannot already
# reach. Measured rather than assumed: D freight against F metro status is
# -0.036 size-controlled whether the input is raw log tonnage or per-capita
# tonnage, to three decimals (`source-d-findings.md` §10).
#
# It would matter for a consumer fitting *without* a size control, which is a
# serving-format question rather than a feature question.

# Pure scale measures, pulled out of their pillars and re-exposed log-scaled as
# SIZE_FEATURES. `wages_salaries_thousands` joins them because it is a dollar
# total rather than a composition -- Source E's actual signal is the
# capital-to-wage *ratio*, which stays in E's block.
SIZE_COLUMNS: tuple[str, ...] = (
    "num_returns",
    "agi_thousands",
    "wages_salaries_thousands",
    "gdp_latest",
    # The other two Source E dollar totals, excluded on the rule already stated
    # above and previously applied to only one of the three: in logs they run
    # r = 0.894 and r = 0.875 against log population, and r = 0.928 / 0.912
    # against `log_agi` -- which is itself derived from Source E's
    # `agi_thousands`. Leaving them in E's block put two near-copies of the size
    # control inside a block scored against that control. For comparison, Source
    # A's `n_body_sections` was cut from its scored block at r = 0.550.
    "qualified_dividends_thousands",
    "net_cap_gain_thousands",
    # Counts of returns reporting each income type: scale measures on the same
    # footing as `num_returns`. Their composition counterparts --
    # `capgain_participation_rate`, `dividend_participation_rate` -- are the
    # size-free version and stay in E's block.
    "n_returns_wages",
    "n_returns_qualified_dividends",
    "n_returns_net_cap_gain",
    # Source D's ten raw per-commodity tonnages, on the same rule. See
    # `D_COMMODITY_TONNAGE_COLUMNS` for why they moved here on 2026-08-05.
    *D_COMMODITY_TONNAGE_COLUMNS,
    # Source B's employment levels, added to the parquet 2026-08-05 so the
    # pillar can be re-derived at a coarser geography (`geo_aggregate.py`). They
    # are raw scale measures -- a county's employment in a sector is a level, and
    # the location quotient built from it is the composition that pillar ships.
    # Held out on exactly the rule that holds out Source E's dollar totals.
    "emp_total_private",
    *(f"emp_{code}" for code in B_NAICS2_CODES),
)

# The size control handed to every baseline model, alongside state dummies.
# `log_population` (Census PEP, `county_population.py`) replaced `log_num_returns`
# on 2026-08-04: every other member of this tuple is still a pillar's own column
# re-exposed, so anchoring the control on a source outside all six pillars keeps
# the baseline from being partly Source E measuring itself. The two agree at
# r = 0.998 in logs.
SIZE_FEATURES: tuple[str, ...] = (
    "log_population",
    "log_agi",
    "log_gdp_latest",
)

# Source A columns written for diagnosis rather than prediction, and therefore
# kept out of the matrix entirely.
#
# `has_usda_echo` exists to *detect* a contamination: it marks the counties whose
# Wikipedia intro quotes USDA's own classification back ("considered a
# high-recreation retirement destination by the U.S. Department of
# Agriculture"). Source F's `distress_count` is built from those
# classifications, so letting this column predict it would credit Source A for
# reciting a label it copied. A detector for circularity must not itself be a
# predictor.
#
# `n_body_sections` is a size proxy in disguise -- r = 0.550 against log tax
# returns, above `content_length`'s 0.359 -- and an ablation showed it carries
# 2.4% of its block's lift. Excluded on the same grounds it was excluded from
# Source A's own scored variants.
SOURCE_A_DIAGNOSTIC_COLUMNS: tuple[str, ...] = ("has_usda_echo", "n_body_sections")

# Source E columns that exist for interpretation rather than prediction.
#
# `national_capital_to_wage_ratio` is the same value on every row -- it is the
# divisor behind the normalized columns, carried so the normalization stays
# invertible without refetching. Zero variance, no information.
#
# `capital_to_wage_ratio_normalized` is `capital_to_wage_ratio` divided by that
# constant, so the two are perfectly collinear within a vintage. The cross-year
# columns built from it (`..._normalized_mean`, `..._normalized_std`) are not,
# and stay in the block.
#
# `n_tax_years_observed` records boundary changes -- Connecticut's planning
# regions, the Valdez-Cordova split -- not economics.
SOURCE_E_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "national_capital_to_wage_ratio",
    "capital_to_wage_ratio_normalized",
    "n_tax_years_observed",
)

# Identifier and bookkeeping columns that are never features.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "fips_code",
    "county_name",
    "unemployment_latest_year",
    "gdp_latest_year",
    "as_of_date",
)

logger = logging.getLogger(__name__)


def _load_pillar_frames() -> dict[str, pd.DataFrame]:
    """Read all six pillar parquets from `data/`.

    Source A is read from `source_a_text_features.parquet`; the legacy
    `source_a_embeddings.parquet` is not part of the matrix, since the bge-m3
    step was cut from that pipeline.

    Returns:
        Mapping of pillar letter "A".."F" to its raw DataFrame.

    Raises:
        FileNotFoundError: If any source parquet is absent.
    """
    filenames = {
        "A": "source_a_text_features.parquet",
        "B": "source_b_qcew.parquet",
        "C": "source_c_fred.parquet",
        "D": "source_d_faf.parquet",
        "E": "source_e_irs_soi.parquet",
        "F": "source_f_usda_typology.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for pillar, filename in filenames.items():
        path = DATA_DIR / filename
        try:
            frames[pillar] = pd.read_parquet(path)
        except FileNotFoundError:
            logger.error("Missing pillar parquet for Source %s: %s", pillar, path)
            raise
    return frames


def _derive_pillar_columns(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add the derived columns each pillar contributes, and drop non-features.

    Source A drops its two raw-text columns plus its diagnostics. Source D gains
    log-scaled tonnage. Source F gains `distress_count`.

    Args:
        frames: Raw pillar frames from `_load_pillar_frames`.

    Returns:
        New mapping with the same keys; input frames are not mutated.
    """
    derived = dict(frames)

    derived["A"] = frames["A"].drop(
        columns=["raw_intro_text", "embedding_text", *SOURCE_A_DIAGNOSTIC_COLUMNS],
        errors="ignore",
    )

    derived["E"] = frames["E"].drop(columns=list(SOURCE_E_DIAGNOSTIC_COLUMNS), errors="ignore")

    d = frames["D"]
    total_tons = d[list(D_TONNAGE_COLUMNS)].sum(axis=1)
    # Commodity mix as a share of the county's own directional total. The raw
    # per-commodity tonnages run 0.52-0.97 Spearman against population -- they
    # are levels wearing a commodity label. The shares are the composition those
    # levels were standing in for, and four of the ten clear the size-free bar
    # the raw columns all fail (`source-d-findings.md` §9).
    commodity_shares = {
        f"share_{direction}_{group}": (
            d[f"{direction}_{group}"] / d[f"total_{'outbound' if direction == 'out' else 'inbound'}_tons"]
        )
        for direction in ("out", "in")
        for group in D_COMMODITY_GROUPS
    }
    derived["D"] = d.assign(
        log_total_tons=np.log10(total_tons.clip(lower=1)),
        log_outbound_tons=np.log10(d["total_outbound_tons"].clip(lower=1)),
        log_inbound_tons=np.log10(d["total_inbound_tons"].clip(lower=1)),
        **commodity_shares,
    ).drop(columns=list(D_TONNAGE_COLUMNS))

    f = frames["F"]
    derived["F"] = f.assign(distress_count=f[list(DISTRESS_FLAGS)].astype("float").sum(axis=1))

    return derived


def build_matrix() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Join all six pillars into one county-keyed feature matrix.

    Every pillar column is prefix-free and globally unique, so blocks are
    tracked by name rather than by prefix. Boolean columns are cast to float so
    downstream estimators see a uniform numeric dtype; nulls are preserved
    throughout.

    Returns:
        Tuple of (matrix, blocks). `matrix` carries `fips_code`, `county_name`,
        `state_fips`, the `SIZE_FEATURES` columns, and every pillar feature
        column.
        `blocks` maps pillar letter to its list of feature column names,
        excluding size and identifier columns.

    Raises:
        FileNotFoundError: If any source parquet is absent.
        ValueError: If two pillars contribute a column of the same name.
    """
    frames = _derive_pillar_columns(_load_pillar_frames())

    matrix = frames["F"][["fips_code", "county_name"]].copy()
    blocks: dict[str, list[str]] = {}
    seen: set[str] = set()

    for pillar in "ABCDEF":
        frame = frames[pillar]
        feature_columns = [
            col
            for col in frame.columns
            if col not in NON_FEATURE_COLUMNS and col not in SIZE_COLUMNS
        ]
        collisions = seen.intersection(feature_columns)
        if collisions:
            raise ValueError(f"Source {pillar} repeats column name(s): {sorted(collisions)}")
        seen.update(feature_columns)

        blocks[pillar] = feature_columns
        matrix = matrix.merge(
            frame[["fips_code", *feature_columns]], on="fips_code", how="left"
        )

    # Scale measures, re-attached outside every block so they can control
    # without also predicting.
    matrix = (
        matrix.merge(
            frames["E"][["fips_code", "agi_thousands"]], on="fips_code", how="left"
        )
        .merge(frames["C"][["fips_code", "gdp_latest"]], on="fips_code", how="left")
        .merge(load_size_proxy(), on="fips_code", how="left")
    )
    matrix["log_agi"] = np.log10(matrix["agi_thousands"].clip(lower=1))
    matrix["log_gdp_latest"] = np.log10(matrix["gdp_latest"].clip(lower=1))
    matrix = matrix.drop(columns=["agi_thousands", "gdp_latest"])

    # State FIPS is the first two digits of the county code -- the geography
    # control, so "signal" cannot just be one state's counties resembling each
    # other.
    matrix["state_fips"] = matrix["fips_code"].astype(str).str.zfill(5).str[:2]

    boolean_columns = matrix.select_dtypes(include=["bool"]).columns
    matrix[boolean_columns] = matrix[boolean_columns].astype("float")

    logger.info(
        "matrix: %d counties x %d pillar features (%s)",
        len(matrix),
        sum(len(cols) for cols in blocks.values()),
        ", ".join(f"{p}={len(c)}" for p, c in blocks.items()),
    )
    return matrix, blocks


def other_pillar_columns(blocks: dict[str, list[str]], pillar: str) -> list[str]:
    """List every feature column that does not belong to `pillar`.

    Args:
        blocks: Pillar-to-columns mapping from `build_matrix`.
        pillar: Pillar letter to exclude.

    Returns:
        Feature column names drawn from all other pillars, in pillar order.
    """
    return [col for key, cols in blocks.items() if key != pillar for col in cols]
